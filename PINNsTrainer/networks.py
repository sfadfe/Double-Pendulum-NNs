import torch
import torch.nn as nn

"""
- 입력 feats (N, feat_dim): [τ, sinθ1, cosθ1, sinθ2, cosθ2, ω1, ω2, param_embed(32)]
  ParamEmbed는 Dataset._BuildFeats에서 호출 — raw m/L 대신 임베딩 벡터 사용 // option B
- time-marching flow map: τ는 윈도우 내 상대시간(τ∈[0, march_dt]), trig/ω는 윈도우 시작 상태(IC)
- τ는 Fourier Features(물리 주파수, Hz)로 매핑되고, 또한 [0, march_dt]에서 [-1, 1]로 정규화됨
- 출력: (N, 4) = [Δθ1, Δθ2, ω1, ω2]. Δθ = θ(τ) − θ_IC (상대각), ω는 절대값.
  공유 trunk(FiLMBlock 스택, 케이스는 FiLM (γ,β)로만 주입) + head_theta / head_omega 분리 // option B1
"""


class FourierFeatures(nn.Module):
    # Fourier features on t  // 시간 t에 대해서만 fourier features 적용
    def __init__(self, fourier_l, f_min, f_max):
        super().__init__()
        # f_k = f_min * (f_max/f_min)^(k/(L-1)), non-learnable NeRF Style // 고정 주파수 NeRF 방식 채택
        exponent = torch.linspace(0.0, 1.0, fourier_l)
        freqs = f_min * (f_max / f_min) ** exponent
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        proj = 2.0 * torch.pi * t * self.freqs
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ParamEmbed(nn.Module):
    # params_norm(4) → dim → dim, SiLU // option B3
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, params_norm):
        return self.net(params_norm)


class FiLMCond(nn.Module):
    # (IC, param_embed) → 블록별 (γ, β) // 케이스가 trunk를 곱셈 변조한다
    #   zero-init + γ에 1을 더해 시작 시 항등(γ=1, β=0) — 학습 초반이 concat과 같은 출발선.
    #   입력이 τ 무관이라 케이스당 1회만 계산하면 되지만, feats가 이미 (N=케이스×τ점)으로
    #   flatten된 채 들어오므로 여기서는 행마다 계산한다. 비용은 cond MLP 1회 = trunk의 1/6.
    def __init__(self, cond_dim, hidden, width, n_blocks):
        super().__init__()
        self.width = width
        self.n_blocks = n_blocks
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * width * n_blocks),
        )

    def forward(self, c):
        gb = self.net(c).view(-1, self.n_blocks, 2, self.width)
        gamma = 1.0 + gb[:, :, 0]
        beta = gb[:, :, 1]
        return gamma, beta


class FiLMBlock(nn.Module):
    # FiLM-inside residual block (2026-08-30, 확정 2026-08-31) // 변조가 잔차 가지 안에만 들어가는 블록
    #   h ← h + γ(c)·SiLU(W h + b) + β(c). skip 경로는 변조되지 않는다.
    #   구 gated 블록((1−z)u + zv + h 뒤 γ·h + β)은 skip까지 γ가 곱해져 깊이만큼 γ가 곱셈 증식했다.
    #   여기선 γ가 가지 하나에만 걸려 블록 야코비안이 I + O(γ)로 유계 → 좁고 깊게(256×12) 쌓는다.
    #   판정: FiLM-inside 256×12 > gated 384×6 (rollout 3.2×); τ 재주입(film_tau)은 3.9× 악화로 기각.
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.act = nn.SiLU()

    def forward(self, h, gamma, beta):
        return h + gamma * self.act(self.fc1(h)) + beta


class FFNBlock(FiLMBlock):
    # FiLMBlock with γ≡1, β≡0 (correction-unit screen 2026-09-14) // 조건화 없는 같은 블록, 연산 동일
    #   Same fc1/act/1/√n scale and state-dict keys as FiLMBlock, so arm A−C measures "conditioning removed" only.
    def forward(self, h):
        return h + self.act(self.fc1(h))


class AttnPoolBlock(nn.Module):
    # Attention pooling over sub-tokens of h (correction-unit screen 2026-09-14, arm E)
    #   h (N, width) → n_tokens tokens of d = width/n_tokens; one learned query q attends over K = W_k·t,
    #   pooled = Σ_j softmax_j(K_j·q/√d)·(W_v·t_j); h ← h + W_o·pooled with W_o zero-init → identity at init.
    #   Input-dependent mixing across the hidden vector, which FiLM (per-case affine) cannot do. Manual
    #   softmax/matmul: SDPA's jvp under CUDA+compile is unverified. No τ-grid/x̂ tokens (plan §어텐션 풀링).
    # // h를 8×32 토큰으로 쪼개 학습 쿼리 1개로 풀링, zero-init 출력으로 h에 더한다. 수동 softmax (jvp 안전)
    def __init__(self, width, n_tokens):
        super().__init__()
        if width % n_tokens != 0:
            raise ValueError(f"width {width} must be divisible by attn_tokens {n_tokens}")
        self.n_tokens, self.d = n_tokens, width // n_tokens
        self.key = nn.Linear(self.d, self.d)
        self.value = nn.Linear(self.d, self.d)
        self.query = nn.Parameter(torch.zeros(self.d))
        self.out = nn.Linear(self.d, width)

    def forward(self, h):
        t = h.view(-1, self.n_tokens, self.d)                          # (N, T, d)
        score = self.key(t) @ self.query / self.d ** 0.5               # (N, T)
        att = torch.softmax(score, dim=-1)
        pooled = (att.unsqueeze(-1) * self.value(t)).sum(dim=1)        # (N, d)
        return h + self.out(pooled)


class XAttnBlock(FiLMBlock):
    # Window-token cross-attention corrector block (2026-09-17, config.py xattn_tokens)
    #   Rows are case-major groups of M = n_q + n_tok: n_q query rows (any τ) then n_tok token rows (fixed τ grid,
    #   appended in Networks.forward). Every row of the group attends over the group's token rows only:
    #   h ← h + W_o·MHA(q(h), k(tok), v(tok)); then the FiLMBlock branch h ← h + γ·SiLU(W h + b) + β.
    #   Token rows are keys/values but never queries of other rows' sets, so a query row's output does not
    #   depend on which other queries share the batch (marching at τ=md alone is consistent with training).
    #   Manual softmax/einsum (SDPA jvp under compile unverified, cf. AttnPoolBlock). Token τ are constants →
    #   forward-mode tangent flows only through the query row's own path (StateDerivs jvp exact).
    # // 케이스 토큰(τ 격자의 예측기 출력 임베딩)에 대한 cross-attention + FiLM 잔차. 토큰은 K/V만
    def __init__(self, width, n_heads):
        super().__init__(width)
        if width % n_heads != 0:
            raise ValueError(f"width {width} must be divisible by xattn_heads {n_heads}")
        self.n_heads, self.d = n_heads, width // n_heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def forward(self, h, gamma, beta, n_q, n_tok, tok=None):
        # tok: precomputed token states (n, P, W) → h holds only the n_q query rows per case (jvp path);
        #   None → h holds n_q query rows + n_tok token rows per case. // tok 주어지면 h는 질의 행뿐
        width = h.shape[-1]
        if tok is None:
            m = n_q + n_tok
            hg = h.view(-1, m, width)                                    # (n, M, W)
            tok = hg[:, n_q:]                                            # (n, P, W)
        else:
            m = n_q
            hg = h.view(-1, m, width)
        n = hg.shape[0]
        q = self.q(hg).view(n, m, self.n_heads, self.d)
        k = self.k(tok).view(n, n_tok, self.n_heads, self.d)
        v = self.v(tok).view(n, n_tok, self.n_heads, self.d)
        score = torch.einsum("nmhd,nphd->nhmp", q, k) / self.d ** 0.5   # (n, H, M, P)
        att = torch.softmax(score, dim=-1)
        o = torch.einsum("nhmp,nphd->nmhd", att, v).reshape(-1, width)
        h = h + self.out(o)
        return h + gamma * self.act(self.fc1(h)) + beta


class Networks(nn.Module):
    def __init__(self, netCfg, dataCfg):
        super().__init__()
        self.march_dt = dataCfg.march_dt   # 상대시간 정규화 기준 // window duration for τ normalization
        self.hard_ic = netCfg.hard_ic      # "off" | "c0" | "c1" — τ=0 경계조건 부과 방식 (forward 말미)

        self.fourier = FourierFeatures(netCfg.fourier_l, netCfg.f_min, netCfg.f_max)
        self.param_embed = ParamEmbed(netCfg.param_embed_dim)

        # Trunk gets τ features only; the case (IC, param_embed) enters solely through FiLM (γ, β).
        # // trunk 입력은 τ뿐, 케이스는 FiLMCond → FiLMBlock 곱셈 변조로만 들어간다 (concat 경로 제거)
        gx = netCfg.trunk_in_dim
        width = netCfg.width
        self.proj_in = nn.Sequential(nn.Linear(gx, width), nn.SiLU())

        # Block types (correction-unit screen 2026-09-14): "film" | "ffn" | "attn" per block; () = all film.
        #   film_slot[i] = index into the FiLM (γ, β) stack for film blocks, -1 otherwise. // 블록 타입 열·FiLM 슬롯 맵
        self.block_types = list(netCfg.block_types) or ["film"] * netCfg.n
        if len(self.block_types) != netCfg.n or any(t not in ("film", "ffn", "attn", "xattn") for t in self.block_types):
            raise ValueError(f"block_types must be {netCfg.n} of film/ffn/attn/xattn: {self.block_types}")
        self.film_slot, n_film = [], 0
        for t in self.block_types:                     # xattn carries a FiLM branch → consumes a (γ, β) slot
            self.film_slot.append(n_film if t in ("film", "xattn") else -1)
            n_film += t in ("film", "xattn")
        # Window tokens (xattn): P token rows per case at τ_j = j·md/P, appended in forward. 0 = no xattn block
        # // xattn 블록이 있으면 케이스마다 τ 격자 토큰 P행을 forward에서 덧붙인다
        self.n_tok = netCfg.xattn_tokens if "xattn" in self.block_types else 0
        # Window-state corrector (config.py win_tokens): same token grid/plumbing, but the P token states are
        #   pooled into one z per case at the handoff instead of being attended to // 창 상태 보정기
        self.win_tok = netCfg.win_tokens
        if self.win_tok > 0:
            if self.n_tok > 0 or netCfg.stage_inj != "embed" or len(netCfg.stage_at) != 1:
                raise ValueError("win_tokens needs stage_inj 'embed', one stage and no xattn block")
            self.n_tok = self.win_tok
        if self.n_tok > 0:   # buffer only for xattn nets — keeps older checkpoints strict-loadable
            self.register_buffer("tau_tok", torch.arange(1, self.n_tok + 1, dtype=torch.float32)
                                 * (self.march_dt / self.n_tok))
        self.film_split_at = netCfg.film_split_at
        if self.film_split_at > 0:   # 예측 구간 / 보정 구간 FiLM MLP 분리 (동결 스크린)
            n_film_a = sum(1 for i in range(self.film_split_at) if self.block_types[i] in ("film", "xattn"))
            if not 1 <= self.film_split_at < netCfg.n or n_film_a == 0:
                raise ValueError(f"film_split_at {self.film_split_at} leaves the prediction-region FiLM empty")
        else:
            n_film_a = n_film
        if n_film_a == 0:
            raise ValueError("at least one film block is required")

        self.film_corr = None
        self.film = FiLMCond(netCfg.cond_dim, netCfg.film_hidden, width, n_film_a)
        if self.film_split_at > 0 and n_film > n_film_a:   # blocks ≥ split: own MLP, slots n_film_a.. (arm B: none)
            self.film_corr = FiLMCond(netCfg.cond_dim, netCfg.film_hidden, width, n_film - n_film_a)
        self.blocks = nn.ModuleList([self._MakeBlock(t, width, netCfg.attn_tokens, netCfg.xattn_heads)
                                     for t in self.block_types])
        self.head_theta = nn.Linear(width, 2)   # Δθ1, Δθ2 // option B1
        self.head_omega = nn.Linear(width, 2)   # ω1, ω2

        # State-correction stages (2026-09-11, config.py stage_at): after block k an intermediate head
        # reads the running estimate x̂_k, which is re-injected into h so blocks after k see the state
        # they have to correct. Residual form: heads accumulate in a-space (pre-HardIC), so every stage
        # and the final output are HardIC(Σ a). stage_inj is zero-init → at init the net equals the plain
        # stack with summed heads. τ enters via x̂_k(τ), so the "τ only" arm of the plan is what this is
        # compared against. // 중간 헤드 → x̂_k → zero-init Linear(4→width)로 h에 재주입. a 누적 잔차형
        self.stage_at = sorted(int(k) for k in netCfg.stage_at)
        self.stage_w = netCfg.stage_w            # Loss._DataLossImpl에서 중간 단계 손실 가중
        self.stage_inj_mode = netCfg.stage_inj   # "state" | "tau" (대조 팔) | "embed" (예측기→보정기 handoff)
        if self.stage_inj_mode not in ("state", "tau", "embed"):
            raise ValueError(f"stage_inj {self.stage_inj_mode!r}")
        if any(k < 1 or k > netCfg.n for k in self.stage_at):
            raise ValueError(f"stage_at {self.stage_at} must lie in 1..n_blocks")
        self.stage_heads = nn.ModuleList([nn.Linear(width, 4) for _ in self.stage_at])
        # "embed" (2026-09-16, predictor+corrector): h is *replaced* at stage k by a ParamEmbed-like FFN of
        #   [x̂_k·stage_scale, τ features] — the corrector (blocks >  k, own FiLM via film_split_at) sees only the
        #   predicted state and τ, not the predictor's hidden h. Final heads zero-init → output = x̂_k at init.
        #   Gradient reaches the predictor through x̂_k (no detach); stage_w=1 trains blocks ≤ k as a full predictor.
        # // 예측기 h를 버리고 x̂_k·τ의 FFN 임베딩으로 보정기 시작. 최종 헤드 zero-init(초기 출력 = x̂_k), detach 없음
        if self.stage_inj_mode == "embed":
            self.stage_inj = nn.ModuleList([
                nn.Sequential(nn.Linear(4 + gx, width), nn.SiLU(), nn.Linear(width, width)) for _ in self.stage_at])
        else:
            self.stage_inj = nn.ModuleList([nn.Linear(4, width) for _ in self.stage_at])
        # win_tokens: [x̂_k(τ_1)..x̂_k(τ_P)]·stage_scale (4P) → z (width); last layer zero-init → starts as plain embed
        # // 창 P점 상태 → z. 마지막 층 zero-init (초기엔 행 단위 embed와 동일)
        self.win_enc = None
        if self.win_tok > 0:
            self.win_enc = nn.Sequential(nn.Linear(4 * self.win_tok, width), nn.SiLU(), nn.Linear(width, width))

        self.ResetParameters()

    def ResetParameters(self):
        # 이 백본의 정식 초기화. __init__이 부르고, scratch 재초기화도 **반드시 이걸** 불러야 한다.
        #   nn.Module.reset_parameters()를 모듈마다 부르는 방식은 아래 zero-init 두 개를 덮어써서
        #   FiLM γ가 Kaiming 스케일로 6블록을 곱셈 통과 → h 폭주를 만든다 (adapter도 항등성 상실).
        # Kaiming init for SiLU // Tanh용 Xavier 대신 ReLU-family 휴리스틱
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

        # stage 재주입 zero-init — 초기 출력이 단순 스택(헤드 합)과 같다 (Kaiming 루프 뒤)
        #   embed: 재주입 FFN은 Kaiming 유지(h를 대체하므로 0이면 보정기 입력이 상수), 대신 최종 헤드 zero-init
        if self.stage_inj_mode == "embed":
            for head in (self.head_theta, self.head_omega):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        else:
            for inj in self.stage_inj:
                nn.init.zeros_(inj.weight)
                nn.init.zeros_(inj.bias)
        if self.win_enc is not None:
            nn.init.zeros_(self.win_enc[-1].weight)
            nn.init.zeros_(self.win_enc[-1].bias)

        # FiLMBlock 가지 1/√n 스케일 (Kaiming 루프 뒤). h + SiLU(Wh)는 Kaiming만으로는 블록마다
        #   분산이 2배(12블록 = 실측 |out| 45×) → 가지를 1/√n으로 줄여 (1+1/n)ⁿ ≈ e로 유계화.
        n_active = len(self.blocks)
        for b in self.modules():
            if isinstance(b, FiLMBlock):
                with torch.no_grad():
                    b.fc1.weight.mul_(1.0 / n_active ** 0.5)

        # AttnPoolBlock: 출력 zero-init(항등 출발), 쿼리 N(0, 1/d) (Kaiming 루프는 Linear만 돈다)
        for b in self.modules():
            if isinstance(b, AttnPoolBlock):
                nn.init.zeros_(b.out.weight)
                nn.init.zeros_(b.out.bias)
                nn.init.normal_(b.query, std=1.0 / b.d ** 0.5)

        # XAttnBlock: 출력 zero-init(항등 출발), q/k는 std 1/√W (Kaiming은 score를 O(W)로 키워 softmax 포화)
        for b in self.modules():
            if isinstance(b, XAttnBlock):
                nn.init.zeros_(b.out.weight)
                nn.init.zeros_(b.out.bias)
                for lin in (b.q, b.k):
                    nn.init.normal_(lin.weight, std=1.0 / lin.weight.shape[1] ** 0.5)

        # FiLM 출력도 zero-init → γ=1, β=0에서 출발 (Kaiming 루프 뒤여야 함). film_corr(분리 MLP)도 동일
        for film in (self.film, self.film_corr):
            if film is None:
                continue
            nn.init.zeros_(film.net[-1].weight)
            nn.init.zeros_(film.net[-1].bias)

    @staticmethod
    def _MakeBlock(kind, width, attn_tokens, xattn_heads=4):
        if kind == "film":
            return FiLMBlock(width)
        if kind == "ffn":
            return FFNBlock(width)
        if kind == "xattn":
            return XAttnBlock(width, xattn_heads)
        return AttnPoolBlock(width, attn_tokens)

    def ApplyBlock(self, i, h, gammas, betas, n_q=1):
        # Dense block i by type; gammas/betas are the unbound FiLM stacks (film slots only) // 블록 타입 디스패치
        #   Probes that mirror forward (scratch/depth_readout_probe.py, stage_probe.py) call this too.
        #   n_q: query rows per case (xattn only; rows case-major, token rows already appended — see TokenRows).
        kind = self.block_types[i]
        if kind == "film":
            k = self.film_slot[i]
            return self.blocks[i](h, gammas[k], betas[k])
        if kind == "xattn":
            k = self.film_slot[i]
            return self.blocks[i](h, gammas[k], betas[k], n_q, self.n_tok)
        return self.blocks[i](h)

    def _TokenFeats(self, feats_case, cond_case):
        # (n, F) case rows → token rows (n, P, F) at τ = tau_tok with the case columns copied; cond likewise
        # // 케이스 행 → τ 격자 토큰 행
        n, n_feat = feats_case.shape
        n_tok = self.n_tok
        case_cols = feats_case[:, 1:].unsqueeze(1).expand(n, n_tok, n_feat - 1)
        tok_tau = self.tau_tok.to(feats_case.dtype).view(1, n_tok, 1).expand(n, n_tok, 1)
        tok = torch.cat([tok_tau, case_cols], dim=-1)
        cond_t = tuple(c.unsqueeze(1).expand(n, n_tok, *c.shape[1:]) for c in cond_case)
        return tok, cond_t

    def TokenRows(self, feats, cond, n_q):
        # Append the P window-token rows to each case group (xattn nets). feats rows are case-major with n_q
        # rows per case; the case columns (feats[:, 1:], cond) are copied from the group's first row and τ is
        # replaced by the fixed grid tau_tok. Returns (feats', cond') with M = n_q + P rows per case.
        # // 케이스 첫 행의 조건 열 + τ 격자 → 토큰 행 P개를 케이스 그룹 뒤에 덧붙임
        n_rows, n_feat = feats.shape
        if n_rows % n_q != 0:
            raise ValueError(f"rows {n_rows} not a multiple of n_q {n_q}")
        n = n_rows // n_q
        tok, cond_t = self._TokenFeats(feats[::n_q], tuple(c[::n_q] for c in cond))
        feats_all = torch.cat([feats.view(n, n_q, n_feat), tok], dim=1).reshape(-1, n_feat)
        cond_all = tuple(torch.cat([c.view(n, n_q, *c.shape[1:]), ct], dim=1).reshape(-1, *c.shape[1:])
                         for c, ct in zip(cond, cond_t))
        return feats_all, cond_all

    def TokenStates(self, feats_case, cond_case):
        # Token-only pass (n_q = 0): the token streams at the input of every xattn block, list of (n, P, W).
        # Tokens never depend on query rows, so this can run outside a τ-jvp (StateDerivs) and be handed to
        # forward(tok_hs=) as zero-tangent primals — the forward-AD then covers the query rows only.
        # // 토큰만 통과시켜 xattn 블록 입력 상태를 기록 — jvp 밖에서 1회, 질의 행 jvp에 상수로 전달
        # win_tokens: returns [z] (n, W) instead — predictor states at the P grid rows (no grad) → win_enc.
        # // win_tokens면 토큰 x̂_k(무기울기) → win_enc → [z]
        tok, cond_t = self._TokenFeats(feats_case, cond_case)
        n_feat = tok.shape[-1]
        cond = tuple(ct.reshape(-1, *ct.shape[2:]) for ct in cond_t)
        if self.win_tok > 0:
            with torch.no_grad():
                x_tok = self._Core(tok.reshape(-1, n_feat), cond, 0, None, True)[1][0]
            return [self.win_enc((x_tok * self.stage_scale).view(-1, 4 * self.win_tok))]
        return self._Core(tok.reshape(-1, n_feat), cond, 0, None, True)[2]

    def QueryRows(self, t, n_q):
        # Drop the token rows again: (n·M, k) → (n·n_q, k) // 토큰 행 제거
        m = n_q + self.n_tok
        return t.view(-1, m, t.shape[-1])[:, :n_q].reshape(-1, t.shape[-1])

    def CondOf(self, feats):
        # FiLM (γ, β) for the rows of feats — (N, n_blocks, width) each. τ-independent
        # (feats[:, 1:] only), so callers that evaluate many τ per case (rollout) or differentiate
        # in τ (jvp) compute it once and pass it to forward(cond=).
        # // FiLM 조건을 케이스 단위로 1회 계산해 forward(cond=)에 넘기기 위한 진입점
        c = feats[:, 1:]
        if self.film_corr is None:
            return self.film(c)
        ga, ba = self.film(c)                          # 예측 구간 슬롯
        gc, bc = self.film_corr(c)                     # 보정 구간 슬롯 (film_split_at 이후)
        return torch.cat([ga, gc], dim=1), torch.cat([ba, bc], dim=1)

    def forward(self, feats, cond=None, Stages=False, n_q=1, tok_hs=None):
        # cond: CondOf() 결과를 그대로 (N행 정렬) — None이면 여기서 행마다 계산 (기존 경로)
        # Stages: True면 (out, [x̂_k ...]) — 중간 단계 상태도 반환 (Loss의 stage 감독용). 기본 False = 종전
        # n_q: xattn nets only — rows are case-major with n_q rows per case. tok_hs None → the P token rows are
        #   appended here (and dropped from the outputs); tok_hs = TokenStates(...) → rows are queries only and
        #   the given token streams feed the xattn K/V (jvp path). Ignored when the net has no xattn block.
        # // xattn: 케이스당 질의 행 수. tok_hs가 있으면 토큰 행을 안 붙이고 그 상태를 K/V로 씀

        if cond is None:
            cond = self.CondOf(feats)
        if self.win_tok > 0 and tok_hs is None:          # 창 임베딩 z: 케이스당 1회 (토큰 행은 붙이지 않음)
            tok_hs = self.TokenStates(feats[::n_q], tuple(c[::n_q] for c in cond))
        append = self.n_tok > 0 and tok_hs is None
        if append:
            feats, cond = self.TokenRows(feats, cond, n_q)
        out, stages, _ = self._Core(feats, cond, n_q, tok_hs, False)
        if append:                                      # 토큰 행은 출력에서 제외
            out = self.QueryRows(out, n_q)
            stages = [self.QueryRows(x_k, n_q) for x_k in stages]
        return (out, stages) if Stages else out

    def _Core(self, feats, cond, n_q, tok_hs, record):
        # Trunk on the given rows; xattn blocks take their token K/V from the rows themselves (tok_hs None,
        # n_q + P rows per case) or from tok_hs (n_q rows per case). record → also return the token streams
        # at each xattn block input (rows are then all tokens, n_q = 0). // 공통 트렁크 경로
        tau = feats[:, 0:1] # relative time within window // 윈도우 내 상대시간
        emb_t = self.fourier(tau)  # (N, 2L)
        t_norm = 2.0 * tau / self.march_dt - 1.0
        x = torch.cat([emb_t, t_norm], dim=-1)             # (N, 2L+1) — trunk는 τ만
        gamma, beta = cond                                 # 케이스 → 블록별 (γ, β)
        # unbind once: backward is a single stack. Indexing gamma[:, i] per block instead gave
        # n_blocks SelectBackward nodes, each materializing a fresh (N, n_blocks, width) zero
        # tensor to scatter into — ~7% of the step (profile 2026-08-31).
        # // 블록별 슬라이스 대신 unbind 1회 — SelectBackward의 (N,n,width) 0-텐서 재생성 제거
        gammas, betas = gamma.unbind(1), beta.unbind(1)

        h = self.proj_in(x)

        a_cum, stages, st = 0.0, [], 0
        tok_rec, j = [], 0
        # pos = 지나온 블록 수 (stage_at과 같은 1-based 경계)
        for i in range(len(self.blocks)):
            if self.block_types[i] == "xattn" and (record or tok_hs is not None):
                k = self.film_slot[i]
                if record:
                    tok_rec.append(h.view(-1, self.n_tok, h.shape[-1]))
                h = self.blocks[i](h, gammas[k], betas[k], n_q, self.n_tok,
                                   tok=None if tok_hs is None else tok_hs[j])
                j += 1
            else:
                h = self.ApplyBlock(i, h, gammas, betas, n_q)   # film: FiLM이 잔차 가지 안에서만 변조
            pos = i + 1
            if st < len(self.stage_at) and self.stage_at[st] == pos:
                a_cum = a_cum + self.stage_heads[st](h)          # 잔차 누적 (a-space)
                x_k = self._HardIC(a_cum, feats, tau)             # 물리 상태 [Δθ, ω] (τ=0 항등 유지)
                stages.append(x_k)
                if self.win_tok > 0 and record:                     # token pass: only x̂_k is needed
                    return None, stages, tok_rec
                if self.stage_inj_mode == "embed":                  # handoff: h ← FFN([x̂_k·scale, τ feats])
                    h = self.stage_inj[st](torch.cat([x_k * self.stage_scale, x], dim=-1))
                    if self.win_tok > 0:                            # + window embedding z of the row's case
                        h = h + tok_hs[0].repeat_interleave(n_q, dim=0)
                else:
                    if self.stage_inj_mode == "tau":                # 대조 팔: 학습된 몫을 뺀 τ·IC 항만
                        x_k = self._HardIC(torch.zeros_like(a_cum), feats, tau)
                    h = h + self.stage_inj[st](x_k * self.stage_scale)   # 스케일 맞춰 재주입 (zero-init 출발)
                st += 1

        a = a_cum + torch.cat([self.head_theta(h), self.head_omega(h)], dim=-1)
        out = self._HardIC(a, feats, tau)               # (N, 4) [Δθ1, Δθ2, ω1, ω2]
        return out, stages, tok_rec

    @property
    def stage_scale(self):
        # x̂ → O(1): Δθ ~ ω_rms·march_dt (창 하나 동안의 회전), ω ~ ω_rms. feats의 ω/ω_rms 관례와 동일
        r = (1.0 / self.omega_rms).reshape(())
        return torch.stack([r / self.march_dt, r / self.march_dt, r, r])

    def _HardIC(self, a, feats, tau):
        a_theta, a_omega = a[:, :2], a[:, 2:]
        if self.hard_ic == "off":
            return a                                      # (N, 4) raw [Δθ1, Δθ2, ω1, ω2]

        # Hard IC imposition // τ=0 경계조건을 구조로 강제 — soft IC 손실을 대체한다 (2026-08-01)
        #   soft로는 못 잡는 이유: RolloutLoss의 grad 윈도우는 τ=0을 제외하고(Loss.py grid),
        #   ICSamplesRaw는 저장 데이터 IC만 쓴다 → 마칭이 실제로 마주치는 off-manifold IC에서
        #   Δθ(0)=0을 훈련하는 코드가 없었다. 잔차 RMSE ≈ 8e-4로 목표(1e-3) 예산의 80%.
        #   여기서는 IC가 항등식이라 임의 IC에서 seam 점프가 정확히 0이다.
        #   s = τ/march_dt ∈ [0,1] — 프리팩터를 정규화해 헤드 출력이 O(1)에 머문다.
        s = tau / self.march_dt
        w_ic = feats[:, 5:7] * self.omega_rms   # _BuildFeats 열 5,6 = ω/ω_rms → 절대 ω_IC 복원
        omega = w_ic + s * a_omega              # ω(0) = ω_IC
        if self.hard_ic == "c1":    
            # dΔθ/dτ(0) = ω_IC까지 강제 → 경계에서 kin 잔차가 항등적으로 0. 두 헤드가 어긋난
            # 상태(trap 16: corr(dθ/dτ, ω_true)=0.07)를 최소한 seam에서는 붙여 놓는다.
            # 헤드는 O(τ²) 보정만 담당 — τ→0 부근 오차가 s²로 눌린다.
            theta = s * (self.march_dt * w_ic + s * a_theta)
        else:
            theta = s * a_theta                 # ref: Δθ(0) = 0 만
        return torch.cat([theta, omega], dim=-1)  # (N, 4) [Δθ1, Δθ2, ω1, ω2]
