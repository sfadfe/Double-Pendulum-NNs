import torch
import torch.nn as nn

"""
- 입력 feats (N, feat_dim): [τ, sinθ1, cosθ1, sinθ2, cosθ2, ω1, ω2, param_embed(32), (energy_gate)]
  energy_gate=true면 맨 끝에 flip 가능성 마진 1열 추가 — τ 무관 케이스 상수라 jvp/StateDerivs 불변
  ParamEmbed는 Dataset._BuildFeats에서 호출 — raw m/L 대신 임베딩 벡터 사용 // option B
- time-marching flow map: τ는 윈도우 내 상대시간(τ∈[0, march_dt]), trig/ω는 윈도우 시작 상태(IC)
- τ는 Fourier Features(물리 주파수, Hz)로 매핑되고, 또한 [0, march_dt]에서 [-1, 1]로 정규화됨
- 출력: (N, 4) = [Δθ1, Δθ2, ω1, ω2]. Δθ = θ(τ) − θ_IC (상대각), ω는 절대값.
  공유 trunk + head_theta / head_omega 분리 // option B1
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


class ResidualBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.act = nn.SiLU()  # option B2: Tanh → SiLU

    def forward(self, h, u, v):
        z = self.act(self.fc1(h))
        return (1.0 - z) * u + z * v + h  # U, V gating + residual connection 


class FlipAdapter(nn.Module):
    # Energy-gated residual adapter // flip 영역에서만 활성화되는 층 (bottleneck LoRA 형태)
    #   up을 zero-init → 로드 직후 항등사상이라 pretrain 출력과 수치적으로 동일.
    #   g≈0인 nonflip 윈도우에서는 기여도 0 + gradient도 0 → nonflip 표현을 건드리지 않는다.
    #   idx가 주어지면 flip 행만 뽑아 GEMM → nonflip 행은 연산 자체를 건너뛴다 (sparse 경로).
    def __init__(self, hidden, bottleneck):
        super().__init__()
        self.down = nn.Linear(hidden, bottleneck)
        self.act = nn.SiLU()
        self.up = nn.Linear(bottleneck, hidden)

    def forward(self, h, g, idx=None):
        if idx is None:                                   # dense: 전 행 계산 후 g로 스케일
            return h + g * self.up(self.act(self.down(h)))
        if idx.numel() == 0:                              # 전부 nonflip → adapter 미실행
            return h
        delta = g[idx] * self.up(self.act(self.down(h[idx])))
        return h.index_add(0, idx, delta)                 # out-of-place: autograd 안전


class Networks(nn.Module):
    def __init__(self, netCfg, dataCfg):
        super().__init__()
        self.march_dt = dataCfg.march_dt   # 상대시간 정규화 기준 // window duration for τ normalization
        self.hard_ic = netCfg.hard_ic      # "off" | "c0" | "c1" — τ=0 경계조건 부과 방식 (forward 말미)

        self.fourier = FourierFeatures(netCfg.fourier_l, netCfg.f_min, netCfg.f_max)
        self.param_embed = ParamEmbed(netCfg.param_embed_dim)

        self.cond_mode = netCfg.cond_mode  # "concat" | "film" — 케이스가 trunk에 들어가는 경로
        gx = netCfg.trunk_in_dim           # film이면 τ features만
        width = netCfg.width
        self.proj_u = nn.Sequential(nn.Linear(gx, width), nn.SiLU())
        self.proj_v = nn.Sequential(nn.Linear(gx, width), nn.SiLU())
        self.proj_in = nn.Sequential(nn.Linear(gx, width), nn.SiLU())

        self.film = None
        if self.cond_mode == "film":
            self.film = FiLMCond(netCfg.cond_dim, netCfg.film_hidden, width, netCfg.n)
        elif self.cond_mode != "concat":
            raise ValueError(f"cond_mode must be 'concat' or 'film', got {netCfg.cond_mode!r}")

        self.blocks = nn.ModuleList(
            [ResidualBlock(width) for i in range(netCfg.n)]
        )
        self.head_theta = nn.Linear(width, 2)   # Δθ1, Δθ2 // option B1
        self.head_omega = nn.Linear(width, 2)   # ω1, ω2

        # flip 전용 layer: 마지막 flip_adapter_n개 블록 뒤에 게이트된 adapter // 2026-07-30
        self.flip_adapters = nn.ModuleList()
        self.adapter_at = set()
        if netCfg.flip_adapter_dim > 0:
            if not netCfg.energy_gate:
                raise ValueError("flip_adapter_dim > 0 requires energy_gate = true")
            n_ad = min(netCfg.flip_adapter_n, netCfg.n)
            self.adapter_at = set(range(netCfg.n - n_ad, netCfg.n))
            self.flip_adapters = nn.ModuleList(
                [FlipAdapter(width, netCfg.flip_adapter_dim) for _ in range(n_ad)]
            )
        self.gate_center = netCfg.gate_center
        self.gate_temp = netCfg.gate_temp
        self.gate_sparse = netCfg.gate_sparse
        self.gate_eps = netCfg.gate_eps

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

        # adapter 출력만 zero-init — 항등 시작 보장 (Kaiming 루프 뒤에 와야 함)
        for ad in self.flip_adapters:
            nn.init.zeros_(ad.up.weight)
            nn.init.zeros_(ad.up.bias)

        # FiLM 출력도 zero-init → γ=1, β=0에서 출발 (Kaiming 루프 뒤여야 함)
        #   시작점이 concat 팔과 같은 출발선이 되어 A/B가 초기화 차이에 오염되지 않는다
        if self.film is not None:
            nn.init.zeros_(self.film.net[-1].weight)
            nn.init.zeros_(self.film.net[-1].bias)

    def forward(self, feats):

        tau = feats[:, 0:1] # relative time within window // 윈도우 내 상대시간
        emb_t = self.fourier(tau)  # (N, 2L)
        t_norm = 2.0 * tau / self.march_dt - 1.0
        if self.cond_mode == "film":
            x = torch.cat([emb_t, t_norm], dim=-1)             # (N, 2L+1) — trunk는 τ만
            gamma, beta = self.film(feats[:, 1:])              # 케이스 → 블록별 (γ, β)
        else:
            x = torch.cat([emb_t, t_norm, feats[:, 1:]], dim=-1)  # (N, gx_dim)
            gamma = beta = None

        u = self.proj_u(x)
        v = self.proj_v(x)
        h = self.proj_in(x)

        # flip gate: 마지막 열 log(e_rel/e_flip)을 [0,1]로 // 장벽 밴드 중앙에서 스위칭
        # gate_sparse면 g > gate_eps인 행만 골라 adapter를 태운다 — nonflip은 실제로 연산 없음.
        #   버려지는 항의 크기는 g·|adapter| ≤ gate_eps·|adapter|로 유계. 실측 nonflip g는 0.006
        #   (< eps 0.05)이라 dense 경로에서도 이미 무시할 수준 → 값 차이는 절단오차뿐.
        #   idx는 τ 무관 열에서만 나오므로 jvp의 dual 섭동에 불변 → StateDerivs 안전.
        g, idx = None, None
        if len(self.flip_adapters) > 0:
            g = torch.sigmoid((feats[:, -1:] - self.gate_center) / self.gate_temp)
            if self.gate_sparse:
                idx = (g[:, 0].detach() > self.gate_eps).nonzero(as_tuple=True)[0]

        ad = 0
        for i, block in enumerate(self.blocks):  # Pass all residual blocks through U and V gates. // 모든 residual block을 U, V 게이트에 통과시킴
            h = block(h, u, v)
            if gamma is not None:
                h = gamma[:, i] * h + beta[:, i]   # FiLM: 케이스가 블록 출력을 곱셈 변조
            if i in self.adapter_at:
                h = self.flip_adapters[ad](h, g, idx)
                ad += 1

        a_theta = self.head_theta(h)
        a_omega = self.head_omega(h)
        if self.hard_ic == "off":
            return torch.cat([a_theta, a_omega], dim=-1)  # (N, 4) [Δθ1, Δθ2, ω1, ω2]

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
            theta = s * a_theta                 # c0: Δθ(0) = 0 만
        return torch.cat([theta, omega], dim=-1)  # (N, 4) [Δθ1, Δθ2, ω1, ω2] 