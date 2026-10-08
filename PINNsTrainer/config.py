import tomllib
from dataclasses import dataclass


@dataclass
class NetCfg:
    fourier_l: int = 32       # number of frequencies L // 주파수 개수
    f_min: float = 0.2        # Hz, lowest oscillation // 최저 진동 주파수
    f_max: float = 20.0       # Hz // 데이터 대역폭 측정치 기반 상한
    # 측정(mixed corpus, 1s 윈도우, Hann): 데이터 Nyquist는 50Hz라 >50Hz 파워는 정확히 0.
    #   flip ω 케이스별 f(99% 파워) p99 = 15.8Hz, >20Hz 잔여 파워 7e-4. nonflip은 그 1/10.
    # 56Hz였을 때 문제: θ 진폭 17.24/(2π·56)=0.049면 dθ/dτ에 std 17의 노이즈를 만드는데,
    #   이는 Val θ RMS(~0.12) 아래라 data loss가 못 본다 → 무제약 성장. 20Hz면 0.137로
    #   data 오차 수준까지 올라와 data loss가 억제한다. // [[kin-derivative-noise]]

    ic_feat_dim: int = 6      # 4 trig(θ0) + 2 ω(0) // raw m/L 제외 (ParamEmbed로 대체)
    param_embed_dim: int = 32 # ParamEmbed 출력 차원 // option B3
    width: int = 384          # hidden layer width // 은닉층 너비 (M tier)
    n: int = 6                # FiLMBlock count // 잔차 블록 수

    # τ=0 경계조건 부과 방식 // Networks.forward 말미의 출력 ansatz
    #   "off" — 종전. 출력이 raw [Δθ, ω]이고 IC는 soft 손실로만 유도된다.
    #   "c0"  — Δθ = s·A_θ, ω = ω_IC + s·A_ω. Δθ(0)=0, ω(0)=ω_IC 가 항등식.
    #   "c1"  — 추가로 dΔθ/dτ(0) = ω_IC. Δθ = s·(march_dt·ω_IC + s·A_θ).
    #   "off"가 아니면 IC 손실이 정확히 0이므로 loop가 ic 항 자체를 끈다(비용 절감).
    #   구조 변경이라 체크포인트 비호환 — 값을 바꾸면 fresh pretrain이 필요하다.
    hard_ic: str = "c1"

    # 조건화: trunk 입력은 τ features뿐, 케이스(IC·param_embed)는 FiLMCond를
    #   타고 블록마다 (γ, β)로 잔차 가지를 곱셈 변조한다 (networks.FiLMBlock). 옵션 아님.
    film_hidden: int = 128    # 조건화 MLP 은닉폭. 출력은 2·width·n (블록당 γ, β)

    # State-correction stages: after block k (1-based, k in stage_at) an
    #   intermediate head reads x̂_k = HardIC(Σ_{j≤k} a_j), a zero-init Linear(4→width) adds it back to h,
    #   and later blocks correct that estimate instead of mapping τ from scratch. Final output is
    #   HardIC(Σ a_j + head_final). stage_w weights the intermediate data losses (deep supervision); 0 = none.
    #   () = off, unchanged network. Checkpoint-incompatible with () → fresh runs only.
    # // 상태 보정 단계: stage_at 블록 뒤에 중간 헤드 → x̂_k → h에 재주입(zero-init). 잔차형(a 누적), 중간 손실 stage_w
    #   stage_inj: "state" = x̂_k 재주입 | "tau" = 대조 팔, HardIC(a=0)=[s·md·ω_IC, ω_IC]만 재주입 (헤드·중간 손실은
    #   동일) → 두 팔의 차 = "상태 정보"의 몫, τ 재주입 자체의 몫과 분리.
    #   stage_inj "embed": predictor→corrector handoff. At stage k, h is replaced by
    #   FFN([x̂_k·stage_scale, τ feats]) (Linear→SiLU→Linear, width), final heads zero-init (output = x̂_k at init).
    #   Use with film_split_at = k (corrector gets its own FiLM) and stage_w = 1.0 (blocks ≤ k = full predictor).
    # // "embed": 예측기 h 대신 x̂_k·τ 임베딩으로 보정기 시작. film_split_at=k, stage_w=1.0과 함께 쓴다
    stage_at: tuple = ()
    stage_w: float = 0.1
    stage_inj: str = "state"
    # Correction-unit screen: per-block type list of length n.
    #   "film" = FiLMBlock (γ,β from FiLMCond) | "ffn" = same block with γ≡1, β≡0 (no conditioning, same MACs)
    #   | "attn" = AttnPoolBlock (h → attn_tokens sub-tokens, one learned query, zero-init output).
    #   FiLMCond is sized to the number of "film" blocks; () = all film (checkpoint-compatible, unchanged net).
    # // 블록 타입 열: film/ffn/attn. FiLMCond 출력은 film 블록 수만큼, () = 전부 film(종전과 동일)
    #   film_split_at > 0: blocks ≥ split get their own FiLMCond MLP (film_corr) so the prediction-region
    #   FiLM can be frozen exactly (requires_grad, no partial-row masking under AdamW decay). 0 = one shared MLP.
    # // film_split_at: 이 블록부터 별도 FiLMCond(film_corr). 동결 스크린용, 0 = 공유 1개(종전)
    block_types: tuple = ()
    film_split_at: int = 0
    attn_tokens: int = 8
    # "xattn" (window-token corrector): per case the predictor is also evaluated on a fixed τ grid
    #   τ_j = j·march_dt/P (j=1..P, xattn_tokens); those P rows go through the same handoff (stage_inj embed)
    #   and become tokens. An xattn block = multi-head cross-attention of every row (query and token rows alike)
    #   over the case's token rows (zero-init output), followed by the FiLM residual branch. Rows must be
    #   case-major with n_q rows per case (forward(n_q=)); the P token rows are appended inside forward.
    #   Corrector thus sees the predicted trajectory shape across τ, which no per-row input carries.
    # // 창 안 τ 토큰 보정기: 케이스별 예측기 x̂를 τ 격자 P점에서 뽑아 토큰으로, 각 행이 그 집합에 cross-attention
    xattn_tokens: int = 8
    xattn_heads: int = 4
    # Window-state corrector: per case the predictor is evaluated on the τ grid τ_j = j·march_dt/P
    #   (j=1..P, win_tokens; 25 = the dt 0.01 data grid of a 0.25 s window). The P states x̂_k·stage_scale are
    #   concatenated (4P) → FFN → z (width), added to the handoff embedding of every row of the case. Corrector
    #   blocks stay film. Token states are detached (no grad into the predictor through z). Needs stage_inj
    #   "embed", one stage, no xattn block; rows case-major with n_q per case (same plumbing as xattn). 0 = off.
    # // 창 상태 보정기: 예측기 x̂_k를 창 격자 P점에서 한 번에 뽑아 4P 벡터 → FFN → z, 케이스의 모든 행 handoff 임베딩에 더함
    win_tokens: int = 0

    @property
    def gx_dim(self) -> int:
        # Fourier(2L) + t_norm(1) + IC(6) + param_embed // concat trunk 입력 차원
        return 2 * self.fourier_l + 1 + self.ic_feat_dim + self.param_embed_dim

    @property
    def cond_dim(self) -> int:
        # IC(6) + param_embed // τ 무관 열만 = feats[:, 1:]
        #   τ에 의존하면 StateDerivs의 jvp dual 섭동이 γ·β를 오염시킨다
        return self.ic_feat_dim + self.param_embed_dim

    @property
    def trunk_in_dim(self) -> int:
        # trunk는 τ만 받는다 — 케이스는 γ·β로만 들어온다
        return 2 * self.fourier_l + 1


@dataclass
class TrainCfg:
    lambda_ic: float = 1.0
    lambda_data: float = 1.0
    lambda_kin: float = 1.0
    lambda_phys: float = 1.0
    lambda_energy: float = 0.5
    lambda_roll: float = 1.0      # rollout-aware (pushforward) 손실 base 가중치 // 윈도우 핸드오프 교정
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    # 항별 clip 예산은 grad_clip 고정, RebalanceGradScales clamp 하한은 없음(0) — 하드코딩.
    #   하한은 아무것도 보호하지 않음, 폭주 방지는 상한 역할 // [[grad-scale-floor-pinned]]
    grad_scale_max: float = 100.0 # RebalanceGradScales clamp 상한 // energy가 10에 붙박이라 완화
    replay_frac: float = 0.25
    n_colloc_cases: int = 0       # finetune colloc case 풀 크기 (0 → max_cases)
    phys_ramp_center: int = 30    # e2 = epoch - warmup_epochs 기준 중심
    phys_ramp_width: int = 10     # sigmoid 폭 // ramp steepness
    lr_drop_epoch: int = 0        # 고정 LR 1회 하향 에폭 (0=비활성) // OdeScheduler와 별개
    lr_drop_to: float = 0.0       # lr_drop_epoch부터 적용할 LR 상한 // min(cur, lr_drop_to)
    ema_decay: float = 0.999      # Polyak weight EMA decay (0=비활성) // best.pt/평가 안정화
    roll_robust_k: float = 4.0    # rollout Cauchy 스케일 = k × 배치 median (0=비활성, 순수 mean) // heavy-tail 억제
    roll_balance_cases: int = 128 # RebalanceGradScales의 roll grad 노름 측정용 케이스 수 // 측정 비용 절감
    roll_balance_points: int = 12 # 위 측정의 윈도우 내 시점 수 // 학습 스텝의 roll_points와 동일 스케일

    # 스텝당 (k0, depth) 추첨 횟수 (1 = 단일 추첨) // roll gradient 분산 축소
    #   roll gradient 분산은 케이스 수가 아니라 추첨 개수가 결정한다 — roll_cases를 늘려도 안 줄어든다.
    #   케이스 예산을 n_draws로 쪼개므로 네트워크 evaluation 수는 불변 = 비용 거의 동일.
    #   k0는 윈도우 축으로 stratify하고, robust mean은 합친 뒤 1회만 건다.
    roll_draws: int = 1
    # Cauchy 스케일 c의 median EMA 계수 (0 = 종전, 매 스텝 배치 median 그대로) // 스케일 자체가
    #   추첨마다 흔들리면 그것도 노이즈원이다. 0.9면 c가 학습 진행을 따라가되 스텝 노이즈는 뺀다.
    roll_median_ema: float = 0.0
    # robust mean 형태: "cauchy" = c·log1p(l/c) (tail grad ∝ c/l), "huber" = min(l, 2√(δl)−δ)
    #   (tail grad ∝ √(δ/l)) — Cauchy가 tail gradient를 끊어 flip 평균을 악화시킨 것의 완화
    #   후보. δ/c = roll_robust_k × median 공용. // [[rollout-tail-cauchy-tradeoff]]
    roll_robust_kind: str = "cauchy"
    # Drop the top-q fraction of rows (by per-row loss) from each term's gradient (0 = plain mean) // 손실 상위 q 행을
    #   grad에서 제외. 상위 소수 행이 손실·grad 노이즈를 지배해서. 분위는 청크(배치)별.
    trim_q_data: float = 0.0
    trim_q_phys: float = 0.0
    trim_q_kin: float = 0.0

    # IC 항 스텝당 케이스 상한 (0 = 전체 active_cases) // IC는 세그먼트마다 반복되는 항이라
    #   비용이 max_cases × n_segments로 곱셈이다 (march_dt를 줄이면 조용히 커진다).
    #   서브샘플은 사실상 공짜: 1024에서도 전체 대비 gradient cos ≈ 0.998로 data 미니배치 노이즈보다
    #   조용하고, 손실이 평균 정규화라 유효 가중치도 불변. // [[ic-term-dominates-step]]
    ic_max_n: int = 0

    # Frozen-prediction-region screen: load proj_in / param_embed / blocks[:freeze_below] /
    #   stage_heads[0] / film (split A) / scaler buffers from freeze_from and freeze them, so every arm sees the
    #   same x̂_{freeze_below}; the rest (blocks ≥ freeze_below, later stage heads, final heads, stage_inj,
    #   film_corr) starts fresh. Requires net.film_split_at == freeze_below and stage_at[0] == freeze_below.
    # // 예측 구간 동결: freeze_from ckpt에서 블록 1..freeze_below·stage head·FiLM(A)·사영 로드 후 동결. ""=off
    freeze_from: str = ""
    freeze_below: int = 0

    # torch.compile: data/IC/kin/phys 손실 슬라이스를 Inductor로 컴파일 // 고정 shape 전제
    #   스텝 ~1.66× 빠름, 컴파일 대기 ~27s. 전제: 청크 shape이 몇 개로 고정. n_colloc을 colloc_chunk 배수로 두면 그래프 1개,
    #   나머지 청크가 생기면 shape별로 1개 추가(허용치 내). // [[fp64-not-bottleneck]]
    use_compile: bool = True   # 전 config에서 true로 확정 — 프로브만 코드에서 끈다(probe_common)

    # FP32 matmul 정밀도. "high"=TF32 허용, "highest"=순수 FP32. 학습은 "high" 고정(toml 옵션 없음):
    #   현재 n_colloc에서 TF32 반올림 노이즈는 콜로케이션 추첨 노이즈보다 작다. 이상치(ic 항)는 트랩 18.
    #   **추론은 별개** — eval/프로브는 TF32 off (트랩 25).
    matmul_precision: str = "high"

@dataclass
class CollocCfg:
    n_colloc: int = 20000
    ## 비용은 n_colloc에 선형 — colloc_chunk 단위 fwd+bwd 횟수가 그대로 늘어난다. 병목은 jvp fwd+bwd.
    ## use_compile 시에는 colloc_chunk의 배수로 두는 게 좋다 (나머지 청크가 그래프를 하나 더 만든다).

    colloc_flip_bias: float = 0.0   # finetune: flip case 하한 비율 (0=uniform) // 명시적 is_flip balancing

    ic_sigma: float = 0.0             # Phase 3: 물리 콜로케이션 IC 섭동 target (0=비활성)
    ic_sigma_warmup: int = 200        # sigma 0→target 선형 램프 에폭 (Phase 2 진입 후)
    # xattn nets: colloc rows grouped per case — n_colloc/per_case cases × per_case iid τ each,
    #   so every chunk is case-major for forward(n_q=per_case). IC perturbation is drawn per case.
    #   Ignored (1) when the net has no xattn block. colloc_chunk and n_colloc must be multiples of it.
    # // xattn: 콜로케이션을 케이스당 per_case개 τ로 묶음 (토큰 행 공유), 비-xattn net에선 무시
    per_case: int = 8


@dataclass
class DataCfg:
    g: float = 9.81

    t_data_max: float = 3.0
    march_dt: float = 1.0     # time-marching window duration (s) // 플로우맵 윈도우 길이 — 네트워크가 보는 상대시간 τ∈[0,march_dt]
    energy_eps: float = 0.01  # energy 상대잔차 분모 바닥값 // floor when |E0|≈|E|≈0 (symmetric denom)
    phys_eps: float = 1.0     # EOM 상대잔차 분모 바닥값 // floor for relative physics residual denom (≈ small frac of typical |dω/dt|)
    kin_eps: float = 1.0      # kin 분모(= omega_rms²)에 더하는 바닥값 // floor, ω_rms²에 비해 작음

    nonflip_path: str = "data/nonflip_RK4_0_3s.npy"
    data_path: str = ""           # finetune mixed dataset (empty → nonflip_path)
    scaler_name: str = "scaler.npy"
    scaler_extra_omega: str = ""  # pretrain scaler ω RMS에 포함할 추가 궤적(예: mixed) — flip OOD 방지
    batch_size: int = 8192
    # kin/phys/ic backward 청크 크기 // 0이면 batch_size 사용. data 배치와 분리해 physics
    #   transient peak 억제. use_compile 하 스위트스팟 4096 (그 위는 수확체감).
    colloc_chunk: int = 0


def LoadConfig(path):
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    t   = raw["train"]
    n   = raw["net"]
    o   = raw["optimizer"]
    s   = raw.get("scheduler", {})   # val scheduler 제거 후 하위호환용 // min_lr 폴백 소스
    lam = raw["lambda"]
    c   = raw["colloc"]
    d   = raw["data"]
    os_ = raw.get("ode_scheduler", {})

    net_cfg = NetCfg(
        fourier_l=n["fourier_l"],
        f_min=n["f_min"],
        f_max=n["f_max"],
        width=n["width"],
        n=n["n_blocks"],
        param_embed_dim=n.get("param_embed_dim", 32),
        hard_ic=n.get("hard_ic", "c1"),
        film_hidden=n.get("film_hidden", 128),
        stage_at=tuple(n.get("stage_at", [])), stage_w=n.get("stage_w", 0.1),
        stage_inj=n.get("stage_inj", "state"),
        block_types=tuple(n.get("block_types", [])), film_split_at=n.get("film_split_at", 0),
        attn_tokens=n.get("attn_tokens", 8),
        xattn_tokens=n.get("xattn_tokens", 8), xattn_heads=n.get("xattn_heads", 4),
        win_tokens=n.get("win_tokens", 0),
    )

    train_cfg = TrainCfg(
        lambda_ic=lam["ic"],
        lambda_data=lam["data"],
        lambda_kin=lam.get("kin", 1.0),
        lambda_phys=lam.get("phys", 1.0),
        lambda_energy=lam.get("energy", 0.5),
        lambda_roll=lam.get("roll", 1.0),
        lr=o["lr"],
        weight_decay=o["weight_decay"],
        grad_clip=o["grad_clip"],
        grad_scale_max=t.get("grad_scale_max", 100.0),
        replay_frac=t.get("replay_frac", 0.25),
        n_colloc_cases=t.get("n_colloc_cases", c.get("n_colloc_cases", 0)),
        phys_ramp_center=t.get("phys_ramp_center", 30),
        phys_ramp_width=t.get("phys_ramp_width", 10),
        lr_drop_epoch=t.get("lr_drop_epoch", 0),
        lr_drop_to=t.get("lr_drop_to", 0.0),
        ema_decay=t.get("ema_decay", 0.999),
        roll_robust_k=t.get("roll_robust_k", 4.0),
        roll_balance_cases=t.get("roll_balance_cases", 128),
        roll_draws=t.get("roll_draws", 1),
        roll_median_ema=t.get("roll_median_ema", 0.0),
        roll_robust_kind=t.get("roll_robust_kind", "cauchy"),
        trim_q_data=t.get("trim_q_data", 0.0), trim_q_phys=t.get("trim_q_phys", 0.0),
        trim_q_kin=t.get("trim_q_kin", 0.0),
        roll_balance_points=t.get("roll_balance_points", 12),
        ic_max_n=t.get("ic_max_n", 0),
        freeze_from=t.get("freeze_from", ""), freeze_below=t.get("freeze_below", 0),
        use_compile=t.get("use_compile", True),
        matmul_precision=t.get("matmul_precision", "high"),
    )

    colloc_cfg = CollocCfg(
        n_colloc=c["n_colloc"],
        colloc_flip_bias=c.get("colloc_flip_bias", 0.0),
        ic_sigma=c.get("ic_sigma", 0.0),
        ic_sigma_warmup=c.get("ic_sigma_warmup", 200),
        per_case=c.get("per_case", 8),
    )

    data_cfg = DataCfg(
        g=d["g"],
        t_data_max=d["t_data_max"],
        march_dt=d.get("march_dt", 1.0),   # 구 config 하위호환 // backward-compat for pre-marching configs
        energy_eps=d["energy_eps"],
        phys_eps=d.get("phys_eps", 1.0),   # 구 config 하위호환 // backward-compat for configs predating relative phys loss
        kin_eps=d.get("kin_eps", 1.0),     # 구 config 하위호환 // backward-compat for configs predating relative kin loss
        batch_size=d["batch_size"],
        colloc_chunk=d.get("colloc_chunk", 0),
        nonflip_path=d["nonflip_path"],
        data_path=d.get("data_path", d["nonflip_path"]),
        scaler_name=d["scaler_name"],
        scaler_extra_omega=d.get("scaler_extra_omega", ""),
    )

    # 단위 주의: rel_tol_*은 에폭당 개선율 하한, patience_*는 에폭 수.
    #   StallTracker가 측정 간 경과 에폭으로 복리 환산하므로 측정 주기와 무관하게 게이트 간 잣대가 같다.
    #   rel_tol_*는 "그 지표가 아직 살아있는가"의 하한이지 목표 개선율이 아니다 — 작게 잡을 것.
    # phys 트래커 없음 — patience는 decay 시도 간격(에폭). 구 config의 ema_beta/rel_tol 키는 무시된다.
    ode_s_params = {
        "patience":        os_.get("patience", 80),      # decay 시도 간격(에폭) // 구 phys patience와 실효 동일
        "factor":          os_.get("factor", 0.75),
        "min_lr":          os_.get("min_lr", s.get("min_lr", 1e-6)),
        "phase2_lr":       os_.get("phase2_lr", 0.0),    # Phase 2 진입 1회 LR 캡 (0 = 비활성)
        # veto 예산: phys 정체인 채 연속 veto된 에폭이 이 값을 넘으면 강제 decay (0 = 무제한)
        #   게이트가 작은 개선 한 번에 리셋되는 구조라 veto가 영구 무장될 수 있다 // 손해 상한
        "max_veto_epochs": os_.get("max_veto_epochs", 0),
        "patience_val":    os_.get("patience_val", 20),
        "patience_ext":    os_.get("patience_ext", 50),
        "patience_roll":   os_.get("patience_roll", 75),
        "rel_tol_val":     os_.get("rel_tol_val", 4e-4),
        "rel_tol_ext":     os_.get("rel_tol_ext", 4e-4),
        "rel_tol_roll":    os_.get("rel_tol_roll", 4e-4),
        "ema_beta_val":    os_.get("ema_beta_val", 0.9),
        "ema_beta_ext":    os_.get("ema_beta_ext", 0.9),
        "ema_beta_roll":   os_.get("ema_beta_roll", 0.8),  # 측정이 드물어 약하게
    }

    return net_cfg, train_cfg, colloc_cfg, data_cfg, t, ode_s_params
    