import tomllib
from dataclasses import dataclass


@dataclass
class NetCfg:
    fourier_l: int = 32       # number of frequencies L // 주파수 개수
    f_min: float = 0.2        # Hz, lowest oscillation // 최저 진동 주파수
    f_max: float = 20.0       # Hz // 데이터 대역폭 측정치 기반 상한 (2026-07-28)
    # 측정(mixed corpus, 1s 윈도우, Hann): 데이터 Nyquist는 50Hz라 >50Hz 파워는 정확히 0.
    #   flip ω 케이스별 f(99% 파워) p99 = 15.8Hz, >20Hz 잔여 파워 7e-4. nonflip은 그 1/10.
    # 56Hz였을 때 문제: θ 진폭 17.24/(2π·56)=0.049면 dθ/dτ에 std 17의 노이즈를 만드는데,
    #   이는 Val θ RMS(~0.12) 아래라 data loss가 못 본다 → 무제약 성장. 20Hz면 0.137로
    #   data 오차 수준까지 올라와 data loss가 억제한다. // [[kin-derivative-noise]]

    ic_feat_dim: int = 6      # 4 trig(θ0) + 2 ω(0) // raw m/L 제외 (ParamEmbed로 대체)
    param_embed_dim: int = 32 # ParamEmbed 출력 차원 // option B3
    width: int = 384          # hidden layer width // 은닉층 너비 (M tier)
    n: int = 6                # ResidualBlock count // 잔차 블록 수

    # flip 조건화 실험 (2026-07-30): IC 에너지가 θ2 반전 임계를 넘는 여유분을 입력에 명시.
    #   E는 trig(θ0)·ω0·param_embed로 이미 결정되므로 정보 추가가 아니라 feature engineering —
    #   효과가 없으면 "flip 난이도는 조건화 부족이 아니다"가 결론. // Step A 진단용
    energy_gate: bool = False

    # τ=0 경계조건 부과 방식 (2026-08-01) // Networks.forward 말미의 출력 ansatz
    #   "off" — 종전. 출력이 raw [Δθ, ω]이고 IC는 soft 손실로만 유도된다.
    #   "c0"  — Δθ = s·A_θ, ω = ω_IC + s·A_ω. Δθ(0)=0, ω(0)=ω_IC 가 항등식.
    #   "c1"  — 추가로 dΔθ/dτ(0) = ω_IC. Δθ = s·(march_dt·ω_IC + s·A_θ).
    #   "off"가 아니면 IC 손실이 정확히 0이므로 loop가 ic 항 자체를 끈다(비용 절감).
    #   구조 변경이라 체크포인트 비호환 — 값을 바꾸면 fresh pretrain이 필요하다.
    hard_ic: str = "c1"

    # 조건화 방식 (2026-08-02) // 케이스(IC·params)가 trunk에 들어가는 경로
    #   "concat" — 종전. (τ features, IC, param_embed)를 한 벡터로 붙여 proj_u/v/in에 넣는다.
    #              케이스는 매 블록 u·v로 재주입되지만 **덧셈**이라 τ-함수를 평행이동만 시킨다.
    #   "film"   — trunk 입력은 τ features뿐. 케이스는 별도 MLP를 타고 블록마다 (γ, β)를 만들어
    #              h ← γ·h + β로 **곱셈 변조**한다 → 케이스가 τ-함수의 모양 자체를 고른다.
    #              근거: fourier_l 32→64 A/B가 기저를 3.3배 늘려도 오차가 τ 안에서 재배분만
    #              됐다 = 기저 부족이 아니라 케이스별 재배분 능력 부족. 그리고 창끝 오차가
    #              진폭(+0.28)보다 국소 곡률/빠른 시간상수(+0.50)와 더 상관한다.
    #   비용: FiLM은 **케이스당 1회**라 τ점 N개를 뽑는 롤아웃에서 추론 비용이 거의 안 는다
    #   (width 증량이 τ점마다 2.6배인 것과 반대). 구조 변경이라 체크포인트 비호환 — fresh 전용.
    cond_mode: str = "film"
    film_hidden: int = 128    # 조건화 MLP 은닉폭. 출력은 2·width·n (블록당 γ, β)

    # flip 전용 용량 (2026-07-30): energy gate로 켜지는 bottleneck adapter.
    #   dim=0이면 비활성. 실측상 flip/nonflip은 log(e_rel/e_flip) 기준 [-0.511, +0.009]로
    #   완전히 간격 분리되므로 center=-0.255(밴드 중앙)/temp=0.05면 g는 실질 이진값
    #   (nonflip 0.006, flip 0.995)이면서 미분 가능 — 미관측 영역에서도 정의된다.
    flip_adapter_dim: int = 0
    flip_adapter_n: int = 2     # 마지막 N개 블록 뒤에 삽입
    gate_center: float = -0.255
    gate_temp: float = 0.05

    # sparse 경로: g > gate_eps인 행만 adapter에 통과 → nonflip은 GEMM 자체를 건너뛴다.
    #   실측 nonflip g 최대 0.006 < eps 0.05 < flip g 최소 0.995라 절단오차는 dense 경로의
    #   nonflip 기여(≤0.006×adapter)와 같은 크기. false면 전 행 계산 후 g로 스케일(수학적 정확).
    gate_sparse: bool = True
    gate_eps: float = 0.05

    @property
    def gate_dim(self) -> int:
        return 1 if self.energy_gate else 0

    @property
    def gx_dim(self) -> int:
        # Fourier(2L) + t_norm(1) + IC(6) + param_embed + gate // concat trunk 입력 차원
        return (
            2 * self.fourier_l + 1 + self.ic_feat_dim
            + self.param_embed_dim + self.gate_dim
        )

    @property
    def cond_dim(self) -> int:
        # IC(6) + param_embed + gate // τ 무관 열만 = feats[:, 1:]
        #   τ에 의존하면 StateDerivs의 jvp dual 섭동이 γ·β를 오염시킨다 (gate idx와 같은 논리)
        return self.ic_feat_dim + self.param_embed_dim + self.gate_dim

    @property
    def trunk_in_dim(self) -> int:
        # film이면 trunk는 τ만 받는다 — 케이스는 γ·β로만 들어온다
        return 2 * self.fourier_l + 1 if self.cond_mode == "film" else self.gx_dim


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
    # 항별 clip 예산은 grad_clip 고정, RebalanceGradScales clamp 하한은 없음(0) — 둘 다 옵션이었으나
    #   전 config가 같은 값으로 수렴해 하드코딩 (term_grad_clip/grad_scale_min 제거, 2026-08-19).
    #   하한은 아무것도 보호하지 않음, 폭주 방지는 상한 역할 // [[grad-scale-floor-pinned]]
    grad_scale_max: float = 100.0 # RebalanceGradScales clamp 상한 // energy가 10에 붙박이라 완화
    replay_frac: float = 0.25
    adapter_only: bool = False    # finetune에서 flip_adapters 외 전부 동결 // 게이트는 출력만 막고
                                  #   gradient는 trunk를 통과하므로 동결이 nonflip 드리프트의 유일한
                                  #   완전 차단 (2026-08-17 실측: nonflip 결합 0.53×→1.62×)
    n_colloc_cases: int = 0       # finetune colloc case 풀 크기 (0 → max_cases)
    phys_ramp_center: int = 30    # e2 = epoch - warmup_epochs 기준 중심
    phys_ramp_width: int = 10     # sigmoid 폭 // ramp steepness
    lr_drop_epoch: int = 0        # 고정 LR 1회 하향 에폭 (0=비활성) // OdeScheduler와 별개
    lr_drop_to: float = 0.0       # lr_drop_epoch부터 적용할 LR 상한 // min(cur, lr_drop_to)
    ema_decay: float = 0.999      # Polyak weight EMA decay (0=비활성) // best.pt/평가 안정화
    roll_robust_k: float = 4.0    # rollout Cauchy 스케일 = k × 배치 median (0=비활성, 순수 mean) // heavy-tail 억제
    roll_balance_cases: int = 128 # RebalanceGradScales의 roll grad 노름 측정용 케이스 수 // 측정 비용 절감
    roll_balance_points: int = 12 # 위 측정의 윈도우 내 시점 수 // 학습 스텝의 roll_points와 동일 스케일

    # 스텝당 (k0, depth) 추첨 횟수 (1 = 종전) // roll gradient 분산 축소, 2026-08-01
    #   근거(grad_conflict.log): cos(total, roll)=+0.547로 스텝 방향 최대 기여인데
    #   draw-to-draw cos = -0.030 — 매 스텝 재현되지 않는 방향을 가리킨다. 원인은 케이스 수가
    #   아니라 추첨 개수다: k0/depth가 스텝당 각 1회라 roll_cases를 늘려도 이 분산은 안 줄어든다.
    #   케이스 예산을 n_draws로 쪼개므로 네트워크 evaluation 수는 불변 = 비용 거의 동일.
    #   k0는 윈도우 축으로 stratify하고, robust mean은 합친 뒤 1회만 건다.
    roll_draws: int = 1
    # Cauchy 스케일 c의 median EMA 계수 (0 = 종전, 매 스텝 배치 median 그대로) // 스케일 자체가
    #   추첨마다 흔들리면 그것도 노이즈원이다. 0.9면 c가 학습 진행을 따라가되 스텝 노이즈는 뺀다.
    roll_median_ema: float = 0.0
    # robust mean 형태: "cauchy" = c·log1p(l/c) (tail grad ∝ c/l), "huber" = min(l, 2√(δl)−δ)
    #   (tail grad ∝ √(δ/l)) — Cauchy가 tail gradient를 끊어 flip 평균을 악화시킨 것의 완화
    #   후보. δ/c = roll_robust_k × median 공용. // [[rollout-tail-cauchy-tradeoff]] 2026-08-17
    roll_robust_kind: str = "cauchy"

    # IC 항 스텝당 케이스 상한 (0 = 전체 active_cases) // IC는 세그먼트마다 반복되는 항이라
    #   비용이 max_cases × n_segments로 곱셈이다 (실측 2026-07-31: 10000×10 = 10만 행,
    #   스텝 시간의 69.4%. march_dt 1.0→0.5로 세그먼트가 배가되며 조용히 2배가 됐다).
    #   서브샘플이 사실상 공짜인 근거: 전체 대비 gradient cos가 4096/2048/1024/512에서
    #   0.9997/0.9992/0.9985/0.9964이고 노름비 ≈1.00 — 이미 감수 중인 data 미니배치
    #   draw 간 cos 0.9337보다 한 자릿수 조용하다. 손실이 평균 정규화라 유효 가중치도 불변.
    #   1024에서 154.6→62.8 ms/step (2.46×), 512부터 수확체감. // [[ic-term-dominates-step]]
    ic_max_n: int = 0

    # torch.compile: data/IC/kin/phys 손실 슬라이스를 Inductor로 컴파일 // 고정 shape 전제
    #   측정(2026-07-30, RTX 5070 Laptop, 실제 스텝 max_cases 10000/n_colloc 10240):
    #   eager+high 117.98 → compiled+high 71.12 ms/step (1.66×),
    #   eager+highest 139.97 → compiled+highest 98.12 ms/step (1.43×).
    #   FP64는 그대로 보존됨(eager 대비 rel err 2e-16; FP32 강등이면 2e-7). 컴파일 대기 ~27s.
    #   전제: 청크 shape이 몇 개로 고정. n_colloc을 colloc_chunk 배수로 두면 그래프 1개,
    #   나머지 청크가 생기면 shape별로 1개 추가(허용치 내). gate_sparse는 nonzero()가 동적
    #   shape라 자동으로 dense로 강제된다(PINNTrainer.__init__). // [[fp64-not-bottleneck]]
    use_compile: bool = True   # 전 config에서 true로 확정 — 프로브만 코드에서 끈다(probe_common)

    # FP32 matmul 정밀도. "high"=TF32 허용, "highest"=순수 FP32.
    #   2026-07-30 측정(구 config: n_colloc=10240, IC 전체): TF32에서 물리항(jvp 경유) gradient가
    #   순수 FP32 대비 cosine 0.70 (랜덤 입력 -0.10) — 반올림 노이즈가 방향의 상당 부분을 차지.
    #   2026-07-31 재측정(현재 config: n_colloc=20480/chunk 4096/ic_max_n=1024/per-term clip)에서
    #   뒤집힘: cos(TOTAL)=0.996 > 콜로케이션 추첨 노이즈 바닥 0.845 — n_colloc 증량이 반올림
    #   오차를 평균화로 씻어냄. "high"로 확정, toml 옵션 제거(2026-08-19) — 07-31 이전의
    #   "highest" 런은 전부 폐기라 resume 호환 불필요. 이상치(ic 항 cos 0.979)는 트랩 18.
    #   **추론은 별개** — eval/프로브는 TF32 off (트랩 25).
    matmul_precision: str = "high"

@dataclass
class CollocCfg:
    n_colloc: int = 20000
    ## 비용은 n_colloc에 선형 — colloc_chunk 단위 fwd+bwd 횟수가 그대로 늘어난다.
    ## 구 주석("시간은 FP64에서 소모되니 늘려도 무관")은 오측이었다. 측정(2026-07-30):
    ##   물리 슬라이스의 AngularAccel/GetEnergy를 FP32로 바꿔도 8.58 vs 8.89 ms — 차이가 노이즈.
    ##   FP64는 청크 행수만큼의 elementwise라 비용이 없고, 실제 병목은 FP32 jvp fwd(3.0ms)+bwd(5.5ms).
    ## use_compile 시에는 colloc_chunk의 배수로 두는 게 좋다 (나머지 청크가 그래프를 하나 더 만든다).

    colloc_flip_bias: float = 0.0   # finetune: flip case 하한 비율 (0=uniform) // 명시적 is_flip balancing

    ic_sigma: float = 0.0             # Phase 3: 물리 콜로케이션 IC 섭동 target (0=비활성)
    ic_sigma_warmup: int = 200        # sigma 0→target 선형 램프 에폭 (Phase 2 진입 후)


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
    # kin/phys/ic backward 청크 크기 // 0이면 batch_size 사용. data 배치와 분리해 FP64 physics
    #   transient peak 억제. 스위트스팟은 eager 시절 2048이었으나 use_compile 하에서 4096으로
    #   옮겨갔다 (실측 2026-07-31: 149.2 vs 154.6 ms/step, 5120은 148.2로 수확체감).
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
        energy_gate=n.get("energy_gate", False),
        hard_ic=n.get("hard_ic", "c1"),
        cond_mode=n.get("cond_mode", "film"),
        film_hidden=n.get("film_hidden", 128),
        flip_adapter_dim=n.get("flip_adapter_dim", 0),
        flip_adapter_n=n.get("flip_adapter_n", 2),
        gate_center=n.get("gate_center", -0.255),
        gate_temp=n.get("gate_temp", 0.05),
        gate_sparse=n.get("gate_sparse", True),
        gate_eps=n.get("gate_eps", 0.05),
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
        adapter_only=t.get("adapter_only", False),
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
        roll_balance_points=t.get("roll_balance_points", 12),
        ic_max_n=t.get("ic_max_n", 0),
        use_compile=t.get("use_compile", True),
        matmul_precision=t.get("matmul_precision", "high"),
    )

    colloc_cfg = CollocCfg(
        n_colloc=c["n_colloc"],
        colloc_flip_bias=c.get("colloc_flip_bias", 0.0),
        ic_sigma=c.get("ic_sigma", 0.0),
        ic_sigma_warmup=c.get("ic_sigma_warmup", 200),
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

    # 단위 주의 (2026-07-31 재설계): rel_tol_*은 에폭당 개선율 하한, patience_*는 에폭 수.
    #   StallTracker가 측정 간 경과 에폭으로 복리 환산하므로 측정 주기와 무관하게 게이트 간 잣대가 같다.
    #   rel_tol_*는 "그 지표가 아직 살아있는가"의 하한이지 목표 개선율이 아니다 — 작게 잡을 것.
    # phys 트래커(ema_beta/rel_tol)는 2026-07-31 제거 — patience는 이제 decay 시도 간격(에폭).
    #   구 config에 남은 ema_beta/rel_tol 키는 무시된다. 실효 동작은 종전과 동일(근거는 Scheduler.py).
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
    