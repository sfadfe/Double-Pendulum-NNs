import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class NetCfg:
    fourier_l: int = 32       # number of frequencies L // 주파수 개수
    f_min: float = 0.2        # Hz, lowest oscillation // 최저 진동 주파수
    f_max: float = 56.0       # Hz, < Nyquist 50Hz // 샤프 과도성분 상한

    ic_feat_dim: int = 6      # 4 trig(θ0) + 2 ω(0) // raw m/L 제외 (ParamEmbed로 대체)
    param_embed_dim: int = 32 # ParamEmbed 출력 차원 // option B3
    width: int = 384          # hidden layer width // 은닉층 너비 (M tier)
    n: int = 6                # ResidualBlock count // 잔차 블록 수
    out_dim: int = 4          # state output [Δθ1, Δθ2, ω1, ω2] // 상태공간 출력

    @property
    def feat_dim(self) -> int:
        # τ(1) + IC(6) + param_embed // Dataset._BuildFeats 출력 폭
        return 1 + self.ic_feat_dim + self.param_embed_dim

    @property
    def gx_dim(self) -> int:
        # Fourier(2L) + t_norm(1) + IC(6) + param_embed // trunk 입력 차원
        return 2 * self.fourier_l + 1 + self.ic_feat_dim + self.param_embed_dim


@dataclass
class TrainCfg:
    warmup_steps: int = 100
    lambda_ic: float = 1.0
    lambda_data: float = 1.0
    lambda_kin: float = 1.0
    lambda_phys: float = 1.0
    lambda_energy: float = 0.5
    lambda_roll: float = 1.0      # rollout-aware (pushforward) 손실 base 가중치 // 윈도우 핸드오프 교정
    lambda_min: float = 0.3       # ReLoBRaLo λ_relo clamp 하한 // B가 절대 스케일 담당, relo는 nudge만
    lambda_max: float = 3.0       # ReLoBRaLo λ_relo clamp 상한
    relobralo_tau: float = 0.1    # softmax temperature // 낮을수록 수렴 속도 차이에 민감
    relobralo_alpha: float = 0.999  # EMA smoothing // 클수록 lambda 변화 느림
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    replay_frac: float = 0.25
    n_colloc_cases: int = 0       # finetune colloc case 풀 크기 (0 → max_cases)
    phys_ramp_epochs: int = 60    # sigmoid ramp 참고 길이 (문서·로그) // reference span
    phys_ramp_center: int = 30    # e2 = epoch - warmup_epochs 기준 중심
    phys_ramp_width: int = 10     # sigmoid 폭 // ramp steepness
    lr_drop_epoch: int = 0        # 고정 LR 1회 하향 에폭 (0=비활성) // OdeScheduler와 별개
    lr_drop_to: float = 0.0       # lr_drop_epoch부터 적용할 LR 상한 // min(cur, lr_drop_to)
    ema_decay: float = 0.999      # Polyak weight EMA decay (0=비활성) // best.pt/평가 안정화
    roll_robust_k: float = 4.0    # rollout Cauchy 스케일 = k × 배치 median (0=비활성, 순수 mean) // heavy-tail 억제
    roll_balance_cases: int = 128 # RebalanceGradScales의 roll grad 노름 측정용 케이스 수 // 측정 비용 절감
    roll_balance_points: int = 12 # 위 측정의 윈도우 내 시점 수 // 학습 스텝의 roll_points와 동일 스케일

@dataclass
class CollocCfg:
    seg_count: int = 3
    overlap_frac: float = 0.15

    n_colloc: int = 20000
    ## VRAM 공간 남으면 20000에서 30000~50000으로 늘릴 가능성 고려.
    ## 실제 학습 시간은 FP64 연산에서 많이 소모됨. 늘려도 상관없음

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
    kin_eps: float = 1.0      # kin 상대정규화 분모 바닥값 // floor for mean(dθ/dτ²) denom (dθ/dτ→0 blowup 방지, ≈ small frac of physical ω²)

    nonflip_path: str = "data/nonflip_RK4_0_3s.npy"
    flip_path: str = "data/flip_RK4_0_3s.npy"
    data_path: str = ""           # finetune mixed dataset (empty → nonflip_path)
    replay_path: str = ""         # finetune replay buffer (empty → nonflip_path)
    scaler_name: str = "scaler.npy"
    scaler_extra_omega: str = ""  # pretrain scaler ω RMS에 포함할 추가 궤적(예: mixed) — flip OOD 방지
    batch_size: int = 8192
    colloc_chunk: int = 0  # kin/phys/ic backward 청크 크기 // 0이면 batch_size 사용. data 배치와 분리해 FP64 physics transient peak 억제 (측정: 2048이 sweet spot)


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
    )

    train_cfg = TrainCfg(
        warmup_steps=o["warmup_steps"],
        lambda_ic=lam["ic"],
        lambda_data=lam["data"],
        lambda_kin=lam.get("kin", 1.0),
        lambda_phys=lam.get("phys", 1.0),
        lambda_energy=lam.get("energy", 0.5),
        lambda_roll=lam.get("roll", 1.0),
        lambda_min=lam.get("lambda_min", 0.3),
        lambda_max=lam.get("lambda_max", 3.0),
        relobralo_tau=lam.get("relobralo_tau", 0.1),
        relobralo_alpha=lam.get("relobralo_alpha", 0.999),
        lr=o["lr"],
        weight_decay=o["weight_decay"],
        grad_clip=o["grad_clip"],
        replay_frac=t.get("replay_frac", 0.25),
        n_colloc_cases=t.get("n_colloc_cases", c.get("n_colloc_cases", 0)),
        phys_ramp_epochs=t.get("phys_ramp_epochs", 60),
        phys_ramp_center=t.get("phys_ramp_center", 30),
        phys_ramp_width=t.get("phys_ramp_width", 10),
        lr_drop_epoch=t.get("lr_drop_epoch", 0),
        lr_drop_to=t.get("lr_drop_to", 0.0),
        ema_decay=t.get("ema_decay", 0.999),
        roll_robust_k=t.get("roll_robust_k", 4.0),
        roll_balance_cases=t.get("roll_balance_cases", 128),
        roll_balance_points=t.get("roll_balance_points", 12),
    )

    colloc_cfg = CollocCfg(
        seg_count=c["seg_count"],
        overlap_frac=c["overlap_frac"],
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
        replay_path=d.get("replay_path", d.get("nonflip_path", d["nonflip_path"])),
        scaler_name=d["scaler_name"],
        scaler_extra_omega=d.get("scaler_extra_omega", ""),
    )

    ode_s_params = {
        "ema_beta":           os_.get("ema_beta",           0.85),
        "patience":           os_.get("patience",           80),
        "rel_tol":            os_.get("rel_tol",            0.005),
        "factor":             os_.get("factor",             0.8),
        "min_lr":             os_.get("min_lr", s.get("min_lr", 1e-6)),
        "always_active":        os_.get("always_active",        True),
        "activate_threshold": os_.get("activate_threshold", 500.0),
        "rel_tol_decay":      os_.get("rel_tol_decay",      0.75),
        "patience_decay":     os_.get("patience_decay",     1.0),   # LR 감쇄마다 patience에 곱할 비율 (1.0 = 비활성)
        "min_patience":       os_.get("min_patience",       10),    # patience 하한 // 후반 노이즈 과민반응 방지
        "min_rel_tol":        os_.get("min_rel_tol",        0.0),   # rel_tol 하한
        "phase2_lr":          os_.get("phase2_lr",          0.0),   # Phase 2 진입 1회 LR (0 = 비활성)
        # --- Val/Extrap veto gate // 개선 중이면 LR decay 보류 ---
        "patience_val":       os_.get("patience_val",       3),     # val 미개선 연속 측정 횟수 (× val_interval)
        "patience_ext":       os_.get("patience_ext",       2),     # extrap 미개선 연속 측정 횟수 (× extrap_sched_interval)
        "rel_tol_val":        os_.get("rel_tol_val",        0.02),  # val 개선 판정 상대 임계
        "rel_tol_ext":        os_.get("rel_tol_ext",        0.05),  # extrap 개선 판정 상대 임계
        "ema_beta_val":       os_.get("ema_beta_val",       0.9),   # val EMA smoothing
        "ema_beta_ext":       os_.get("ema_beta_ext",       0.9),   # extrap EMA smoothing
    }

    return net_cfg, train_cfg, colloc_cfg, data_cfg, t, ode_s_params
    