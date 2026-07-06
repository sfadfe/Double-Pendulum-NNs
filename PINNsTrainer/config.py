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
    n_colloc_cases: int = 0       # finetune RAR 풀 크기 (0 → max_cases)

@dataclass
class CollocCfg:
    seg_count: int = 3
    overlap_frac: float = 0.15

    n_colloc: int = 20000
    ## VRAM 공간 남으면 20000에서 30000~50000으로 늘릴 가능성 고려.
    ## 실제 학습 시간은 FP64 연산에서 많이 소모됨. 늘려도 상관없음

    rar_top_frac: float = 0.2
    rar_bot_frac: float = 0.2
    rar_every: int = 1000
    rar_jitter_frac: float = 0.08   # τ jitter = ± frac * march_dt
    colloc_flip_bias: float = 0.0   # finetune: flip case 하한 비율 (0=uniform)

    ic_sigma: float = 0.0             # Phase 3: 물리 콜로케이션 IC 섭동 target (0=비활성)
    ic_sigma_warmup: int = 200        # sigma 0→target 선형 램프 에폭 (Phase 2 진입 후)


@dataclass
class DataCfg:
    g: float = 9.81

    t_data_max: float = 3.0
    march_dt: float = 1.0     # time-marching window duration (s) // 플로우맵 윈도우 길이 — 네트워크가 보는 상대시간 τ∈[0,march_dt]
    energy_eps: float = 1e-3
    phys_eps: float = 1.0     # EOM 상대잔차 분모 바닥값 // floor for relative physics residual denom (≈ small frac of typical |dω/dt|)

    nonflip_path: str = "data/nonflip_RK4_0_3s.npy"
    flip_path: str = "data/flip_RK4_0_3s.npy"
    data_path: str = ""           # finetune mixed dataset (empty → nonflip_path)
    replay_path: str = ""         # finetune replay buffer (empty → nonflip_path)
    scaler_name: str = "scaler.npy"
    scaler_extra_omega: str = ""  # pretrain scaler ω RMS에 포함할 추가 궤적(예: mixed) — flip OOD 방지
    batch_size: int = 8192


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
    )

    colloc_cfg = CollocCfg(
        seg_count=c["seg_count"],
        overlap_frac=c["overlap_frac"],
        n_colloc=c["n_colloc"],
        rar_top_frac=c["rar_top_frac"],
        rar_bot_frac=c["rar_bot_frac"],
        rar_every=c["rar_every"],
        rar_jitter_frac=c.get("rar_jitter_frac", 0.08),
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
        batch_size=d["batch_size"],
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
        "activate_threshold": os_.get("activate_threshold", 500.0),
        "rel_tol_decay":      os_.get("rel_tol_decay",      0.75),
        "patience_decay":     os_.get("patience_decay",     1.0),   # LR 감쇄마다 patience에 곱할 비율 (1.0 = 비활성)
        "min_patience":       os_.get("min_patience",       10),    # patience 하한 // 후반 노이즈 과민반응 방지
        # Warm restart: 무개선 감쇄 연속 감지 시 LR 복원 // 기본값 = 비활성 (구 config 동작 불변)
        "min_rel_tol":        os_.get("min_rel_tol",        0.0),   # rel_tol 하한 겸 감쇄 생산성 판정 기준
        "max_bad_decays":     os_.get("max_bad_decays",     10**9), # 연속 무개선 감쇄 허용 횟수 (초과 시 restart)
        "restart_lr0":        os_.get("restart_lr0",        0.0),   # 첫 restart 복원 LR (0 = restart 비활성)
        "restart_decay":      os_.get("restart_decay",      0.5),   # restart마다 복원 LR에 곱할 비율 // SGDR 진폭 감쇄
        "restart_cooldown":   os_.get("restart_cooldown",   30),    # restart 후 stall 동결 에폭
    }

    return net_cfg, train_cfg, colloc_cfg, data_cfg, t, ode_s_params
    