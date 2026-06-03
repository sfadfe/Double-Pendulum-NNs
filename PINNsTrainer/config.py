import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class NetCfg:
    fourier_l: int = 24       # number of frequencies L // 주파수 개수
    f_min: float = 0.2        # Hz, lowest oscillation // 최저 진동 주파수
    f_max: float = 48.0       # Hz, < Nyquist 50Hz // 샤프 과도성분 상한

    in_params: int = 8 
    width: int = 256          # hidden layer width // 은닉층 너비
    n: int = 4                # ResidualBlock count // 잔차 블록 수
    out_dim: int = 2

    @property
    def gx_dim(self) -> int:
        # Fourier(2L) + raw t_norm(1) + 8 // 임베딩 차원 = 57
        return 2 * self.fourier_l + 1 + self.in_params


@dataclass
class TrainCfg:
    warmup_steps: int = 100           
    lambda_ic: float = 1.0
    lambda_data: float = 1.0
    lambda_phys_init: float = 0.01    # sigmoid start // 시작값
    lambda_phys_final: float = 1.0    # sigmoid end // 최종값
    lambda_energy_init: float = 0.01
    lambda_energy_final: float = 0.5
    lambda_sigmoid_steps: int = 5000  # sigmoid transition span // sigmoid 전이 길이
    lambda_sigmoid_mid: int = 2500    # sigmoid midpoint // sigmoid 중점
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    lbfgs_max_iter: int = 20
    lbfgs_history: int = 50
    lbfgs_batch: int = 8192    # L-BFGS 고정 미니배치 크기 // deterministic 손실
    lbfgs_accum: int = 2       # collocation grad accumulation 스텝 수 // peak VRAM 절감

    replay_frac: float = 0.25

@dataclass
class CollocCfg:
    seg_count: int = 3
    overlap_frac: float = 0.15

    n_colloc: int = 20000
    ## VRAM 공간 남으면 20000에서 30000~50000으로 늘릴 가능성 고려.
    ## 실제 학습 시간은 FP64 연산에서 많이 소모됨. 늘려도 상관없음

    lbfgs_n_colloc: int = 40000   # L-BFGS 단일구간 collocation 수 // accumulation step = 2 로 분할 처리

    rar_top_frac: float = 0.2         
    rar_bot_frac: float = 0.2         
    rar_every: int = 1000             


@dataclass
class DataCfg:
    g: float = 9.81

    t_data_max: float = 3.0 
    energy_eps: float = 1e-3

    nonflip_path: str = "data/nonflip_RK4_0_3s.npy"
    flip_path: str = "data/flip_RK4_0_3s.npy"
    scaler_name: str = "scaler.npy"
    batch_size: int = 8192


def LoadConfig(path):
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    t   = raw["train"]
    n   = raw["net"]
    o   = raw["optimizer"]
    s   = raw["scheduler"]
    lam = raw["lambda"]
    c   = raw["colloc"]
    d   = raw["data"]
    os_ = raw.get("ode_scheduler", {})

    total_opt_steps = t["max_epochs"] * c["seg_count"] * t["steps_per_segment"]
    sig_mid = int(total_opt_steps * t["sigmoid_frac"])

    net_cfg = NetCfg(
        fourier_l=n["fourier_l"],
        f_min=n["f_min"],
        f_max=n["f_max"],
        width=n["width"],
        n=n["n_blocks"],
    )

    train_cfg = TrainCfg(
        warmup_steps=o["warmup_steps"],
        lambda_ic=lam["ic"],
        lambda_data=lam["data"],
        lambda_phys_init=lam["phys_init"],
        lambda_phys_final=lam["phys_final"],
        lambda_energy_init=lam["energy_init"],
        lambda_energy_final=lam["energy_final"],
        lambda_sigmoid_mid=sig_mid,
        lambda_sigmoid_steps=sig_mid,
        lr=o["lr"],
        weight_decay=o["weight_decay"],
        grad_clip=o["grad_clip"],
        lbfgs_max_iter=o["lbfgs_max_iter"],
        lbfgs_history=o["lbfgs_history"],
        lbfgs_batch=o.get("lbfgs_batch", 8192),
        lbfgs_accum=o.get("lbfgs_accum", 2),
    )

    colloc_cfg = CollocCfg(
        seg_count=c["seg_count"],
        overlap_frac=c["overlap_frac"],
        n_colloc=c["n_colloc"],
        lbfgs_n_colloc=c.get("lbfgs_n_colloc", 40000),
        rar_top_frac=c["rar_top_frac"],
        rar_bot_frac=c["rar_bot_frac"],
        rar_every=c["rar_every"],
    )

    data_cfg = DataCfg(
        g=d["g"],
        t_data_max=d["t_data_max"],
        energy_eps=d["energy_eps"],
        batch_size=d["batch_size"],
        nonflip_path=d["nonflip_path"],
        scaler_name=d["scaler_name"],
    )

    ode_s_params = {
        "ema_beta":           os_.get("ema_beta",           0.85),
        "patience":           os_.get("patience",           80),
        "rel_tol":            os_.get("rel_tol",            0.005),
        "factor":             os_.get("factor",             0.8),
        "min_lr":             s.get("min_lr",               1e-6),
        "activate_threshold": os_.get("activate_threshold", 500.0),
    }

    return net_cfg, train_cfg, colloc_cfg, data_cfg, t, s, ode_s_params
