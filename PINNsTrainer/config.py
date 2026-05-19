from dataclasses import dataclass, field


@dataclass
class Config:
    # ===== Physics constants // 물리 상수 =====
    g: float = 9.81

    # ===== Data // 데이터 =====
    # dfss.py output: (N, 301, 13), channels // dfss.py 산출물 채널
    #   [t, th1, w1, th2, w2, m1, m2, L1, L2, sin t1, cos t1, sin t2, cos t2]
    t_data_max: float = 5.0          # raw t_norm = 2t/t_data_max - 1 (design 3.2) // raw t 정규화 기준
    energy_eps: float = 1.0          # |E0| 작을 때 분모 안정화 // energy denom floor

    # ===== Fourier feature (design 3.2) // 푸리에 임베딩 =====
    fourier_l: int = 24              # number of frequencies L // 주파수 개수
    f_min: float = 0.2               # Hz, lowest oscillation // 최저 진동 주파수
    f_max: float = 48.0              # Hz, < Nyquist 50Hz // 샤프 과도성분 상한

    # ===== Network (design 3.1, 3.4) // 네트워크 =====
    in_params: int = 8               # sin/cos t1, sin/cos t2, m1, m2, L1, L2 // 비-t 입력 8개
    hidden: int = 256                # width // 히든 폭
    n_blocks: int = 4                # ResidualBlock count // 잔차 블록 수
    out_dim: int = 2                 # [th1, th2]; w via autograd // 출력 (ω는 autograd)

    @property
    def gx_dim(self) -> int:
        # Fourier(2L) + raw t_norm(1) + 8 // 임베딩 차원 = 57
        return 2 * self.fourier_l + 1 + self.in_params

    # ===== Loss / lambda curriculum (design 4.3, 4.4) // 손실·λ 커리큘럼 =====
    warmup_steps: int = 100          # L_*_0 normalization warmup // 정규화 워밍업 스텝
    lambda_ic: float = 1.0           # fixed // 고정
    lambda_data: float = 1.0         # fixed // 고정
    lambda_phys_init: float = 0.01   # sigmoid start // 시작값
    lambda_phys_final: float = 1.0   # sigmoid end // 최종값
    lambda_energy_init: float = 0.01
    lambda_energy_final: float = 0.5
    lambda_sigmoid_steps: int = 5000  # sigmoid transition span // sigmoid 전이 길이
    lambda_sigmoid_mid: int = 2500    # sigmoid midpoint // sigmoid 중점

    # ===== Time-marching (design 4.1) // 타임마칭 =====
    seg_count: int = 3               # micro segments within a file window // 파일 구간 내 미시 구간 수
    overlap_frac: float = 0.15       # delta = 10~20% of segment // 구간 겹침 비율

    # ===== Collocation (design 4.2) // 콜로케이션 =====
    n_colloc: int = 20000            # points per segment (kept fixed under RAR) // 구간당 포인트 수
    rar_top_frac: float = 0.2        # densify near top residual // 잔차 상위 밀집
    rar_bot_frac: float = 0.2        # drop low residual // 잔차 하위 제거
    rar_every: int = 1000            # RAR period (steps) // RAR 주기

    # ===== Mixed precision (design 4.6) // 혼합 정밀도 =====
    physics_float64: bool = True     # residual path full float64 (option A) // 잔차 경로 float64

    # ===== Optimizer (design 4.5) // 최적화 =====
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    lbfgs_max_iter: int = 20
    lbfgs_history: int = 50

    # ===== Difficulty curriculum (design 4.7) // 난이도 커리큘럼 =====
    replay_frac: float = 0.25        # easy-pool replay ratio (20~30%) // replay 혼합 비율

    # ===== Files // 파일 =====
    nonflip_path: str = "data/nonflip_RK4_0_3s.npy"
    flip_path: str = "data/flip_RK4_0_3s.npy"
    scaler_name: str = "scaler.npy"
    batch_size: int = 8192
