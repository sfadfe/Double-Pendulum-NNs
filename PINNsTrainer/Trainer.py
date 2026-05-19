import os

import torch

from .Dataset import Dataset
from .Loss import Loss
from .networks import Networks
from .ODE import Physics
from .Trainstep import LambdaBalance, TimeMarching


class PINNTrainer(Networks, Physics, Dataset, Loss, LambdaBalance, TimeMarching):
    # Mixin 합성: 연속시간 PINN 학습 객체 (학습 루프는 미구현 — 메소드만 제공)
    def __init__(self, config, device):
        Networks.__init__(self, config)               # nn.Module 초기화 + 백본
        self.config = config
        self.device = torch.device(device)
        self.device_type = self.device.type           # autocast용 ("cuda"/"cpu")
        self.g = config.g
        self.to(self.device)

    def Setup(self, base_dir):
        # 데이터/스케일러/구간/λ 초기화 // training-ready state 구성
        cfg = self.config
        self.LoadData(os.path.join(base_dir, cfg.nonflip_path))   # 단계 1: 비플립 풀
        self.ComputeScaler(base_dir)
        self.to(self.device)                          # 새 buffer(scaler) 디바이스 이동

        t_min = float(self.t_grid[0])
        t_max = float(self.t_grid[-1])
        self.BuildSegments(t_min, t_max)
        self.InitLambda()
        self.SetOptimizerAdamW()

    def SetOptimizerAdamW(self):
        # Opt-A: AdamW (design 4.5) // 빠른 수렴
        c = self.config
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=c.lr, weight_decay=c.weight_decay
        )
        self.using_lbfgs = False

    def SwitchToLBFGS(self):
        # Opt-B: L-BFGS 정밀 수렴 (design 4.5) // 전환만 제공, 호출은 학습 루프
        c = self.config
        self.optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=1.0,
            max_iter=c.lbfgs_max_iter,
            history_size=c.lbfgs_history,
            line_search_fn="strong_wolfe",
        )
        self.using_lbfgs = True
