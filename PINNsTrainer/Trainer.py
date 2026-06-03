import os

import torch

from .Dataset import Dataset
from .Loss import Loss
from .networks import Networks
from .ODE import Physics
from .Trainstep import LambdaBalance, TimeMarching


class PINNTrainer(Networks, Physics, Dataset, Loss, LambdaBalance, TimeMarching):
    def __init__(self, net_cfg, train_cfg, colloc_cfg, data_cfg, device):
        Networks.__init__(self, net_cfg, data_cfg)    # nn.Module 초기화 + 백본
        self.netCfg = net_cfg
        self.trainCfg = train_cfg
        self.collocCfg = colloc_cfg
        self.dataCfg = data_cfg
        self.device = torch.device(device)
        self.device_type = self.device.type           # autocast용 ("cuda"/"cpu")
        self.g = data_cfg.g
        self.to(self.device)

    def Setup(self, base_dir, run_dir=None):
        cfg = self.dataCfg
        self.LoadData(os.path.join(base_dir, cfg.nonflip_path))
        self.ComputeScaler(run_dir if run_dir is not None else base_dir)
        self.to(self.device)
        self.data = self.data.to(self.device)
        self.params_raw = self.params_raw.to(self.device)
        self.ic_trig = self.ic_trig.to(self.device)
        self.t_grid = self.t_grid.to(self.device)

        t_min = float(self.t_grid[0])
        t_max = float(self.t_grid[-1])
        self.BuildSegments(t_min, t_max)
        self.InitLambda()
        self.SetOptimizerAdamW()
        self.best_metric = float("inf")
        self.best_ode_metric = float("inf")

    def SetOptimizerAdamW(self):
        c = self.trainCfg
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=c.lr, weight_decay=c.weight_decay
        )
        self.using_lbfgs = False

    def SwitchToLBFGS(self):
        c = self.trainCfg
        self.optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=1.0,
            max_iter=c.lbfgs_max_iter,
            history_size=c.lbfgs_history,
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )
        self.using_lbfgs = True

    def IsNaN(self, losses):
        # NaN 감지 시 grad 초기화 후 True 반환:  학습 루프에서 step skip
        if any(torch.isnan(v) for v in losses.values()):
            self.optimizer.zero_grad()
            nan_keys = [k for k, v in losses.items() if torch.isnan(v)]
            print(f"[NaN] skipping step — affected: {nan_keys}")
            return True
        return False

    def _SaveCheckpoint(self, path, step, metric):
        tmp = path + ".tmp"
        torch.save(
            {
                "step": step,
                "metric": metric,
                "best_metric": self.best_metric,
                "best_ode_metric": self.best_ode_metric,
                "model_state": self.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "l0": self.l0,
            },
            tmp,
        )
        os.replace(tmp, path)

    def SaveLatest(self, ckpt_dir, step, metric):
        self._SaveCheckpoint(os.path.join(ckpt_dir, "latest.pt"), step, metric)

    def MaybeSaveBest(self, ckpt_dir, step, metric):
        # metric 개선 시에만 best.pt 갱신 // best model 보존
        if metric < self.best_metric:
            self.best_metric = metric
            self._SaveCheckpoint(os.path.join(ckpt_dir, "best.pt"), step, metric)
            return True
        return False

    def MaybeSaveBestODE(self, ckpt_dir, step, ode_loss):
        # ODE loss 기준 best 갱신 // physics 수렴 best model 보존
        if ode_loss < self.best_ode_metric:
            self.best_ode_metric = ode_loss
            self._SaveCheckpoint(os.path.join(ckpt_dir, "best_ode.pt"), step, ode_loss)
            return True
        return False

    def LoadCheckpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.best_metric = ckpt.get("best_metric", float("inf"))
        self.best_ode_metric = ckpt.get("best_ode_metric", self.best_ode_metric)
        self.l0 = ckpt.get("l0", self.l0)
        return ckpt["step"], ckpt["metric"]
