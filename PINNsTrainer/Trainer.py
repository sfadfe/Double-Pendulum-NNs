import os
from pathlib import Path

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
        self.device_type = self.device.type          
        self.g = data_cfg.g
        self.to(self.device)

    def SetupFinetune(self, base_dir, run_dir, pretrain_dir, seed=42):
        # Mixed data + replay buffer + frozen pretrain scaler // 파인튜닝 초기화
        cfg = self.dataCfg
        mixed_path = Path(cfg.data_path)
        if not mixed_path.is_absolute():
            mixed_path = Path(base_dir) / cfg.data_path
        self.LoadMixedData(str(mixed_path))
        if self.replay_n_case == 0:
            raise RuntimeError(
                "meta에 replay_idx가 없음 — state/dfss.py mixed로 재빌드하세요"
            )

        scaler_path = Path(pretrain_dir) / cfg.scaler_name
        self.LoadScaler(str(scaler_path))
        self.to(self.device)

        self.data = self.data.to(self.device)
        self.params_raw = self.params_raw.to(self.device)
        if self.is_flip is not None:
            self.is_flip = self.is_flip.to(self.device)
        if self.replay_idx is not None:
            self.replay_idx = self.replay_idx.to(self.device)

        self.t_grid = self.t_grid.to(self.device)
        self.dt = float(self.t_grid[1] - self.t_grid[0])
        t_min = float(self.t_grid[0])
        t_max = float(self.t_grid[-1])
        self.BuildSegments(t_min, t_max)
        self.InitLambda()
        self.SetOptimizerAdamW()
        self.best_metric = float("inf")
        self.best_ode_metric = float("inf")
        self.best_extrap_metric = float("inf")
        self._replay_active = False
        self._colloc_inited = False
        return self.n_case

    def LoadWeights(self, path):
        # Pretrain weights only; optimizer/scheduler fresh start // 가중치만 이어받기
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt["model_state"])
        saved_gs = {k: v for k, v in ckpt.get("grad_scale", {}).items() if k != "roll"}
        self.grad_scale = {**self.grad_scale, **saved_gs}
        self.roll_ramp = ckpt.get("roll_ramp", self.roll_ramp)
        self.phys_ramp = ckpt.get("phys_ramp", self.phys_ramp)
        self._lr_drop_done = ckpt.get("lr_drop_done", self._lr_drop_done)
        self._phys_balanced = ckpt.get("phys_balanced", self._phys_balanced)
        return ckpt.get("step", 0)

    def Setup(self, base_dir, run_dir=None):
        cfg = self.dataCfg
        self.LoadData(os.path.join(base_dir, cfg.nonflip_path))
        extra_omega = []
        if cfg.scaler_extra_omega:
            p = Path(cfg.scaler_extra_omega)
            extra_omega.append(str(p if p.is_absolute() else Path(base_dir) / p))
        self.ComputeScaler(
            run_dir if run_dir is not None else base_dir, extra_omega_paths=extra_omega
        )
        self.to(self.device)
        self.data = self.data.to(self.device)
        self.params_raw = self.params_raw.to(self.device)
        self.t_grid = self.t_grid.to(self.device)
        self.dt = float(self.t_grid[1] - self.t_grid[0])

        t_min = float(self.t_grid[0])
        t_max = float(self.t_grid[-1])
        self.BuildSegments(t_min, t_max)
        self.InitLambda()
        self.SetOptimizerAdamW()
        self.best_metric = float("inf")
        self.best_ode_metric = float("inf")
        self.best_extrap_metric = float("inf")

    def SetOptimizerAdamW(self):
        c = self.trainCfg
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=c.lr, weight_decay=c.weight_decay
        )
    def IsNaN(self, losses):
        # NaN 감지 시 grad 초기화 후 True 반환:  학습 루프에서 step skip
        if any(torch.isnan(v) for v in losses.values()):
            self.optimizer.zero_grad()
            nan_keys = [k for k, v in losses.items() if torch.isnan(v)]
            print(f"[NaN] skipping step — affected: {nan_keys}")
            return True
        return False

    def _SaveCheckpoint(self, path, step, metric, sched_state=None):
        tmp = path + ".tmp"
        torch.save(
            {
                "step": step,
                "metric": metric,
                "best_metric": self.best_metric,
                "best_ode_metric": self.best_ode_metric,
                "best_extrap_metric": self.best_extrap_metric,
                "model_state": self.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "grad_scale": self.grad_scale,        # B 균등화 스케일 // 10에폭마다 EMA 갱신 → resume 복원 필수
                "roll_ramp": self.roll_ramp,          # rollout 0→1 램프 계수 // resume 시 램프 위치 복원
                "phys_ramp": self.phys_ramp,          # physics sigmoid ramp // resume 시 ramp 위치 복원
                "lr_drop_done": self._lr_drop_done,   # 고정 LR drop 1회 완료 // resume 시 중복 로그 방지
                "phys_balanced": self._phys_balanced, # Phase 2 진입(첫 B 호출) 여부
                "sched_state": sched_state,  # OdeScheduler 내부 상태 // scheduler resume 복원용
            },
            tmp,
        )
        os.replace(tmp, path)

    def SaveLatest(self, ckpt_dir, step, metric, sched_state=None):
        self._SaveCheckpoint(os.path.join(ckpt_dir, "latest.pt"), step, metric, sched_state)

    def MaybeSaveBest(self, ckpt_dir, step, metric, sched_state=None):
        # metric 개선 시에만 best.pt 갱신 // best model 보존
        if metric < self.best_metric:
            self.best_metric = metric
            self._SaveCheckpoint(os.path.join(ckpt_dir, "best.pt"), step, metric, sched_state)
            return True
        return False

    def MaybeSaveBestODE(self, ckpt_dir, step, ode_loss, sched_state=None):
        # ODE loss 기준 best 갱신 // physics 수렴 best model 보존
        if ode_loss < self.best_ode_metric:
            self.best_ode_metric = ode_loss
            self._SaveCheckpoint(os.path.join(ckpt_dir, "best_ode.pt"), step, ode_loss, sched_state)
            return True
        return False

    def MaybeSaveBestExtrap(self, ckpt_dir, step, extrap_loss, sched_state=None):
        # 외삽(마칭 t_data_max 이후) 오차 기준 best 갱신 // 물리 외삽 능력 best model 보존
        if extrap_loss < self.best_extrap_metric:
            self.best_extrap_metric = extrap_loss
            self._SaveCheckpoint(os.path.join(ckpt_dir, "best_extrap.pt"), step, extrap_loss, sched_state)
            return True
        return False

    def LoadCheckpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.best_metric = ckpt.get("best_metric", float("inf"))
        self.best_ode_metric = ckpt.get("best_ode_metric", self.best_ode_metric)
        self.best_extrap_metric = ckpt.get("best_extrap_metric", self.best_extrap_metric)
        # merge so newly-added keys survive resume from older checkpoints // 구 ckpt 키 누락 방지
        saved_gs = {k: v for k, v in ckpt.get("grad_scale", {}).items() if k != "roll"}  # 구 ckpt의 roll 슬롯 재활용 잔재 제거
        self.grad_scale = {**self.grad_scale, **saved_gs}
        self.roll_ramp = ckpt.get("roll_ramp", self.roll_ramp)
        self.phys_ramp = ckpt.get("phys_ramp", self.phys_ramp)
        self._lr_drop_done = ckpt.get("lr_drop_done", self._lr_drop_done)
        self._phys_balanced = ckpt.get("phys_balanced", self._phys_balanced)
        return ckpt["step"], ckpt["metric"], ckpt.get("sched_state", None)
