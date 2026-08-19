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
        if train_cfg.use_compile and self.gate_sparse and len(self.flip_adapters) > 0:
            # nonzero()가 데이터 의존 shape을 만들어 스텝마다 재컴파일을 유발한다.
            # dense 경로는 수학적으로 더 정확한 쪽(sparse가 g<gate_eps 기여를 버리는 근사)이라
            # 강제해도 잃는 게 없다 — 실측 nonflip g 0.006 × adapter만큼의 절단오차가 사라진다.
            self.gate_sparse = False
            print("[compile] gate_sparse → dense 경로 강제 (nonzero()는 동적 shape)")
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

    def _CheckHardIC(self, ckpt, path):
        # 출력 ansatz 불일치 차단 // trap 17과 같은 계열: shape이 같아 조용히 로드되지만 의미가 다르다
        #   hard_ic가 다르면 head 출력이 "A_θ/A_ω 계수"냐 "절대 Δθ/ω"냐가 뒤바뀐다.
        #   키가 없는 구 ckpt는 hard_ic="off" 시절이므로 그렇게 간주한다.
        want = getattr(self.netCfg, "hard_ic", "off")
        got = ckpt.get("hard_ic", "off")
        if want != got:
            raise ValueError(
                f"hard_ic 불일치: config={want!r} vs ckpt={got!r} ({path}). "
                "출력 ansatz가 달라 가중치를 그대로 쓸 수 없다 — config를 맞추거나 fresh 학습할 것."
            )

    def _AdaptState(self, state):
        # 구조가 커진 ckpt 로드용 어댑터 // energy_gate / flip_adapter가 pretrain 대비 순수 증분이 되도록
        #  1) 입력열이 부족한 (out, in) 가중치 뒤에 0열을 덧붙인다 (gx_dim 증가분).
        #  2) ckpt에 없는 신규 키는 현재 초기값을 그대로 쓴다 (flip_adapter는 up이 zero-init).
        # 두 경우 모두 새 파라미터의 기여가 0이라 로드 직후 모델은 ckpt와 수치적으로 동일하다.
        own = self.state_dict()
        out = dict(own)
        for k, v in state.items():
            w = own.get(k)
            if w is None:
                print(f"[load] {k}: ckpt에만 있음 — 무시 // unexpected key")
                continue
            if (
                v.dim() == 2 and w.dim() == 2
                and w.shape[0] == v.shape[0] and w.shape[1] > v.shape[1]
            ):
                n_old = v.shape[1]
                v = torch.cat([v, v.new_zeros(v.shape[0], w.shape[1] - n_old)], dim=1)
                print(f"[load] {k}: in {n_old} → {w.shape[1]} (zero-pad)")
            out[k] = v
        missing = [k for k in own if k not in state]
        if missing:
            print(f"[load] 신규 파라미터 {len(missing)}개 초기값 유지: {missing[:4]}"
                  f"{' ...' if len(missing) > 4 else ''}")
        return out

    def LoadWeights(self, path):
        # Pretrain weights only; optimizer/scheduler fresh start // 가중치만 이어받기
        ckpt = torch.load(path, map_location=self.device)
        self._CheckHardIC(ckpt, path)
        self.load_state_dict(self._AdaptState(ckpt["model_state"]))
        saved_gs = {k: v for k, v in ckpt.get("grad_scale", {}).items() if k != "roll"}
        self.grad_scale = {**self.grad_scale, **saved_gs}
        self.roll_ramp = ckpt.get("roll_ramp", self.roll_ramp)
        self.phys_ramp = ckpt.get("phys_ramp", self.phys_ramp)
        self._phys_balanced = ckpt.get("phys_balanced", self._phys_balanced)
        self._ResetEMA()   # 파인튜닝: 로드한 pretrain 가중치로 EMA 재초기화
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
        params = [p for p in self.parameters() if p.requires_grad]  # adapter_only 동결 존중
        self.optimizer = torch.optim.AdamW(
            params, lr=c.lr, weight_decay=c.weight_decay
        )

    def FreezeToAdapters(self):
        # Adapter-only finetune: freeze all but flip_adapters // trunk 드리프트 구조 차단
        #   FlipAdapter의 격리 보증은 자기 기여에만 성립하고 flip gradient는 공유 trunk를
        #   통과한다 — nonflip 결합 0.53×→1.62× 악화의 원인 (md/window_scale_coupling.md §9).
        #   동결 후 nonflip 출력 변화는 g·|adapter| ≤ 0.006·|adapter|로 유계.
        #   SetOptimizerAdamW 전에 불러야 옵티마이저가 trainable만 받는다.
        if len(self.flip_adapters) == 0:
            raise ValueError(
                "adapter_only = true인데 flip_adapter_dim = 0 — 학습할 파라미터가 없다"
            )
        for p in self.parameters():
            p.requires_grad_(False)
        n_train = 0
        for ad in self.flip_adapters:
            for p in ad.parameters():
                p.requires_grad_(True)
                n_train += p.numel()
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[freeze] adapter-only: {n_train:,}/{n_total:,} 파라미터만 학습")

    # --- Weight EMA (Polyak averaging) — 후반 진동 매끈화 + best.pt 안정화 // B3 ---
    def InitEMA(self, decay):
        # decay<=0 → 비활성 (ema_state=None, 기존 동작 유지) // opt-out
        self.ema_decay = float(decay)
        self._ema_backup = None
        if self.ema_decay > 0.0:
            self.ema_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        else:
            self.ema_state = None

    def _ResetEMA(self):
        # 가중치 교체(LoadWeights/resume) 후 EMA를 현재 가중치로 재초기화 // stale shadow 방지
        if self.ema_state is not None:
            self.ema_state = {k: v.detach().clone() for k, v in self.state_dict().items()}

    @torch.no_grad()
    def UpdateEMA(self):
        # 매 optimizer.step 후 호출 // shadow ← decay·shadow + (1-decay)·raw
        if self.ema_state is None:
            return
        d = self.ema_decay
        for k, v in self.state_dict().items():
            s = self.ema_state[k]
            if torch.is_floating_point(v):
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)  # 상수 버퍼는 EMA해도 불변
            else:
                s.copy_(v)

    def ApplyEMA(self):
        # 평가용: raw 백업 후 EMA 가중치 로드 // returns True if applied
        if self.ema_state is None:
            return False
        self._ema_backup = {k: v.detach().clone() for k, v in self.state_dict().items()}
        self.load_state_dict(self.ema_state)
        return True

    def RestoreRaw(self):
        # ApplyEMA 이후 학습용 raw 가중치 복원 // 학습·latest.pt는 항상 raw
        if self._ema_backup is None:
            return
        self.load_state_dict(self._ema_backup)
        self._ema_backup = None
    def _SaveCheckpoint(self, path, step, metric, sched_state=None, model_state=None):
        # model_state 주어지면 그 가중치로 저장 (best는 EMA 가중치) // latest는 raw(기본)
        tmp = path + ".tmp"
        torch.save(
            {
                "step": step,
                "metric": metric,
                "best_metric": self.best_metric,
                "best_ode_metric": self.best_ode_metric,
                "best_extrap_metric": self.best_extrap_metric,
                "hard_ic": getattr(self.netCfg, "hard_ic", "off"),   # 출력 ansatz — 로드 시 대조 (trap 23)
                "model_state": model_state if model_state is not None else self.state_dict(),
                "ema_state": self.ema_state,          # Polyak shadow // resume 시 EMA 궤적 복원
                "optimizer_state": self.optimizer.state_dict(),
                "grad_scale": self.grad_scale,        # B 균등화 스케일 // 10에폭마다 EMA 갱신 → resume 복원 필수
                "roll_ramp": self.roll_ramp,          # rollout 0→1 램프 계수 // resume 시 램프 위치 복원
                "phys_ramp": self.phys_ramp,          # physics sigmoid ramp // resume 시 ramp 위치 복원
                "phys_balanced": self._phys_balanced, # Phase 2 진입(첫 B 호출) 여부
                "sched_state": sched_state,  # OdeScheduler 내부 상태 // scheduler resume 복원용
            },
            tmp,
        )
        os.replace(tmp, path)

    def SaveLatest(self, ckpt_dir, step, metric, sched_state=None):
        # latest는 항상 raw 가중치 + optimizer → 학습 재개용 // resume continuity
        self._SaveCheckpoint(os.path.join(ckpt_dir, "latest.pt"), step, metric, sched_state)

    def MaybeSaveBest(self, ckpt_dir, step, metric, sched_state=None, model_state=None):
        # metric 개선 시에만 best.pt 갱신 // best model 보존 (EMA 가중치)
        if metric < self.best_metric:
            self.best_metric = metric
            self._SaveCheckpoint(os.path.join(ckpt_dir, "best.pt"), step, metric, sched_state, model_state)
            return True
        return False

    def MaybeSaveBestODE(self, ckpt_dir, step, ode_loss, sched_state=None, model_state=None):
        # ODE loss 기준 best 갱신 // physics 수렴 best model 보존
        if ode_loss < self.best_ode_metric:
            self.best_ode_metric = ode_loss
            self._SaveCheckpoint(os.path.join(ckpt_dir, "best_ode.pt"), step, ode_loss, sched_state, model_state)
            return True
        return False

    def MaybeSaveBestExtrap(self, ckpt_dir, step, extrap_loss, sched_state=None, model_state=None):
        # 외삽(마칭 t_data_max 이후) 오차 기준 best 갱신 // 물리 외삽 능력 best model 보존
        if extrap_loss < self.best_extrap_metric:
            self.best_extrap_metric = extrap_loss
            self._SaveCheckpoint(os.path.join(ckpt_dir, "best_extrap.pt"), step, extrap_loss, sched_state, model_state)
            return True
        return False

    def LoadCheckpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self._CheckHardIC(ckpt, path)
        self.load_state_dict(self._AdaptState(ckpt["model_state"]))
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        # EMA shadow 복원 — 구 ckpt(키 없음)면 현재 가중치로 재초기화 // stale/missing shadow 방지
        if self.ema_state is not None:
            saved_ema = ckpt.get("ema_state", None)
            if saved_ema is not None:
                saved_ema = self._AdaptState(saved_ema)
                self.ema_state = {k: v.to(self.device) for k, v in saved_ema.items()}
            else:
                self._ResetEMA()
        self.best_metric = ckpt.get("best_metric", float("inf"))
        self.best_ode_metric = ckpt.get("best_ode_metric", self.best_ode_metric)
        self.best_extrap_metric = ckpt.get("best_extrap_metric", self.best_extrap_metric)
        # merge so newly-added keys survive resume from older checkpoints // 구 ckpt 키 누락 방지
        saved_gs = {k: v for k, v in ckpt.get("grad_scale", {}).items() if k != "roll"}  # 구 ckpt의 roll 슬롯 재활용 잔재 제거
        self.grad_scale = {**self.grad_scale, **saved_gs}
        self.roll_ramp = ckpt.get("roll_ramp", self.roll_ramp)
        self.phys_ramp = ckpt.get("phys_ramp", self.phys_ramp)
        self._phys_balanced = ckpt.get("phys_balanced", self._phys_balanced)
        return ckpt["step"], ckpt["metric"], ckpt.get("sched_state", None)
