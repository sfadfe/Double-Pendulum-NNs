from tqdm import tqdm


class OdeScheduler:
    def __init__(self, optimizer, params, ema_init=None):
        self.optimizer = optimizer
        self.p = params
        self.ema = ema_init
        self.best_ema = ema_init if ema_init is not None else float("inf")
        self.stall = 0
        self.always_active = bool(params.get("always_active", True)) if params else True
        self.activated = self.always_active  # always_active면 Phase 2부터 즉시 stall/LR 감쇄
        self.activate_epoch = None  # 활성화된 에폭 // rollout-aware 게이트 앵커 (resume 안전)
        self.rel_tol = params["rel_tol"]  # LR 감쇄마다 decay되므로 인스턴스 변수로 관리
        self.patience = params["patience"] if params else None  # LR 감쇄마다 줄어듦 // coarse-to-fine
        self.phase2_dropped = False            # warmup→Phase2 1회 LR 드롭 완료 여부

    def StateDict(self):
        return {
            "ema":      self.ema,
            "best_ema": self.best_ema,
            "stall":    self.stall,
            "activated": self.activated,
            "activate_epoch": self.activate_epoch,
            "rel_tol":  self.rel_tol,
            "patience": self.patience,
            "phase2_dropped":      self.phase2_dropped,
        }

    def LoadStateDict(self, d):
        self.ema       = d.get("ema",      self.ema)
        self.best_ema  = d.get("best_ema", self.best_ema)
        self.stall     = d.get("stall",    self.stall)
        self.activated = d.get("activated", self.activated)
        self.activate_epoch = d.get("activate_epoch", self.activate_epoch)
        self.rel_tol   = d.get("rel_tol",  self.rel_tol)
        self.patience  = d.get("patience", self.patience)
        self.phase2_dropped      = d.get("phase2_dropped",      self.phase2_dropped)
        if self.always_active:
            self.activated = True

    def DropOnPhase2(self, epoch):
        # Phase 2 진입 시 1회 LR 하향 — warmup 고LR 유지 후 physics 도입 쇼크 완화
        phase2_lr = self.p.get("phase2_lr", 0.0) if self.p else 0.0
        if phase2_lr <= 0.0 or self.phase2_dropped:
            return
        cur_lr = self.optimizer.param_groups[0]["lr"]
        new_lr = min(cur_lr, phase2_lr)
        if new_lr < cur_lr:
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr
            tqdm.write(
                f"[ode_sched] Phase 2 LR drop  epoch {epoch}"
                f"  lr {cur_lr:.2e} -> {new_lr:.2e}"
            )
        self.phase2_dropped = True

    def _Decay(self, epoch):
        # ratchet: LR × factor + rel_tol/patience coarse-to-fine // 진짜 개선이 이어지는 동안의 경로
        cur_lr = self.optimizer.param_groups[0]["lr"]
        new_lr = max(cur_lr * self.p["factor"], self.p["min_lr"])
        if new_lr < cur_lr:
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr
            # rel_tol 하한: 개선 판정 기준이 0으로 수렴하는 것 방지 // floor prevents meaningless tolerance
            self.rel_tol = max(self.rel_tol * self.p["rel_tol_decay"], self.p["min_rel_tol"])
            self.patience = max(self.patience * self.p["patience_decay"], self.p["min_patience"])
            tqdm.write(
                f"[ode_sched] epoch {epoch}  ode_ema={self.ema:.3e}"
                f"  lr {cur_lr:.2e} -> {new_lr:.2e}"
                f"  rel_tol -> {self.rel_tol:.4f}  patience -> {self.patience:.1f}"
            )

    def Step(self, avg_phys, epoch):
        if self.p is None:
            return
        if self.ema is None:
            self.ema = avg_phys
            self.best_ema = self.ema
            if self.always_active and self.activate_epoch is None:
                self.activate_epoch = epoch
            return

        self.ema = self.p["ema_beta"] * self.ema + (1.0 - self.p["ema_beta"]) * avg_phys

        if self.always_active:
            if self.activate_epoch is None:
                self.activate_epoch = epoch
        elif self.ema < self.p["activate_threshold"]:
            if not self.activated:
                self.activate_epoch = epoch   # 첫 활성화 시점 기록 // rollout-aware 게이트 기준
            self.activated = True

        if self.activated:
            if self.ema < self.best_ema * (1.0 - self.rel_tol):
                self.best_ema = self.ema
                self.stall = 0
            else:
                self.stall += 1

        if self.stall >= self.patience:
            self._Decay(epoch)
            self.stall = 0
