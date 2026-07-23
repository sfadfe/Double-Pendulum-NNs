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

        # --- Val/Extrap veto gates // phys가 정체해도 Val/Extrap이 개선 중이면 LR decay 보류 ---
        self.ema_val = None
        self.best_val = None
        self.val_stall = 0
        self.val_seen = False    # val이 1회 이상 측정된 뒤에만 gate 적용
        self.ema_ext = None
        self.best_ext = None
        self.ext_stall = 0
        self.ext_seen = False    # extrap 비활성(finetune)이면 영영 False → gate skip

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
            "ema_val":   self.ema_val,
            "best_val":  self.best_val,
            "val_stall": self.val_stall,
            "val_seen":  self.val_seen,
            "ema_ext":   self.ema_ext,
            "best_ext":  self.best_ext,
            "ext_stall": self.ext_stall,
            "ext_seen":  self.ext_seen,
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
        self.ema_val   = d.get("ema_val",   self.ema_val)
        self.best_val  = d.get("best_val",  self.best_val)
        self.val_stall = d.get("val_stall", self.val_stall)
        self.val_seen  = d.get("val_seen",  self.val_seen)
        self.ema_ext   = d.get("ema_ext",   self.ema_ext)
        self.best_ext  = d.get("best_ext",  self.best_ext)
        self.ext_stall = d.get("ext_stall", self.ext_stall)
        self.ext_seen  = d.get("ext_seen",  self.ext_seen)
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

    def Step(self, avg_phys, epoch, val=None, extrap=None):
        if self.p is None:
            return

        # --- phys EMA + stall: LR decay 주신호 (매 에폭) ---
        if self.ema is None:
            self.ema = avg_phys
            self.best_ema = self.ema
            if self.always_active and self.activate_epoch is None:
                self.activate_epoch = epoch
        else:
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

        # --- Val veto: 개선 중이면 stall 리셋, 정체하면 누적 ---
        if val is not None and val == val:  # not NaN
            if self.ema_val is None:
                self.ema_val = val
                self.best_val = val
            else:
                self.ema_val = self.p["ema_beta_val"] * self.ema_val + (1.0 - self.p["ema_beta_val"]) * val
                if self.ema_val < self.best_val * (1.0 - self.p["rel_tol_val"]):
                    self.best_val = self.ema_val
                    self.val_stall = 0
                else:
                    self.val_stall += 1
            self.val_seen = True

        # --- Extrap veto (sparse) ---
        if extrap is not None and extrap == extrap:  # not NaN
            if self.ema_ext is None:
                self.ema_ext = extrap
                self.best_ext = extrap
            else:
                self.ema_ext = self.p["ema_beta_ext"] * self.ema_ext + (1.0 - self.p["ema_beta_ext"]) * extrap
                if self.ema_ext < self.best_ext * (1.0 - self.p["rel_tol_ext"]):
                    self.best_ext = self.ema_ext
                    self.ext_stall = 0
                else:
                    self.ext_stall += 1
            self.ext_seen = True

        # --- LR decay는 phys·Val·Extrap 셋 다 정체했을 때만 (AND) // 하나라도 개선 중이면 veto ---
        if not self.activated:
            return
        phys_ready = self.stall >= self.patience
        val_ready = (not self.val_seen) or (self.val_stall >= self.p["patience_val"])
        ext_ready = (not self.ext_seen) or (self.ext_stall >= self.p["patience_ext"])
        if phys_ready and val_ready and ext_ready:
            self._Decay(epoch)
            self.stall = 0
            self.val_stall = 0
            self.ext_stall = 0
