from tqdm import tqdm


class OdeScheduler:
    def __init__(self, optimizer, params, ema_init=None):
        self.optimizer = optimizer
        self.p = params
        self.ema = ema_init
        self.best_ema = ema_init if ema_init is not None else float("inf")
        self.stall = 0
        self.activated = False  # activate_threshold 이하로 한번이라도 내려가면 영구 활성화
        self.activate_epoch = None  # 활성화된 에폭 // rollout-aware 게이트 앵커 (resume 안전)
        self.rel_tol = params["rel_tol"]  # LR 감쇄마다 decay되므로 인스턴스 변수로 관리
        self.patience = params["patience"] if params else None  # LR 감쇄마다 줄어듦 // coarse-to-fine

        # Warm restart: 연속 무개선 감쇄 감지 → LR 복원 // 단방향 ratchet의 LR≈0 공회전 제거
        self.bad_decays = 0                    # 직전 감쇄 이후 best_ema 무개선 연속 횟수
        self.last_decay_best_ema = None        # 직전 LR 감쇄 시점의 best_ema (생산성 판정 기준)
        self.restart_lr = params.get("restart_lr0", 0.0) if params else 0.0  # 다음 restart 복원 LR (SGDR 진폭 감쇄)
        self.cooldown_until = -1               # restart 직후 stall 동결 구간 끝 에폭

    def StateDict(self):
        return {
            "ema":      self.ema,
            "best_ema": self.best_ema,
            "stall":    self.stall,
            "activated": self.activated,
            "activate_epoch": self.activate_epoch,
            "rel_tol":  self.rel_tol,
            "patience": self.patience,
            "bad_decays":          self.bad_decays,
            "last_decay_best_ema": self.last_decay_best_ema,
            "restart_lr":          self.restart_lr,
            "cooldown_until":      self.cooldown_until,
        }

    def LoadStateDict(self, d):
        self.ema       = d.get("ema",      self.ema)
        self.best_ema  = d.get("best_ema", self.best_ema)
        self.stall     = d.get("stall",    self.stall)
        self.activated = d.get("activated", self.activated)
        self.activate_epoch = d.get("activate_epoch", self.activate_epoch)
        self.rel_tol   = d.get("rel_tol",  self.rel_tol)
        self.patience  = d.get("patience", self.patience)
        self.bad_decays          = d.get("bad_decays",          self.bad_decays)
        self.last_decay_best_ema = d.get("last_decay_best_ema", self.last_decay_best_ema)
        self.restart_lr          = d.get("restart_lr",          self.restart_lr)
        self.cooldown_until      = d.get("cooldown_until",      self.cooldown_until)

    def _Decay(self, epoch):
        # 기존 ratchet: LR × factor + rel_tol/patience coarse-to-fine // 진짜 개선이 이어지는 동안의 경로
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

    def _Restart(self, epoch):
        # Warm restart (SGDR): 무의미해진 저LR 대신 생산적 대역으로 복원, 진폭은 restart마다 감쇄
        # // best_ema는 유지 → restart가 역사적 best를 넘을 때만 개선으로 인정 (폭주 방지)
        cur_lr = self.optimizer.param_groups[0]["lr"]
        new_lr = max(self.restart_lr, cur_lr)   # 하향 restart 금지 // LR이 아직 복원 대역 위면 유지
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr
        if self.restart_lr >= cur_lr:
            # 실제로 복원이 적용된 경우에만 진폭 감쇄 — 하한은 min_lr 위로 // overall anneal preserved
            self.restart_lr = max(self.restart_lr * self.p["restart_decay"], self.p["min_lr"] * 10.0)
        self.cooldown_until = epoch + self.p["restart_cooldown"]
        tqdm.write(
            f"[ode_sched] epoch {epoch}  WARM RESTART  ode_ema={self.ema:.3e}"
            f"  best_ema={self.best_ema:.3e}  lr {cur_lr:.2e} -> {new_lr:.2e}"
            f"  next_restart_lr={self.restart_lr:.2e}  cooldown {self.p['restart_cooldown']}ep"
        )

    def Step(self, avg_phys, epoch):
        if self.p is None:
            return
        if self.ema is None:
            self.ema = avg_phys
            self.best_ema = self.ema
            return

        self.ema = self.p["ema_beta"] * self.ema + (1.0 - self.p["ema_beta"]) * avg_phys

        if self.ema < self.p["activate_threshold"]:
            if not self.activated:
                self.activate_epoch = epoch   # 첫 활성화 시점 기록 // rollout-aware 게이트 기준
            self.activated = True

        if self.activated:
            if self.ema < self.best_ema * (1.0 - self.rel_tol):
                self.best_ema = self.ema
                self.stall = 0
            elif epoch >= self.cooldown_until:
                # restart 직후 cooldown 동안은 EMA 과도응답을 정체로 오판하지 않음 // freeze stall
                self.stall += 1

        if self.stall >= self.patience:
            # 감쇄 생산성 판정: 직전 감쇄 이후 best_ema가 min_rel_tol 이상 내려왔는가 // unproductive-decay detection
            if self.last_decay_best_ema is not None and self.best_ema >= self.last_decay_best_ema * (
                1.0 - self.p["min_rel_tol"]
            ):
                self.bad_decays += 1
            else:
                self.bad_decays = 0
            self.last_decay_best_ema = self.best_ema

            if self.bad_decays >= self.p["max_bad_decays"] and self.restart_lr > 0.0:
                self._Restart(epoch)
                self.bad_decays = 0
            else:
                self._Decay(epoch)
            self.stall = 0
