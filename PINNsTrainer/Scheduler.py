from tqdm import tqdm


class LrPolicy:
    """Sole writer of optimizer LR // LR을 쓰는 유일한 창구 — 모든 변경이 태그와 함께 로그 1줄

    종전엔 optimizer init / DropOnPhase2 / ApplyLrDrop / _Decay 네 경로가 서로 모른 채
    LR을 썼다. 이제 1회성 캡(CapOnce)과 곱셈 감쇄(Decay) 둘만 있고, 어떤 경로가 언제
    바꿨는지 [lr] 로그로 한 곳에서 추적된다.
    """

    def __init__(self, optimizer, min_lr=0.0):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.done = set()   # consumed one-shot cap tags // resume 시 재발동·중복 로그 방지

    def Get(self):
        return self.optimizer.param_groups[0]["lr"]

    def MarkDone(self, tag):
        # Consume a one-shot cap without applying it // 구 체크포인트 resume 하위호환
        self.done.add(tag)

    def CapOnce(self, tag, cap, epoch):
        # One-shot LR cap, cap<=0 disables // phase2 드롭과 lr_drop_epoch 겸용
        if cap <= 0.0 or tag in self.done:
            return
        self.done.add(tag)
        self._Set(min(self.Get(), cap), epoch, tag)

    def Decay(self, factor, epoch, tag):
        # Multiplicative decay floored at min_lr // 바닥에 닿으면 no-op(False)
        return self._Set(max(self.Get() * factor, self.min_lr), epoch, tag)

    def _Set(self, new_lr, epoch, tag):
        cur = self.Get()
        if new_lr >= cur:
            return False
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr
        tqdm.write(f"[lr] epoch {epoch}  {cur:.2e} -> {new_lr:.2e}  ({tag})")
        return True

    def StateDict(self):
        return {"done": sorted(self.done)}

    def LoadStateDict(self, d):
        self.done = set(d.get("done", []))


class StallTracker:
    """Stall detector for one metric // 지표 하나의 정체 판정: EMA·best·stall 한 벌

    EMA가 best 대비 rel_tol 이상 개선되면 stall 리셋, 아니면 경과 에폭만큼 누적.
    rel_tol은 **에폭당** 개선율 하한, patience는 **에폭** 단위다. 측정 주기가 달라도
    (val 5에폭, rollout 75에폭) 경과 에폭으로 복리 환산해 같은 잣대를 쓴다 — 종전엔
    측정당 절대값이라 주기가 긴 게이트일수록 판정이 관대해졌다(roll이 val의 3배).
    rel_tol은 "그 지표가 아직 살아있는가"의 하한이지 목표 개선율이 아니다. 작게 잡을 것.
    """

    def __init__(self, beta, rel_tol, patience):
        self.beta = beta
        self.rel_tol = rel_tol      # per-epoch improvement floor // 에폭당
        self.patience = patience    # stall epochs before Stalled // 에폭 단위
        self.ema = None
        self.best = float("inf")
        self.stall = 0              # stalled epochs, not measurements // 정체 에폭 수
        self.last_epoch = None
        self.seen = False

    def Seed(self, value):
        # EMA warm start on resume // 구 sched_state 마이그레이션용
        self.ema = value
        self.best = value
        self.seen = True

    def Update(self, value, epoch):
        if self.ema is None:
            self.Seed(value)
            self.last_epoch = epoch
            return
        elapsed = 1 if self.last_epoch is None else max(epoch - self.last_epoch, 1)
        self.last_epoch = epoch
        self.ema = self.beta * self.ema + (1.0 - self.beta) * value
        need = 1.0 - (1.0 - self.rel_tol) ** elapsed   # 에폭당 하한을 경과분만큼 복리 환산
        if self.ema < self.best * (1.0 - need):
            self.best = self.ema
            self.stall = 0
        else:
            self.stall += elapsed

    def Stalled(self):
        return self.stall >= self.patience

    def StateDict(self):
        return {"ema": self.ema, "best": self.best, "stall": self.stall,
                "last_epoch": self.last_epoch, "seen": self.seen}

    def LoadStateDict(self, d):
        self.ema = d.get("ema", self.ema)
        self.best = d.get("best", self.best)
        self.stall = d.get("stall", self.stall)
        self.last_epoch = d.get("last_epoch", self.last_epoch)
        self.seen = d.get("seen", self.seen)


class OdeScheduler:
    """decay_every(=patience) 에폭 경과 시 LR decay 시도. Val/Extrap/Rollout이 실제
    개선 중이면 보류(veto), 보류 누적이 max_veto_epochs를 넘으면 강제 decay.

    2026-07-31 축약: phys StallTracker(ema_beta/rel_tol) 제거. rel_tol 0.2~0.3은
    **에폭당** 20~30% 개선 요구라 phase 2에서 도달 불가 → phys는 상시 stalled였다
    (실측: pretrain_termclip.log의 veto 로그마다 stall phys=patience 포화, 최대 1430).
    즉 실효 동작이 "patience 에폭 경과 후 decay 시도"였으므로 그 타이머만 남긴다.
    판정만 담당 — LR 쓰기는 전부 self.lr(LrPolicy), 게이트 정체 판정은 StallTracker 위임.
    """

    GATE_KEYS = {
        "val":  ("ema_beta_val",  "rel_tol_val",  "patience_val"),
        "ext":  ("ema_beta_ext",  "rel_tol_ext",  "patience_ext"),
        "roll": ("ema_beta_roll", "rel_tol_roll", "patience_roll"),
    }

    def __init__(self, optimizer, params):
        p = params or {}
        self.p = p
        self.enabled = bool(p)
        self.lr = LrPolicy(optimizer, p.get("min_lr", 0.0))
        self.decay_every = p.get("patience", 80)   # decay 시도 간격(에폭) // 구 phys patience와 실효 동일
        # 키가 없는 게이트는 patience 0 → 측정돼도 즉시 Stalled → veto 불참 // finetune의 ext
        self.gates = {
            name: StallTracker(p.get(bk, 0.9), p.get(tk, 4e-4), p.get(pk, 0))
            for name, (bk, tk, pk) in self.GATE_KEYS.items()
        }
        self.since_decay = 0     # 마지막 decay 이후 경과 에폭 // decay_every 타이머
        self.last_epoch = None
        self.veto_run = 0        # 타이머 만료 후 연속 veto된 에폭 // max_veto_epochs 예산
        self._veto_said = False  # veto 안내 스트릭당 1회 출력 // 직렬화 안 함

    def DropOnPhase2(self, epoch):
        # One-shot LR cap entering Phase 2 // warmup 고LR 유지 후 physics 도입 쇼크 완화
        self.lr.CapOnce("phase2", self.p.get("phase2_lr", 0.0), epoch)

    def Step(self, epoch, val=None, extrap=None, rollout=None):
        if not self.enabled:
            return
        elapsed = 1 if self.last_epoch is None else max(epoch - self.last_epoch, 1)
        self.last_epoch = epoch
        self.since_decay += elapsed
        for name, v in (("val", val), ("ext", extrap), ("roll", rollout)):
            if v is not None and v == v:   # not NaN // sparse 측정은 그냥 건너뜀
                self.gates[name].Update(v, epoch)

        if self.since_decay < self.decay_every:
            self.veto_run = 0
            self._veto_said = False
            return

        # Veto: 측정된 게이트 중 아직 개선 중인 것이 있으면 decay 보류 // 미측정 게이트는 불참
        blockers = [n for n, g in self.gates.items() if g.seen and not g.Stalled()]
        budget = self.p.get("max_veto_epochs", 0)
        if blockers:
            self.veto_run += elapsed
        forced = budget > 0 and self.veto_run >= budget

        if blockers and not forced:
            if not self._veto_said:
                stalls = " ".join(f"{n}={g.stall}" for n, g in self.gates.items() if g.seen)
                tqdm.write(
                    f"[ode_sched] epoch {epoch}  decay VETOED by {'+'.join(blockers)}"
                    f"  (since_decay={self.since_decay} {stalls}"
                    f"  veto_run={self.veto_run}/{budget if budget > 0 else '∞'})"
                )
                self._veto_said = True
            return

        if forced and blockers:
            tqdm.write(
                f"[ode_sched] epoch {epoch}  veto budget exhausted"
                f" ({self.veto_run} epochs by {'+'.join(blockers)}) -> forced decay"
            )
        self.lr.Decay(self.p["factor"], epoch, "ode_sched")
        self.since_decay = 0
        for g in self.gates.values():
            g.stall = 0
        self.veto_run = 0
        self._veto_said = False

    def StateDict(self):
        return {
            "format": 3,
            "since_decay": self.since_decay,
            "last_epoch": self.last_epoch,
            "gates": {n: g.StateDict() for n, g in self.gates.items()},
            "veto_run": self.veto_run,
            "lr": self.lr.StateDict(),
        }

    def LoadStateDict(self, d):
        fmt = d.get("format", 1)
        if fmt < 3:
            self._LoadLegacy(d, fmt)
            return
        self.since_decay = d.get("since_decay", 0)
        self.last_epoch = d.get("last_epoch", None)
        for n, g in self.gates.items():
            if n in d.get("gates", {}):
                g.LoadStateDict(d["gates"][n])
        self.veto_run = d.get("veto_run", 0)
        self.lr.LoadStateDict(d.get("lr", {}))

    def _LoadLegacy(self, d, fmt):
        # 구 sched_state 마이그레이션 — 기존 latest.pt/best*.pt resume용.
        # phys 트래커의 stall(정체 에폭 수)이 곧 "마지막 decay 이후 경과"였으므로
        # since_decay로 그대로 옮긴다. ema/best는 버림(트래커 제거).
        if fmt >= 2:   # format 2: {phys, gates, veto_run, lr}
            phys = d.get("phys", {})
            self.since_decay = phys.get("stall", 0)
            self.last_epoch = phys.get("last_epoch", None)
            for n, g in self.gates.items():
                if n in d.get("gates", {}):
                    g.LoadStateDict(d["gates"][n])
            self.veto_run = d.get("veto_run", 0)
            self.lr.LoadStateDict(d.get("lr", {}))
            return
        # format 1: 평면 21필드
        self.since_decay = d.get("stall", 0)
        for name, g in self.gates.items():
            if d.get(f"{name}_seen") and d.get(f"ema_{name}") is not None:
                g.Seed(d[f"ema_{name}"])
                g.best = d.get(f"best_{name}", g.ema)
                g.stall = d.get(f"{name}_stall", 0)
        self.veto_run = d.get("veto_run", 0)
        if d.get("phase2_dropped"):
            self.lr.MarkDone("phase2")
