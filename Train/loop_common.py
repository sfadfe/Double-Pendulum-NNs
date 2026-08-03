import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import state.Double_pendulum as Dp

CSV_HEADER = [
    "Epoch", "LR",
    "AvgLoss_Data", "AvgLoss_Kin", "AvgLoss_ODE", "AvgLoss_Energy", "AvgLoss_IC",
    "Val", "ValOmega", "RolloutBestVal", "RolloutOmega", "BestVal",
    "BestMetric",
    "Lambda_ODE", "Lambda_Energy",
    "ExtrapBest", "ExtrapOmega",
    "AvgLoss_Roll", "Lambda_Roll",
    "PhysRamp",
    # Per-population eval // flip이 집계 지표를 지배해 단독 스칼라로는 해석 불가 — 분리 기록
    # 반드시 끝에만 추가할 것: ReadLogBests가 fieldnames=CSV_HEADER로 구 로그를 읽음 // append-only
    "Val_Flip", "ValOmega_Flip", "Val_NF", "ValOmega_NF",
    "Rollout_Flip", "RolloutOmega_Flip", "Rollout_NF", "RolloutOmega_NF",
    # goal 1 판정 지표 (2026-08-01) — 평균이 1e-3이어도 seam에서 튀면 목표 미달이므로
    # 집계가 아니라 **per-τ 최대**로 판정한다. RollTauMax는 running min이 아닌 per-epoch 값.
    "RollTauMax", "RollTauArgmax", "RollSeamJump", "RollTauMaxBest",
]


def InitCkptDir(suffix=""):
    # Timestamped model run directory // 학습 결과 폴더 생성
    ts = time.strftime("%Y_%m_%d_%H_%M_%S")
    name = f"{ts}{suffix}"
    ckpt_dir = Path(__file__).parent.parent / "model" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def SetupCaseSplit(trainer, t_params):
    # Held-out val + train pool // 앞 n_val은 검증, 나머지 학습 풀
    max_cases = t_params["max_cases"]
    n_val = min(t_params.get("n_val", 2000), trainer.n_case - max_cases)
    trainer.max_cases = max_cases
    all_idx = torch.arange(trainer.n_case, device=trainer.device)
    trainer.val_cases = all_idx[:n_val]
    trainer.train_pool = all_idx[n_val:]
    trainer.active_cases = trainer.train_pool[:max_cases]


_NAN2 = (float("nan"), float("nan"))


class EvalPrecision:
    """평가 구간만 TF32 off // 2026-08-01

    matmul_precision="high"(TF32)는 matmul당 상대오차 ~5e-4로, 목표 정확도 1e-3과 **같은
    자릿수**다. 학습은 TF32를 유지하되(속도 26% 이득, trap 18) 평가만 순수 FP32로 돌려야
    "1e-3 달성"이 실제인지 반올림 노이즈인지 구분된다. 평가는 드물어 비용은 무시 가능.
    """

    def __enter__(self):
        self._mm = torch.backends.cuda.matmul.allow_tf32
        self._cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return self

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32 = self._mm
        torch.backends.cudnn.allow_tf32 = self._cudnn
        return False


def _SplitByFlip(per_case, is_flip_mask):
    # (n,) per-case error → (all, flip, nonflip) means // 모집단별 평균, 빈 그룹은 nan
    out = {"all": float(per_case.mean()) if per_case.numel() else float("nan")}
    if is_flip_mask is None:
        out["flip"] = out["nf"] = float("nan")
        return out
    fm = is_flip_mask
    out["flip"] = float(per_case[fm].mean()) if int(fm.sum()) else float("nan")
    out["nf"] = float(per_case[~fm].mean()) if int((~fm).sum()) else float("nan")
    return out


def ComputeVal(trainer, device, batch_size):
    # Teacher-forced segment MSE over active_cases // 윈도우 시작 IC를 참값으로 주는 검증
    # flip/nonflip을 같은 forward에서 분리 집계 — 집계 스칼라는 flip이 지배해 단독 해석 불가.
    # feats는 배치 단위로 생성 (구현: 세그먼트 전체를 한 번에 만들지 않음) // peak VRAM ↓
    trainer.eval()
    is_flip = getattr(trainer, "is_flip", None)

    th_sum = {"all": 0.0, "flip": 0.0, "nf": 0.0}
    om_sum = {"all": 0.0, "flip": 0.0, "nf": 0.0}
    cnt = {"all": 0, "flip": 0, "nf": 0}

    with torch.no_grad():
        for t_lo, t_hi in trainer.segments:
            frame = trainer.SegmentFrame(t_lo, t_hi)
            n_total = frame["n_total"]
            n_case = max(len(trainer.active_cases), 1)
            tw = n_total // n_case
            # SegmentFrame flattens case-major (n_case, tw) // 원소별 flip 라벨 복원
            elem_flip = (
                is_flip[trainer.active_cases].repeat_interleave(tw)
                if is_flip is not None and tw > 0
                else None
            )
            for start in range(0, n_total, batch_size):
                end = min(start + batch_size, n_total)
                feats = trainer._BuildFeats(
                    frame["tau"][start:end],
                    frame["params"][start:end],
                    frame["ic_flat"][start:end],
                )
                out = trainer(feats)
                th_e = ((out[:, :2] - frame["theta_true"][start:end]) ** 2).mean(dim=1)
                om_e = ((out[:, 2:] - frame["omega_true"][start:end]) ** 2).mean(dim=1)
                th_sum["all"] += float(th_e.sum())
                om_sum["all"] += float(om_e.sum())
                cnt["all"] += th_e.numel()
                if elem_flip is not None:
                    fm = elem_flip[start:end]
                    n_f = int(fm.sum())
                    if n_f:
                        th_sum["flip"] += float(th_e[fm].sum())
                        om_sum["flip"] += float(om_e[fm].sum())
                        cnt["flip"] += n_f
                    n_n = th_e.numel() - n_f
                    if n_n:
                        th_sum["nf"] += float(th_e[~fm].sum())
                        om_sum["nf"] += float(om_e[~fm].sum())
                        cnt["nf"] += n_n
            del frame

    torch.cuda.empty_cache()
    trainer.train()
    return {
        k: (th_sum[k] / cnt[k], om_sum[k] / cnt[k]) if cnt[k] else _NAN2
        for k in ("all", "flip", "nf")
    }


def SelectRolloutCases(trainer, n_cases, flip_frac=None):
    # Fixed eval subset // val_idx는 빌드 시 셔플돼 있어 앞에서 n개 자르면 계층 비율이 유지됨.
    # flip_frac 지정 시에만 명시적 stratify — flip 표본이 너무 적어 모집단 추정이 흔들릴 때 사용.
    val = trainer.val_cases
    n = min(n_cases, len(val))
    is_flip = getattr(trainer, "is_flip", None)
    if flip_frac is None or is_flip is None:
        return val[:n]
    mask = is_flip[val]
    flip_ids, nf_ids = val[mask], val[~mask]
    n_f = min(len(flip_ids), max(1, int(round(flip_frac * n))))
    n_n = min(len(nf_ids), n - n_f)
    n_f = min(len(flip_ids), n - n_n)
    return torch.cat([flip_ids[:n_f], nf_ids[:n_n]])


def ComputeRollout(trainer, device, n_cases=100, case_idx=None):
    # True time-marching over [0, t_data_max] // 마칭 롤아웃
    # 한 번의 마칭에서 per-case 오차를 내고 flip/nonflip으로 쪼갠다 // 추가 forward 없음
    trainer.eval()
    if case_idx is None:
        case_idx = SelectRolloutCases(trainer, n_cases)
    n_windows = int(round(trainer.dataCfg.t_data_max / trainer.dataCfg.march_dt))

    times, theta_pred, omega_pred = trainer.MarchRollout(case_idx, n_windows)
    T = times.shape[0]

    gi = torch.arange(T, device=device).clamp(max=trainer.n_step - 1)
    theta_true = trainer.data[case_idx][:, gi][:, :, [1, 3]]
    omega_true = trainer.data[case_idx][:, gi][:, :, [2, 4]]

    # per-case mean → 케이스 수가 같으므로 all은 기존 전체 평균과 동일 // exact, not an approximation
    se_th = (theta_pred - theta_true) ** 2
    th_case = se_th.mean(dim=(1, 2))
    om_case = ((omega_pred - omega_true) ** 2).mean(dim=(1, 2))
    is_flip = getattr(trainer, "is_flip", None)
    fm = is_flip[case_idx] if is_flip is not None else None
    th = _SplitByFlip(th_case, fm)
    om = _SplitByFlip(om_case, fm)

    # goal 1: 시간축을 지우기 전에 per-τ 프로파일을 뽑는다 // 케이스축만 평균 → (T,)
    #   집계 RolloutBestVal은 "평균 1e-3인데 seam에서 스파이크"를 구조적으로 못 본다.
    #   판정 지표는 tau_max(전 구간 최악), seam_jump(윈도우 경계 계단)이다.
    tau_prof = se_th.mean(dim=(0, 2))                      # (T,)
    steps_pw = max(1, int(round(trainer.dataCfg.march_dt / trainer.dt)))
    i_max = int(tau_prof.argmax())
    prof = {
        "curve": tau_prof.detach().float().cpu().numpy(),
        "tau_max": float(tau_prof[i_max]),
        "tau_argmax": float(times[i_max]),
        "seam_jump": 0.0,
    }
    if tau_prof.numel() > steps_pw:
        post = tau_prof[steps_pw::steps_pw]                # 각 윈도우의 첫 샘플 (핸드오프 직후)
        pre = tau_prof[steps_pw - 1::steps_pw][: post.numel()]   # 직전 윈도우 마지막 샘플
        prof["seam_jump"] = float((post - pre).max())

    del theta_pred, omega_pred, theta_true, omega_true, th_case, om_case, se_th
    torch.cuda.empty_cache()

    trainer.train()
    out = {k: (th[k], om[k]) for k in ("all", "flip", "nf")}
    out["prof"] = prof
    return out


def _ExtendRK4(state0, params, t_off, dt):
    m1, m2, L1, L2 = (float(x) for x in params)
    out = np.empty((t_off.shape[0], 4), dtype=np.float64)
    s = np.array(state0, dtype=np.float64)
    t = 0.0
    ri = 0
    n_total = int(round(float(t_off[-1]) / dt))
    for _ in range(n_total + 1):
        while ri < t_off.shape[0] and t + 1e-12 >= t_off[ri]:
            out[ri] = s
            ri += 1
        s = Dp.RK4(s, dt, m1, m2, L1, L2, 9.81)
        t += dt
    while ri < t_off.shape[0]:
        out[ri] = s
        ri += 1
    return out


def BuildExtrapGT(trainer, case_idx, t_data_max, t_ext, rk4_dt):
    dev = trainer.device
    n_windows = int(round(t_ext / trainer.dataCfg.march_dt))
    times, _, _ = trainer.MarchRollout(case_idx, n_windows)
    times_np = times.cpu().numpy()
    T = times_np.shape[0]
    ext_mask = times_np > t_data_max + 1e-9

    gi = torch.arange(T, device=dev).clamp(max=trainer.n_step - 1)
    th_true = trainer.data[case_idx][:, gi][:, :, [1, 3]].clone()
    om_true = trainer.data[case_idx][:, gi][:, :, [2, 4]].clone()

    if ext_mask.any():
        last = trainer.data[case_idx][:, trainer.n_step - 1, [1, 2, 3, 4]].double().cpu().numpy()
        params = trainer.params_raw[case_idx].double().cpu().numpy()
        t_off = times_np[ext_mask] - t_data_max
        for c in range(case_idx.shape[0]):
            gt = torch.from_numpy(_ExtendRK4(last[c], params[c], t_off, rk4_dt)).to(dev)
            th_true[c, ext_mask] = gt[:, [0, 2]].float()
            om_true[c, ext_mask] = gt[:, [1, 3]].float()

    ext_mask_t = torch.from_numpy(ext_mask).to(dev)
    return {"n_windows": n_windows, "th_true": th_true, "om_true": om_true, "ext_mask": ext_mask_t}


def ComputeExtrap(trainer, case_idx, gt):
    trainer.eval()
    _, theta_pred, omega_pred = trainer.MarchRollout(case_idx, gt["n_windows"])
    em = gt["ext_mask"]
    th = ((theta_pred[:, em] - gt["th_true"][:, em]) ** 2).mean().item()
    om = ((omega_pred[:, em] - gt["om_true"][:, em]) ** 2).mean().item()
    torch.cuda.empty_cache()
    trainer.train()
    return th, om


def AppendResumeLog(cfg_path, epoch, step, ckpt_name, changes):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"\n# [resume] {ts}  epoch={epoch}  step={step}  ckpt={ckpt_name}"]
    for key, (old_val, new_val) in changes.items():
        lines.append(f"#   {key}: {old_val} -> {new_val}")
    with open(cfg_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def ReadLogBests(log_path, warmup_epochs=0):
    # warmup_epochs 이전(phase1 데이터피팅) 행은 val_best에서 제외 — best.pt 선택이
    # 물리 없는 warmup 모델에 고정되는 것 방지 // phase2 재선택과 일치
    val_best = float("inf")
    rollout_best = float("inf")
    extrap_best = float("inf")
    ode_best = float("inf")
    try:
        rows = []
        with open(log_path, newline="") as f:
            reader = csv.DictReader(f, fieldnames=CSV_HEADER)
            for row in reader:
                if row.get("Epoch") in (None, "", "Epoch"):
                    continue
                try:
                    int(row["Epoch"])
                    rows.append(row)
                except (ValueError, TypeError):
                    pass
        rows.sort(key=lambda r: int(r["Epoch"]))

        for row in rows:
            try:
                # val_best는 θ+ω 합산 지표 기준 (best.pt 선택과 일치) // combined selection metric
                # phase1 행은 skip — warmup 데이터피팅 overfit 값이 phase2를 영원히 이기는 것 방지
                if int(row["Epoch"]) >= warmup_epochs:
                    v = float(row["Val"]) + float(row["ValOmega"])
                    if v == v and v < val_best:
                        val_best = v
            except (ValueError, KeyError, TypeError):
                pass
            try:
                r = float(row["RolloutBestVal"])
                if r == r and r < rollout_best:
                    rollout_best = r
            except (ValueError, KeyError, TypeError):
                pass
            try:
                x = float(row["ExtrapBest"])
                if x == x and x < extrap_best:
                    extrap_best = x
            except (ValueError, KeyError, TypeError):
                pass
            try:
                o = float(row["AvgLoss_ODE"])
                if o == o and o < ode_best:
                    ode_best = o
            except (ValueError, KeyError, TypeError):
                pass
    except FileNotFoundError:
        pass
    return val_best, rollout_best, extrap_best, ode_best


def ReadLogRows(log_path):
    rows = []
    try:
        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Epoch") not in (None, "", "Epoch"):
                    try:
                        int(row["Epoch"])
                        rows.append(dict(row))
                    except (ValueError, KeyError):
                        pass
    except FileNotFoundError:
        pass
    rows.sort(key=lambda r: int(r["Epoch"]))
    return rows


def WriteLog(log_path, log_rows):
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in reversed(log_rows):
            writer.writerow(row)


class PretrainHooks:
    # Default pretrain: LHS collocation + train_pool shuffle // 파인튜닝은 서브클래스로 교체

    def OnLoopStart(self, trainer, device, data_cfg, t_params):
        return None

    def OnEpochStart(self, trainer, epoch, phase1, ode_sched, t_params):
        perm = torch.randperm(len(trainer.train_pool), device=trainer.device)[:trainer.max_cases]
        trainer.active_cases = trainer.train_pool[perm]

    def SampleColloc(self, trainer, t_lo, t_hi, epoch, phase1):
        return trainer.SampleCollocation(t_lo, t_hi)

    def OnStepEnd(self, trainer, step, epoch, phase1, t_lo=None, t_hi=None):
        return False


class FinetuneHooks(PretrainHooks):
    # Replay active_cases + flip-biased fresh collocation // flip 파인튜닝 전용 훅

    def OnLoopStart(self, trainer, device, data_cfg, t_params):
        n = trainer.trainCfg.n_colloc_cases or trainer.max_cases
        bias = trainer.collocCfg.colloc_flip_bias
        trainer.SetupCollocCases(n_colloc=n, flip_bias=bias)
        n_flip = int(trainer.is_flip[trainer.colloc_cases].sum()) if trainer.is_flip is not None else -1
        tqdm.write(
            f"[finetune] colloc_cases={len(trainer.colloc_cases)} flip={n_flip} "
            f"replay_frac={trainer.trainCfg.replay_frac}"
        )

    def OnEpochStart(self, trainer, epoch, phase1, ode_sched, t_params):
        if torch.rand(1).item() < trainer.trainCfg.replay_frac:
            trainer._replay_active = True
            n = min(trainer.max_cases, trainer.replay_n_case)
            perm = torch.randperm(trainer.replay_n_case, device=trainer.device)[:n]
            trainer.active_cases = trainer.replay_idx[perm]
            tqdm.write(f"[replay] epoch {epoch}  pool=nonflip n={n} / {trainer.replay_n_case}")
        else:
            trainer._replay_active = False
            perm = torch.randperm(len(trainer.train_pool), device=trainer.device)[:trainer.max_cases]
            trainer.active_cases = trainer.train_pool[perm]

    def SampleColloc(self, trainer, t_lo, t_hi, epoch, phase1):
        if getattr(trainer, "_replay_active", False):
            return trainer.GetCollocMeta(t_lo, t_hi, case_pool=trainer.active_cases)
        return trainer.GetCollocMeta(t_lo, t_hi)

    # OnStepEnd: PretrainHooks 상속 (콜로케이션은 매 step fresh 재샘플 — refine 훅 불필요)


def RunTrainLoop(
    trainer,
    device,
    train_cfg,
    data_cfg,
    t_params,
    start_epoch,
    step,
    val_best,
    rollout_best,
    extrap_best,
    log_path,
    log_rows,
    ckpt_dir,
    ode_sched_cls,
    ode_s_params=None,
    ode_sched_state=None,
    hooks=None,
    save_rollout_best=False,
):
    if hooks is None:
        hooks = PretrainHooks()

    last_save = time.time()
    metric = float("inf")
    losses = {}
    tau_max_best = float("inf")   # goal 1 running min — 로그에서 재시작(로그 컬럼은 per-epoch 값도 함께 남김)
    ic_term_on = trainer.IcTermOn()
    if not ic_term_on:
        tqdm.write(f"[hard_ic] {trainer.netCfg.hard_ic} — IC 항 비활성 (τ=0 경계조건이 구조적 항등식)")

    DATA_WARMUP_EPOCHS = t_params.get("warmup_epochs", 200)
    GRAD_BALANCE_EVERY = t_params.get("grad_balance_every", 25)
    GRAD_BALANCE_BATCHES = t_params.get("grad_balance_batches", 3)
    USE_GRAD_BALANCE = t_params.get("use_grad_balance", True)
    # drift 트리거 (0 = 비활성 → grad_balance_every 고정 주기, 구 config 하위호환) // trap 18/21
    GRAD_BALANCE_DRIFT = t_params.get("grad_balance_drift", 0.0)
    GRAD_BALANCE_MIN_EVERY = t_params.get("grad_balance_min_every", 0) or GRAD_BALANCE_EVERY
    GRAD_BALANCE_MAX_EVERY = t_params.get("grad_balance_max_every", 0) or GRAD_BALANCE_EVERY

    ROLL_DELAY = t_params.get("roll_activate_delay", 200)
    ROLL_ACTIVATE_EPOCH = t_params.get("roll_activate_epoch", None)
    ROLL_RAMP = t_params.get("roll_ramp_epochs", 150)
    ROLL_CASES = t_params.get("roll_cases", 256)
    ROLL_DEPTH_MAX = t_params.get("roll_depth_max", 3)
    ROLL_POINTS = t_params.get("roll_points", 12)
    ROLL_DEPTH_RAMP = max(t_params.get("roll_depth_ramp_epochs", 100), 1)   # B1: depth 1→max per-step epochs
    KIN_RAMP_EPOCHS = max(t_params.get("kin_ramp_epochs", 40), 1)           # A1: kin 커플러 0→1 램프
    roll_enabled = ROLL_CASES > 0 and trainer.trainCfg.lambda_roll > 0

    # extrap 스케줄러 veto 주기 — rollout_interval과 독립 // 없으면 rollout에 종속(구 동작)
    EXTRAP_SCHED_INTERVAL = t_params.get("extrap_sched_interval", t_params["rollout_interval"])

    IC_MAX_N = train_cfg.ic_max_n or None   # 0 → 전체 (구 config 하위호환) // per-step IC 케이스 상한

    PHYS_RAMP_CENTER = train_cfg.phys_ramp_center
    PHYS_RAMP_WIDTH = max(train_cfg.phys_ramp_width, 1)
    LR_DROP_EPOCH = train_cfg.lr_drop_epoch
    LR_DROP_TO = train_cfg.lr_drop_to

    if ROLL_ACTIVATE_EPOCH is not None and roll_enabled:
        tqdm.write(f"[roll] fixed activate at epoch {int(ROLL_ACTIVATE_EPOCH)} (scheduler gate bypassed)")

    extrap_cases = None
    extrap_gt = None
    n_ext = t_params.get("extrap_cases", 50)
    if n_ext > 0:
        n_ext = min(n_ext, len(trainer.val_cases))
        extrap_cases = trainer.val_cases[:n_ext]
        extrap_gt = BuildExtrapGT(
            trainer, extrap_cases, float(data_cfg.t_data_max),
            t_params.get("extrap_t_ext", 6.0), t_params.get("extrap_rk4_dt", 1e-4),
        )
        tqdm.write(f"[extrap] GT precomputed: {n_ext} cases, t_ext={t_params.get('extrap_t_ext', 6.0)}s")

    # Rollout 평가 케이스는 런 내내 고정 // 에폭 간 차이가 모델 변화만 반영하도록
    # rollout_flip_frac 미지정이면 기존 동작(val_cases 앞에서 n개) 유지 // opt-in stratify
    roll_eval_cases = SelectRolloutCases(
        trainer, t_params["rollout_cases"], t_params.get("rollout_flip_frac")
    )
    if getattr(trainer, "is_flip", None) is not None:
        n_rf = int(trainer.is_flip[roll_eval_cases].sum())
        tqdm.write(
            f"[eval] rollout cases={len(roll_eval_cases)} flip={n_rf} "
            f"nonflip={len(roll_eval_cases) - n_rf} | "
            f"val cases={len(trainer.val_cases)} flip={int(trainer.is_flip[trainer.val_cases].sum())}"
        )

    hooks.OnLoopStart(trainer, device, data_cfg, t_params)

    ode_sched = ode_sched_cls(trainer.optimizer, ode_s_params)
    if ode_sched_state is not None:
        ode_sched.LoadStateDict(ode_sched_state)
    # resume 시 이미 지난 1회성 cap은 소진 처리 — LR 값 자체는 optimizer state가 복원 // 구 ckpt 하위호환
    if start_epoch > DATA_WARMUP_EPOCHS:
        ode_sched.lr.MarkDone("phase2")
    if start_epoch >= LR_DROP_EPOCH and LR_DROP_EPOCH > 0:
        ode_sched.lr.MarkDone("lr_drop")

    # tqdm.write는 stdout에 쓰고 flush하지 않음 — 파일로 리다이렉트하면 stdout이 8KB 블록 버퍼라
    # bar(stderr, 무버퍼)만 흘러나오고 [replay]/[balance]/[best] 로그가 버퍼에 갇힌다.
    # (bar 생성 시 tqdm이 1회 flush하므로 그 이전 메시지만 보였음) // 줄 단위 버퍼로 전환
    for _fp in (sys.stdout, sys.stderr):
        try:
            _fp.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    # drift 트리거 상태 — 체크포인트에 넣지 않는다. resume 첫 사이클만 고정 주기로 떨어졌다가
    # 스스로 복구하므로, 이걸 위해 ckpt state를 늘릴 이유가 없다. // not checkpointed by design
    last_balance_e2 = None
    gnorm_ref = None        # 직전 rebalance 직후의 항별 log 노름비 스냅샷
    snap_pending = False    # rebalance 다음 (non-replay) 에폭 시작에 스냅샷을 뜬다

    pbar = tqdm(range(start_epoch, t_params["max_epochs"]), desc="Training", dynamic_ncols=True)

    try:
        for epoch in pbar:
            epoch_sums = {k: torch.tensor(0.0, device=device)
                          for k in ["data", "kin", "phys", "energy", "ic", "roll"]}
            epoch_steps = 0
            metric_f = float("inf")
            phase1 = epoch < DATA_WARMUP_EPOCHS

            if ROLL_ACTIVATE_EPOCH is not None:
                roll_anchor = int(ROLL_ACTIVATE_EPOCH)
            else:
                # 위상 시계는 loop 소유 — 종전 ode_sched.activate_epoch(=Phase2 첫 에폭)와 동일값
                roll_anchor = DATA_WARMUP_EPOCHS + ROLL_DELAY
            roll_on = roll_enabled and not phase1 and epoch >= roll_anchor
            trainer._roll_active = roll_on          # RebalanceGradScales가 roll 측정 여부 판단
            if roll_on:
                trainer.roll_ramp = min(1.0, (epoch - roll_anchor + 1) / max(ROLL_RAMP, 1))
                trainer._roll_depth = max(1, min(ROLL_DEPTH_MAX, 1 + (epoch - roll_anchor) // ROLL_DEPTH_RAMP))

            hooks.OnEpochStart(trainer, epoch, phase1, ode_sched, t_params)
            if epoch == DATA_WARMUP_EPOCHS:
                ode_sched.DropOnPhase2(epoch)
                # phase1 데이터피팅으로 바닥친 val_best/best.pt 선택 지표 리셋 —
                # 물리 도입 후(phase2) 모델 기준으로 best.pt 재선택되도록
                val_best = float("inf")
                trainer.best_metric = float("inf")
            if LR_DROP_EPOCH > 0 and epoch >= LR_DROP_EPOCH:
                ode_sched.lr.CapOnce("lr_drop", LR_DROP_TO, epoch)   # 고정 에폭 1회 캡 // OdeScheduler 판정과 무관
            # replay 에폭 여부를 에폭 시작 시점에 확정 // val/rollout이 _replay_active를 리셋하기 전 캡처
            replay_epoch = getattr(trainer, "_replay_active", False)

            e2 = 0
            if not phase1:
                e2 = epoch - DATA_WARMUP_EPOCHS
                warm = max(trainer.collocCfg.ic_sigma_warmup, 1)
                trainer._colloc_ic_sigma = trainer.collocCfg.ic_sigma * min(1.0, e2 / warm)
                if e2 >= 0:
                    x = (e2 - PHYS_RAMP_CENTER) / PHYS_RAMP_WIDTH
                    trainer.phys_ramp = 1.0 / (1.0 + math.exp(-x))
                else:
                    trainer.phys_ramp = 0.0
            else:
                trainer.phys_ramp = 0.0

            # A1: kin(헤드 커플러)는 epoch 0부터 0→1 램프, phys_ramp와 독립 상시 활성
            trainer.kin_ramp = min(1.0, (epoch + 1) / KIN_RAMP_EPOCHS)
            # A2: kin은 항상 대칭 — 두 헤드가 함께 커플링 학습.
            #   detach=True로 두면 dθ/dτ에 걸린 유일한 gradient 제약이 사라짐(EOM은 dω/dτ만,
            #   energy는 E만 제약) → θ 헤드 미분이 발산. 측정: corr(ω,dθ/dτ) 0.995 → 0.120,
            #   RMS(dθ/dτ) 16.4 vs RMS(ω) 1.88 (2026_07_24 런). // 항상 False
            trainer._kin_detach = False

            # 직전 rebalance의 효과가 _term_gnorm EMA(0.9/0.1)에 완전히 반영되는 시점은
            # 에폭 1회분 스텝 뒤다(0.9^100 ≈ 2.6e-5). 그래서 스냅샷은 rebalance한 에폭이 아니라
            # **그 다음** non-replay 에폭 시작에 뜬다. replay 에폭은 분포가 달라 노름 계열이
            # 이봉이므로(trap 6) 스냅샷·비교 양쪽에서 제외한다.
            drift = None
            if GRAD_BALANCE_DRIFT > 0 and not phase1 and not replay_epoch:
                if snap_pending:
                    gnorm_ref = trainer.GnormRatios()
                    snap_pending = gnorm_ref is None       # 아직 비었으면 다음 에폭에 재시도
                elif gnorm_ref is not None:
                    drift = trainer.GnormDrift(gnorm_ref)

            do_balance = False
            if USE_GRAD_BALANCE and not phase1 and not replay_epoch and trainer.phys_ramp >= 0.5:
                since = GRAD_BALANCE_MAX_EVERY if last_balance_e2 is None else e2 - last_balance_e2
                if GRAD_BALANCE_DRIFT > 0 and gnorm_ref is not None:
                    # min_every = 폭주 방지 하한, max_every = 후반 희소화 상한 // floor/ceiling
                    do_balance = (
                        since >= GRAD_BALANCE_MAX_EVERY
                        or (since >= GRAD_BALANCE_MIN_EVERY
                            and drift is not None and drift > GRAD_BALANCE_DRIFT)
                    )
                else:
                    # drift 비활성이거나 아직 스냅샷 전(부트스트랩) → 구 동작 그대로
                    do_balance = (e2 % GRAD_BALANCE_EVERY == 0)

            if do_balance:
                last_balance_e2 = e2
                snap_pending = GRAD_BALANCE_DRIFT > 0
                g = trainer.RebalanceGradScales(GRAD_BALANCE_BATCHES)
                gs = trainer.grad_scale
                tqdm.write(
                    f"[balance] epoch {epoch}  ‖g‖ data={g['data']:.2e} kin={g['kin']:.2e} "
                    f"phys={g['phys']:.2e} energy={g['energy']:.2e}"
                    f"{' ic=%.2e' % g['ic'] if 'ic' in g else ''}"
                    f"{' roll=%.2e' % g['roll'] if 'roll' in g else ''} | "
                    f"scale kin={gs['kin']:.2e} phys={gs['phys']:.2e} energy={gs['energy']:.2e} "
                    f"{'ic=%.2e ' % gs['ic'] if 'ic' in g else ''}roll={gs['roll']:.2e}"
                    + (f" | since={since} drift={'--' if drift is None else '%.3f' % drift}"
                       if GRAD_BALANCE_DRIFT > 0 else ""))
                # 항별 clip 직전 노름 EMA — 예산(term_grad_clip) 대비 포화 여부. 예산에 붙은 항은
                # 그만큼 눌리고 있다는 뜻이고, 안 붙은 항은 원래 크기로 통과 중이다. // saturation probe
                tg = trainer._term_gnorm
                if tg:
                    budget = trainer._TermClipBudget()
                    tqdm.write(
                        f"[termclip] epoch {epoch}  budget={budget:.2f}  "
                        + " ".join(f"{k}={float(v):.2e}" for k, v in tg.items()))

            for t_lo, t_hi in trainer.segments:
                frame = trainer.SegmentFrame(t_lo, t_hi)
                bs = min(data_cfg.batch_size, frame["n_total"])

                for _ in range(t_params["steps_per_segment"]):
                    batch = trainer.SegmentBatch(frame, bs)
                    # hard_ic면 IC 잔차가 항등적으로 0 — 샘플링 자체를 건너뛴다 // 스텝 비용 회수
                    ic_parts = trainer.ICSamplesRaw(max_n=IC_MAX_N) if ic_term_on else []
                    # colloc은 Phase 1에서도 샘플 — kin 커플러를 데이터피팅과 함께 조기 학습 // A1
                    # (Phase 1은 ic_sigma=0이라 온-매니폴드, Phase 2에서 오프-매니폴드로 확장)
                    colloc_meta = hooks.SampleColloc(trainer, t_lo, t_hi, epoch, phase1)

                    roll_loss = None
                    if roll_on:
                        sel = trainer.active_cases[
                            torch.randint(0, len(trainer.active_cases), (ROLL_CASES,), device=device)
                        ]
                        # B1: depth 커리큘럼 — 얕은 핸드오프부터 점진 심화 (초반 깊은 rollout 노이즈 억제)
                        # 상한은 에폭 시작 시 trainer._roll_depth로 계산 — 밸런싱 측정과 같은 값 공유
                        # roll_draws>1이면 RolloutLoss가 내부에서 draw별로 다시 추첨한다
                        roll_loss = trainer.RolloutLoss(sel, trainer.RollDepth(), ROLL_POINTS)

                    if phase1:
                        metric_t, losses = trainer.BackwardDataIC(batch, ic_parts, colloc_meta)
                    else:
                        metric_t, losses = trainer.BackwardAll(batch, colloc_meta, ic_parts, roll_loss=roll_loss)

                    # 스텝당 host sync는 여기 1회뿐 — NaN이면 optimizer.step을 건너뛰어야 하므로
                    # 이 판정만은 CPU 값이 필요하다. 항·청크별 float()는 Trainstep에서 제거됨.
                    metric_f = float(metric_t)
                    if not math.isfinite(metric_f):
                        trainer.optimizer.zero_grad()
                        nan_keys = [k for k, v in losses.items() if not math.isfinite(float(v))]
                        print(f"[NaN] skipping step — affected: {nan_keys}")
                        step += 1
                        continue

                    torch.nn.utils.clip_grad_norm_(trainer.parameters(), train_cfg.grad_clip)
                    trainer.optimizer.step()
                    trainer.UpdateEMA()   # B3: Polyak shadow ← 매 step raw 가중치

                    step += 1
                    trainer._global_step = step

                    for k in losses:
                        epoch_sums[k] = epoch_sums[k] + losses[k]
                    epoch_steps += 1

                    hooks.OnStepEnd(trainer, step, epoch, phase1, t_lo, t_hi)

            lam = trainer.LambdaAt()
            lr = trainer.optimizer.param_groups[0]["lr"]
            metric = metric_f
            avg = {k: float(epoch_sums[k]) / epoch_steps if epoch_steps > 0 else float("nan")
                   for k in epoch_sums}
            if phase1:
                # kin은 Phase 1부터 활성(A1)이라 실측값 유지 — phys/energy만 비활성
                avg["phys"] = avg["energy"] = float("nan")
            if not roll_on:
                avg["roll"] = float("nan")

            # B3: 평가·best.pt는 EMA 가중치로 — raw는 백업 후 이 구간 동안만 스왑
            do_val = epoch % t_params["val_interval"] == 0
            do_ext = extrap_gt is not None and epoch % EXTRAP_SCHED_INTERVAL == 0
            do_roll = epoch % t_params["rollout_interval"] == 0
            ema_applied = trainer.ApplyEMA() if (do_val or do_ext or do_roll) else False

            val_loss = float("nan")
            val_omega_loss = float("nan")
            val_metric = float("nan")
            val_res = None
            if do_val:
                trainer._replay_active = False
                trainer.active_cases = trainer.val_cases
                with EvalPrecision():   # 평가만 순수 FP32 // TF32 반올림이 목표 1e-3과 동급이라
                    val_res = ComputeVal(trainer, device, data_cfg.batch_size)
                val_loss, val_omega_loss = val_res["all"]
                # 모델 선택 지표 = θ+ω (핸드오프를 지배하는 ω 오차 포함) // best.pt selection
                val_metric = val_loss + val_omega_loss
                if val_metric < val_best:
                    val_best = val_metric
                if not math.isnan(val_res["flip"][0]):
                    tqdm.write(
                        f"[val]     epoch {epoch}  flip θ={val_res['flip'][0]:.3e}"
                        f" ω={val_res['flip'][1]:.3e} | nonflip θ={val_res['nf'][0]:.3e}"
                        f" ω={val_res['nf'][1]:.3e}"
                    )

            # Extrap: 스케줄러 veto용 독립 주기 (rollout_interval과 분리) // decoupled from rollout
            extrap_loss = float("nan")
            extrap_omega_loss = float("nan")
            if do_ext:
                trainer._replay_active = False
                with EvalPrecision():
                    extrap_loss, extrap_omega_loss = ComputeExtrap(trainer, extrap_cases, extrap_gt)
                if extrap_loss < extrap_best:
                    extrap_best = extrap_loss
                tqdm.write(
                    f"[extrap]  epoch {epoch}  theta={extrap_loss:.3e}"
                    f"  omega={extrap_omega_loss:.3e}  best={extrap_best:.3e}"
                )

            # replay 에폭(쉬운 nonflip)은 LR 스케줄러에서 격리 // 분포 스위칭이 LR 판정을 오염시키지 않도록
            sched_on = not phase1 and epoch_steps > 0 and not replay_epoch

            rollout_loss = float("nan")
            rollout_omega_loss = float("nan")
            roll_res = None
            tau_max = tau_argmax = seam_jump = float("nan")
            if do_roll:
                trainer._replay_active = False
                with EvalPrecision():
                    roll_res = ComputeRollout(trainer, device, case_idx=roll_eval_cases)
                rollout_loss, rollout_omega_loss = roll_res["all"]
                prof = roll_res["prof"]
                tau_max, tau_argmax = prof["tau_max"], prof["tau_argmax"]
                seam_jump = prof["seam_jump"]
                if tau_max < tau_max_best:
                    tau_max_best = tau_max
                    # best.pt와 같은 정책: 개선됐을 때만, 파일 하나만 덮어쓴다 // 매 rollout_interval마다
                    # epoch별로 새 파일을 쌓으면 풀런에서 수백 개까지 무한 누적된다
                    np.save(ckpt_dir / "roll_tau_best.npy", prof["curve"])
                # goal 1 판정 줄 — RMSE로 환산해 목표(1e-3)와 직접 비교 가능하게 // per-τ 최악값
                tqdm.write(
                    f"[goal1]   epoch {epoch}  tauMaxRMSE={math.sqrt(tau_max):.3e}"
                    f" @t={tau_argmax:.2f}s  seamJump={seam_jump:.3e}"
                    f"  meanRMSE={math.sqrt(rollout_loss):.3e}"
                )
                if rollout_loss < rollout_best:
                    rollout_best = rollout_loss
                    if save_rollout_best and not math.isnan(rollout_loss):
                        if trainer.MaybeSaveBest(ckpt_dir, step, rollout_loss, ode_sched.StateDict(),
                                                 model_state=trainer.ema_state):
                            tqdm.write(f"[best_rollout] step {step}  rollout={rollout_loss:.3e}")
                tqdm.write(
                    f"[rollout] epoch {epoch}  theta={rollout_loss:.3e}"
                    f"  omega={rollout_omega_loss:.3e}  best={rollout_best:.3e}"
                )
                if not math.isnan(roll_res["flip"][0]):
                    tqdm.write(
                        f"[rollout] epoch {epoch}  flip θ={roll_res['flip'][0]:.3e}"
                        f" ω={roll_res['flip'][1]:.3e} | nonflip θ={roll_res['nf'][0]:.3e}"
                        f" ω={roll_res['nf'][1]:.3e}"
                    )

            # OdeScheduler: patience 타이머 + Val/Extrap/Rollout veto // 하나라도 개선 중이면 LR decay 보류
            # rollout 계산 뒤로 옮김 (2026-07-28) — 이 프로젝트 1순위 지표를 veto에 넣으려면 값이 먼저 있어야 함
            if sched_on:
                sched_val = val_metric if do_val else None
                sched_ext = extrap_loss if not math.isnan(extrap_loss) else None
                sched_roll = rollout_loss if not math.isnan(rollout_loss) else None
                ode_sched.Step(epoch, val=sched_val, extrap=sched_ext,
                               rollout=sched_roll)

            # EMA 구간 종료 — 학습·latest.pt는 raw 가중치로 복원 // best.pt는 아래에서 ema_state 명시 저장
            if ema_applied:
                trainer.RestoreRaw()

            log_rows.append({
                "Epoch":          epoch,
                "LR":             lr,
                "AvgLoss_Data":   avg["data"],
                "AvgLoss_Kin":    avg["kin"],
                "AvgLoss_ODE":    avg["phys"],
                "AvgLoss_Energy": avg["energy"],
                "AvgLoss_IC":     avg["ic"],
                "Val":            val_loss,
                "ValOmega":       val_omega_loss,
                "RolloutBestVal": rollout_best,
                "RolloutOmega":   rollout_omega_loss,
                "BestVal":        val_best,
                "BestMetric":     trainer.best_metric,
                "Lambda_ODE":     lam["phys"],
                "Lambda_Energy":  lam["energy"],
                "ExtrapBest":     extrap_best,
                "ExtrapOmega":    extrap_omega_loss,
                "AvgLoss_Roll":   avg["roll"],
                "Lambda_Roll":    lam["roll"] * trainer.roll_ramp if roll_on else float("nan"),
                "PhysRamp":       trainer.phys_ramp if not phase1 else float("nan"),
                # 모집단 분리 지표 — 집계 Val/Rollout은 flip이 지배하므로 개선 방향을 못 가림
                # pretrain(is_flip=None)은 전부 nan // per-population, nan when no flip labels
                "Val_Flip":          val_res["flip"][0] if val_res else float("nan"),
                "ValOmega_Flip":     val_res["flip"][1] if val_res else float("nan"),
                "Val_NF":            val_res["nf"][0] if val_res else float("nan"),
                "ValOmega_NF":       val_res["nf"][1] if val_res else float("nan"),
                "Rollout_Flip":      roll_res["flip"][0] if roll_res else float("nan"),
                "RolloutOmega_Flip": roll_res["flip"][1] if roll_res else float("nan"),
                "Rollout_NF":        roll_res["nf"][0] if roll_res else float("nan"),
                "RolloutOmega_NF":   roll_res["nf"][1] if roll_res else float("nan"),
                # goal 1 — 집계가 아니라 per-τ 최악값이 판정 기준 (MSE 단위, RMSE는 sqrt)
                "RollTauMax":        tau_max,
                "RollTauArgmax":     tau_argmax,
                "RollSeamJump":      seam_jump,
                "RollTauMaxBest":    tau_max_best,
            })
            WriteLog(log_path, log_rows)

            pbar.set_postfix({
                "step":    step,
                "data":    f"{avg['data']:.3e}"  if epoch_steps else "-",
                "kin":     f"{avg['kin']:.3e}"   if epoch_steps else "-",
                "phys":    f"{avg['phys']:.3e}"  if epoch_steps else "-",
                "val":     f"{val_best:.3e}"     if val_best < float("inf") else "-",
                "rollout": f"{rollout_best:.3e}" if rollout_best < float("inf") else "-",
            })

            if time.time() - last_save >= t_params["save_interval_sec"]:
                trainer.SaveLatest(ckpt_dir, step, metric, ode_sched.StateDict())
                last_save = time.time()
                tqdm.write(f"[save] latest @ step {step}")

            if do_val:
                if not math.isnan(val_metric):
                    if trainer.MaybeSaveBest(ckpt_dir, step, val_metric, ode_sched.StateDict(),
                                             model_state=trainer.ema_state):
                        tqdm.write(f"[best] step {step}  val(θ+ω)={val_metric:.3e}")

            if not phase1 and epoch_steps > 0:
                if trainer.MaybeSaveBestODE(ckpt_dir, step, avg["phys"], ode_sched.StateDict(),
                                            model_state=trainer.ema_state):
                    tqdm.write(f"[best_ode] step {step}  ode={avg['phys']:.3e}")

            if not math.isnan(extrap_loss):
                if trainer.MaybeSaveBestExtrap(ckpt_dir, step, extrap_loss, ode_sched.StateDict(),
                                               model_state=trainer.ema_state):
                    tqdm.write(f"[best_extrap] step {step}  extrap={extrap_loss:.3e}")

    except KeyboardInterrupt:
        # latest만 저장 — 학습 total loss(metric)와 val 지표는 단위가 달라 best.pt를 덮으면 안 됨 // unit-mismatch guard
        trainer.SaveLatest(ckpt_dir, step, metric, ode_sched.StateDict())
        tqdm.write(f"저장 완료 (latest): {ckpt_dir}")
