import csv
import math
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


def ComputeVal(trainer, device, batch_size):
    trainer.eval()
    theta_loss = 0.0
    omega_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for t_lo, t_hi in trainer.segments:
            feats, theta_t, omega_t, _ = trainer.SegmentSamples(t_lo, t_hi)
            n_total = feats.shape[0]
            for start in range(0, n_total, batch_size):
                end = min(start + batch_size, n_total)
                out = trainer(feats[start:end])
                theta_loss += torch.mean((out[:, :2] - theta_t[start:end]) ** 2).item()
                omega_loss += torch.mean((out[:, 2:] - omega_t[start:end]) ** 2).item()
                total_steps += 1
            del feats, theta_t, omega_t

    torch.cuda.empty_cache()
    trainer.train()
    if total_steps == 0:
        return float("nan"), float("nan")
    return theta_loss / total_steps, omega_loss / total_steps


def ComputeRollout(trainer, device, n_cases=100):
    # True time-marching over [0, t_data_max] // 마칭 롤아웃
    trainer.eval()
    n = min(n_cases, len(trainer.val_cases))
    case_idx = trainer.val_cases[:n]
    n_windows = int(round(trainer.dataCfg.t_data_max / trainer.dataCfg.march_dt))

    times, theta_pred, omega_pred = trainer.MarchRollout(case_idx, n_windows)
    T = times.shape[0]

    gi = torch.arange(T, device=device).clamp(max=trainer.n_step - 1)
    theta_true = trainer.data[case_idx][:, gi][:, :, [1, 3]]
    omega_true = trainer.data[case_idx][:, gi][:, :, [2, 4]]

    theta_mse = torch.mean((theta_pred - theta_true) ** 2).item()
    omega_mse = torch.mean((omega_pred - omega_true) ** 2).item()
    del theta_pred, omega_pred, theta_true, omega_true
    torch.cuda.empty_cache()

    trainer.train()
    return theta_mse, omega_mse


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


def ReadLogBests(log_path):
    val_best = float("inf")
    rollout_best = float("inf")
    extrap_best = float("inf")
    ode_best = float("inf")
    ode_last = None
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
                if o == o:
                    if o < ode_best:
                        ode_best = o
                    ode_last = o
            except (ValueError, KeyError, TypeError):
                pass
    except FileNotFoundError:
        pass
    return val_best, rollout_best, extrap_best, ode_best, ode_last


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


def ApplyLrDrop(trainer, epoch, drop_epoch, drop_to):
    # OdeScheduler와 별개 — 지정 에폭에 LR 1회 하향 // fixed epoch LR cap
    if drop_epoch <= 0 or drop_to <= 0.0:
        return
    if epoch < drop_epoch:
        return
    cur_lr = trainer.optimizer.param_groups[0]["lr"]
    if cur_lr <= drop_to:
        trainer._lr_drop_done = True
        return
    for pg in trainer.optimizer.param_groups:
        pg["lr"] = drop_to
    if not trainer._lr_drop_done:
        tqdm.write(
            f"[lr_drop] epoch {epoch}  lr {cur_lr:.2e} -> {drop_to:.2e}"
            f"  (OdeScheduler independent)"
        )
    trainer._lr_drop_done = True


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
    # Replay active_cases + RAR collocation // flip 파인튜닝 전용 훅

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

    def OnStepEnd(self, trainer, step, epoch, phase1, t_lo=None, t_hi=None):
        # replay epoch: flip RAR 풀 갱신 스킵 // nonflip 콜로케이션은 매 세그먼트 fresh sample
        if getattr(trainer, "_replay_active", False):
            return False
        # PLAN: rar_every마다 모든 세그먼트 풀 refine // 한 세그먼트만 갱신되던 버그 수정
        if phase1:
            return False
        every = trainer.collocCfg.rar_every
        if every <= 0 or step % every != 0:
            return False
        for seg_lo, seg_hi in trainer.segments:
            p50, p90, n_pool = trainer.RefineCollocPool(seg_lo, seg_hi)
            tqdm.write(
                f"[rar] step {step} seg [{seg_lo},{seg_hi}]  "
                f"residual p50={p50:.2e} p90={p90:.2e}  pool={n_pool}"
            )
        return True


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
    ode_ema_init=None,
    ode_sched_state=None,
    hooks=None,
    save_rollout_best=False,
):
    if hooks is None:
        hooks = PretrainHooks()

    last_save = time.time()
    metric = float("inf")
    losses = {}

    DATA_WARMUP_EPOCHS = t_params.get("warmup_epochs", 200)
    GRAD_BALANCE_EVERY = t_params.get("grad_balance_every", 25)
    GRAD_BALANCE_BATCHES = t_params.get("grad_balance_batches", 3)
    USE_GRAD_BALANCE = t_params.get("use_grad_balance", True)

    ROLL_DELAY = t_params.get("roll_activate_delay", 200)
    ROLL_ACTIVATE_EPOCH = t_params.get("roll_activate_epoch", None)
    ROLL_RAMP = t_params.get("roll_ramp_epochs", 150)
    ROLL_CASES = t_params.get("roll_cases", 256)
    ROLL_DEPTH_MAX = t_params.get("roll_depth_max", 3)
    ROLL_POINTS = t_params.get("roll_points", 12)
    roll_enabled = ROLL_CASES > 0 and trainer.trainCfg.lambda_roll > 0

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

    hooks.OnLoopStart(trainer, device, data_cfg, t_params)

    ode_sched = ode_sched_cls(trainer.optimizer, ode_s_params, ode_ema_init)
    if ode_sched_state is not None:
        ode_sched.LoadStateDict(ode_sched_state)
    if start_epoch > DATA_WARMUP_EPOCHS:
        ode_sched.phase2_dropped = True
    if start_epoch >= LR_DROP_EPOCH and LR_DROP_EPOCH > 0:
        trainer._lr_drop_done = True

    pbar = tqdm(range(start_epoch, t_params["max_epochs"]), desc="Training", dynamic_ncols=True)

    try:
        for epoch in pbar:
            epoch_sums = {k: torch.tensor(0.0, device=device)
                          for k in ["data", "kin", "phys", "energy", "ic", "roll"]}
            epoch_steps = 0
            metric_t = torch.tensor(float("inf"), device=device)
            phase1 = epoch < DATA_WARMUP_EPOCHS

            if ROLL_ACTIVATE_EPOCH is not None:
                roll_anchor = int(ROLL_ACTIVATE_EPOCH)
            else:
                roll_anchor = (
                    (ode_sched.activate_epoch + ROLL_DELAY)
                    if ode_sched.activate_epoch is not None
                    else None
                )
            roll_on = roll_enabled and not phase1 and roll_anchor is not None and epoch >= roll_anchor
            if roll_on:
                trainer.roll_ramp = min(1.0, (epoch - roll_anchor + 1) / max(ROLL_RAMP, 1))

            hooks.OnEpochStart(trainer, epoch, phase1, ode_sched, t_params)
            if epoch == DATA_WARMUP_EPOCHS:
                ode_sched.DropOnPhase2(epoch)
            ApplyLrDrop(trainer, epoch, LR_DROP_EPOCH, LR_DROP_TO)
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

            if (USE_GRAD_BALANCE and not phase1 and not replay_epoch
                    and trainer.phys_ramp >= 0.5
                    and (e2 % GRAD_BALANCE_EVERY == 0)):
                g = trainer.RebalanceGradScales(GRAD_BALANCE_BATCHES)
                gs = trainer.grad_scale
                tqdm.write(
                    f"[balance] epoch {epoch}  ‖g‖ data={g['data']:.2e} kin={g['kin']:.2e} "
                    f"phys={g['phys']:.2e} energy={g['energy']:.2e} ic={g['ic']:.2e} | "
                    f"scale kin={gs['kin']:.2e} phys={gs['phys']:.2e} energy={gs['energy']:.2e} ic={gs['ic']:.2e}")

            for t_lo, t_hi in trainer.segments:
                frame = trainer.SegmentFrame(t_lo, t_hi)
                bs = min(data_cfg.batch_size, frame["n_total"])

                for _ in range(t_params["steps_per_segment"]):
                    batch = trainer.SegmentBatch(frame, bs)
                    ic_parts = trainer.ICSamplesRaw()
                    colloc_meta = None
                    if not phase1:
                        colloc_meta = hooks.SampleColloc(trainer, t_lo, t_hi, epoch, phase1)

                    roll_loss = None
                    if roll_on:
                        sel = trainer.active_cases[
                            torch.randint(0, len(trainer.active_cases), (ROLL_CASES,), device=device)
                        ]
                        depth = int(torch.randint(1, ROLL_DEPTH_MAX + 1, (1,)).item())
                        roll_loss = trainer.RolloutLoss(sel, depth, ROLL_POINTS)

                    if phase1:
                        metric_t, losses = trainer.BackwardDataIC(batch, ic_parts)
                    else:
                        metric_t, losses = trainer.BackwardAll(batch, colloc_meta, ic_parts, roll_loss=roll_loss)

                    if not math.isfinite(metric_t):
                        trainer.optimizer.zero_grad()
                        nan_keys = [k for k, v in losses.items() if not math.isfinite(v)]
                        print(f"[NaN] skipping step — affected: {nan_keys}")
                        step += 1
                        continue

                    torch.nn.utils.clip_grad_norm_(trainer.parameters(), train_cfg.grad_clip)
                    trainer.optimizer.step()

                    metric_t = float(metric_t)
                    step += 1
                    trainer._global_step = step

                    for k in losses:
                        epoch_sums[k] = epoch_sums[k] + losses[k]
                    epoch_steps += 1

                    hooks.OnStepEnd(trainer, step, epoch, phase1, t_lo, t_hi)

            lam = trainer.LambdaAt()
            lr = trainer.optimizer.param_groups[0]["lr"]
            metric = float(metric_t)
            avg = {k: float(epoch_sums[k]) / epoch_steps if epoch_steps > 0 else float("nan")
                   for k in epoch_sums}
            if phase1:
                avg["kin"] = avg["phys"] = avg["energy"] = float("nan")
            if not roll_on:
                avg["roll"] = float("nan")

            val_loss = float("nan")
            val_omega_loss = float("nan")
            val_metric = float("nan")
            if epoch % t_params["val_interval"] == 0:
                trainer._replay_active = False
                trainer.active_cases = trainer.val_cases
                val_loss, val_omega_loss = ComputeVal(trainer, device, data_cfg.batch_size)
                # 모델 선택 지표 = θ+ω (핸드오프를 지배하는 ω 오차 포함) // best.pt selection
                val_metric = val_loss + val_omega_loss
                if val_metric < val_best:
                    val_best = val_metric

            # replay 에폭(쉬운 nonflip)은 적응 장치에서 격리 // 분포 스위칭이 λ·LR 스케줄을 오염시키지 않도록
            if not phase1 and epoch_steps > 0 and not replay_epoch:
                if trainer.phys_ramp >= 1.0:
                    trainer.UpdateReLoBRaLo(avg)
                ode_sched.Step(avg["phys"], epoch)

            rollout_loss = float("nan")
            rollout_omega_loss = float("nan")
            extrap_loss = float("nan")
            extrap_omega_loss = float("nan")
            if epoch % t_params["rollout_interval"] == 0:
                trainer._replay_active = False
                rollout_loss, rollout_omega_loss = ComputeRollout(
                    trainer, device, n_cases=t_params["rollout_cases"]
                )
                if rollout_loss < rollout_best:
                    rollout_best = rollout_loss
                    if save_rollout_best and not math.isnan(rollout_loss):
                        if trainer.MaybeSaveBest(ckpt_dir, step, rollout_loss, ode_sched.StateDict()):
                            tqdm.write(f"[best_rollout] step {step}  rollout={rollout_loss:.3e}")
                tqdm.write(
                    f"[rollout] epoch {epoch}  theta={rollout_loss:.3e}"
                    f"  omega={rollout_omega_loss:.3e}  best={rollout_best:.3e}"
                )

                if extrap_gt is not None:
                    extrap_loss, extrap_omega_loss = ComputeExtrap(trainer, extrap_cases, extrap_gt)
                    if extrap_loss < extrap_best:
                        extrap_best = extrap_loss
                    tqdm.write(
                        f"[extrap]  epoch {epoch}  theta={extrap_loss:.3e}"
                        f"  omega={extrap_omega_loss:.3e}  best={extrap_best:.3e}"
                    )

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

            if epoch % t_params["val_interval"] == 0:
                if not math.isnan(val_metric):
                    if trainer.MaybeSaveBest(ckpt_dir, step, val_metric, ode_sched.StateDict()):
                        tqdm.write(f"[best] step {step}  val(θ+ω)={val_metric:.3e}")

            if not phase1 and epoch_steps > 0:
                if trainer.MaybeSaveBestODE(ckpt_dir, step, avg["phys"], ode_sched.StateDict()):
                    tqdm.write(f"[best_ode] step {step}  ode={avg['phys']:.3e}")

            if not math.isnan(extrap_loss):
                if trainer.MaybeSaveBestExtrap(ckpt_dir, step, extrap_loss, ode_sched.StateDict()):
                    tqdm.write(f"[best_extrap] step {step}  extrap={extrap_loss:.3e}")

    except KeyboardInterrupt:
        # latest만 저장 — 학습 total loss(metric)와 val 지표는 단위가 달라 best.pt를 덮으면 안 됨 // unit-mismatch guard
        trainer.SaveLatest(ckpt_dir, step, metric, ode_sched.StateDict())
        tqdm.write(f"저장 완료 (latest): {ckpt_dir}")
