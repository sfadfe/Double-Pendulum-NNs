import argparse
import csv
import math
import shutil
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

import state.Double_pendulum as Dp
from PINNsTrainer import LoadConfig, OdeScheduler, PINNTrainer

CSV_HEADER = [
    "Epoch", "LR",
    "AvgLoss_Data", "AvgLoss_Kin", "AvgLoss_ODE", "AvgLoss_Energy", "AvgLoss_IC",
    "Val", "ValOmega", "RolloutBestVal", "RolloutOmega", "BestVal",
    "BestMetric",
    "Lambda_ODE", "Lambda_Energy",
    "ExtrapBest", "ExtrapOmega",
    "AvgLoss_Roll", "Lambda_Roll",
]


def Init():
    ts = time.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = Path(__file__).parent.parent / "model" / ts
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


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
    # True time-marching over the data window [0, t_data_max]; includes window-stitching error // 마칭 롤아웃
    trainer.eval()
    n = min(n_cases, len(trainer.val_cases))
    case_idx = trainer.val_cases[:n]
    n_windows = int(round(trainer.dataCfg.t_data_max / trainer.dataCfg.march_dt))

    times, theta_pred, omega_pred = trainer.MarchRollout(case_idx, n_windows)
    T = times.shape[0]                                            # n_windows*steps + 1

    # rollout grid matches the data grid (same dt) // 데이터 격자와 정합
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
    # RK4 from state0 over a fine grid, sampled at offsets t_off (s past window end) // 외삽 GT
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
    return out  # (K, 4) [θ1,ω1,θ2,ω2]


def BuildExtrapGT(trainer, case_idx, t_data_max, t_ext, rk4_dt):
    # March 격자에 맞춰 외삽 GT를 한 번만 precompute (case·params 고정) // interp=저장데이터, extrap=RK4 연장
    dev = trainer.device
    n_windows = int(round(t_ext / trainer.dataCfg.march_dt))
    times, _, _ = trainer.MarchRollout(case_idx, n_windows)
    times_np = times.cpu().numpy()
    T = times_np.shape[0]
    ext_mask = times_np > t_data_max + 1e-9

    gi = torch.arange(T, device=dev).clamp(max=trainer.n_step - 1)
    th_true = trainer.data[case_idx][:, gi][:, :, [1, 3]].clone()
    om_true = trainer.data[case_idx][:, gi][:, :, [2, 4]].clone()

    # extrap GT: t_data_max 상태에서 RK4 연장 (한 번만 계산) // chaotic이라 모니터용 dt 사용
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
    # 마칭 외삽 후 t_data_max 이후 구간만 MSE // 캐시된 GT와 비교
    trainer.eval()
    _, theta_pred, omega_pred = trainer.MarchRollout(case_idx, gt["n_windows"])
    em = gt["ext_mask"]
    th = ((theta_pred[:, em] - gt["th_true"][:, em]) ** 2).mean().item()
    om = ((omega_pred[:, em] - gt["om_true"][:, em]) ** 2).mean().item()
    torch.cuda.empty_cache()
    trainer.train()
    return th, om


def _AppendResumeLog(cfg_path, epoch, step, ckpt_name, changes):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"\n# [resume] {ts}  epoch={epoch}  step={step}  ckpt={ckpt_name}"]
    for key, (old_val, new_val) in changes.items():
        lines.append(f"#   {key}: {old_val} -> {new_val}")
    with open(cfg_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _ReadLogBests(log_path):
    val_best = float("inf")
    rollout_best = float("inf")
    extrap_best = float("inf")
    ode_best = float("inf")
    ode_last = None
    try:
        # sort rows by epoch so ode_last reflects the most recent ODE loss // log는 역순 저장이므로 에폭 순 정렬 후 처리
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
                v = float(row["Val"])
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


def _ReadLogRows(log_path):
    # 기존 log 행 읽기 — 에폭 순 정렬 (역순 저장 포맷 모두 처리) // resume 시 log 연속성 유지
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


def _TrainLoop(trainer, device, train_cfg, data_cfg, t_params,
               start_epoch, step, val_best, rollout_best, extrap_best,
               log_path, log_rows, ckpt_dir,
               ode_s_params=None, ode_ema_init=None, ode_sched_state=None):
    last_save = time.time()
    metric = float("inf")
    losses = {}

    DATA_WARMUP_EPOCHS = t_params.get("warmup_epochs", 200)
    GRAD_BALANCE_EVERY = t_params.get("grad_balance_every", 25)
    GRAD_BALANCE_BATCHES = t_params.get("grad_balance_batches", 3)
    # B(grad-norm 균등화) on/off. false면 grad_scale=1.0 고정 → ReLoBRaLo 단독 // ablation 토글
    USE_GRAD_BALANCE = t_params.get("use_grad_balance", True)

    # rollout-aware (pushforward) 손실 설정 // 윈도우 핸드오프 교정
    # 게이트: 고정 에폭이 아니라 스케줄러 활성화(물리가 activate_threshold 돌파 → LR ratchet 시작) 이후. 조기 도입은 활성화 자체를 막음
    ROLL_DELAY = t_params.get("roll_activate_delay", 200)              # 활성화 후 roll 켜기까지 대기 에폭 (LR ratchet 여유)
    ROLL_RAMP = t_params.get("roll_ramp_epochs", 150)                  # roll 가중치 0→1 선형 램프 길이 (충격 방지)
    ROLL_CASES = t_params.get("roll_cases", 256)                       # 0이면 비활성
    ROLL_DEPTH_MAX = t_params.get("roll_depth_max", 3)                 # no_grad 마칭 깊이 상한
    ROLL_POINTS = t_params.get("roll_points", 12)                      # grad 윈도우 데이터 매칭 점 수
    roll_enabled = ROLL_CASES > 0 and trainer.trainCfg.lambda_roll > 0

    # 외삽 평가용 고정 케이스 + GT 한 번만 precompute // case·params 불변이라 재사용
    n_ext = min(t_params.get("extrap_cases", 50), len(trainer.val_cases))
    extrap_cases = trainer.val_cases[:n_ext]
    extrap_gt = BuildExtrapGT(
        trainer, extrap_cases, float(data_cfg.t_data_max),
        t_params.get("extrap_t_ext", 6.0), t_params.get("extrap_rk4_dt", 1e-4),
    )
    tqdm.write(f"[extrap] GT precomputed: {n_ext} cases, t_ext={t_params.get('extrap_t_ext', 6.0)}s")

    ode_sched = OdeScheduler(trainer.optimizer, ode_s_params, ode_ema_init)
    if ode_sched_state is not None:
        ode_sched.LoadStateDict(ode_sched_state)

    pbar = tqdm(range(start_epoch, t_params["max_epochs"]), desc="Training", dynamic_ncols=True)

    try:
        for epoch in pbar:
            # Accumulate as GPU tensors; .item() deferred to end of epoch // GPU→CPU sync 에폭당 1회로 집약
            epoch_sums = {k: torch.tensor(0.0, device=device)
                          for k in ["data", "kin", "phys", "energy", "ic", "roll"]}
            epoch_steps = 0
            metric_t = torch.tensor(float("inf"), device=device)
            phase1 = epoch < DATA_WARMUP_EPOCHS   # 데이터 워밍업 단계 // data-only warmup phase

            # rollout-aware 게이트 + 램프: 스케줄러 활성화 에폭 + 대기 후 켜고, grad_scale[roll] 슬롯에 0→1 램프
            # (B는 roll grad-norm을 측정 안 해 grad_scale[roll]이 1.0 고정 → 램프 슬롯으로 재활용, NormalizedTotal이 자동 적용)
            roll_anchor = (ode_sched.activate_epoch + ROLL_DELAY) if ode_sched.activate_epoch is not None else None
            roll_on = roll_enabled and not phase1 and roll_anchor is not None and epoch >= roll_anchor
            if roll_on:
                trainer.roll_ramp = min(1.0, (epoch - roll_anchor + 1) / max(ROLL_RAMP, 1))

            perm = torch.randperm(len(trainer.train_pool), device=trainer.device)[:trainer.max_cases]
            trainer.active_cases = trainer.train_pool[perm]

            # Phase 3: 콜로케이션 IC 섭동 폭 램프 (ic_sigma=0이면 항상 0=비활성) // B보다 먼저 — B가 같은 sigma로 측정
            if not phase1:
                e2 = epoch - DATA_WARMUP_EPOCHS
                warm = max(trainer.collocCfg.ic_sigma_warmup, 1)
                trainer._colloc_ic_sigma = trainer.collocCfg.ic_sigma * min(1.0, e2 / warm)

            # Phase 2: B(grad-norm 균등화) — 진입 시 + grad_balance_every마다. 그 사이 grad_scale 고정
            if USE_GRAD_BALANCE and not phase1 and ((epoch - DATA_WARMUP_EPOCHS) % GRAD_BALANCE_EVERY == 0):
                g = trainer.RebalanceGradScales(GRAD_BALANCE_BATCHES)
                gs = trainer.grad_scale
                tqdm.write(
                    f"[balance] epoch {epoch}  ‖g‖ data={g['data']:.2e} kin={g['kin']:.2e} "
                    f"phys={g['phys']:.2e} energy={g['energy']:.2e} ic={g['ic']:.2e} | "
                    f"scale kin={gs['kin']:.2e} phys={gs['phys']:.2e} energy={gs['energy']:.2e} ic={gs['ic']:.2e}")

            ic = trainer.ICSamples()
            for t_lo, t_hi in trainer.segments:
                feats_full, theta_t, omega_t, _ = trainer.SegmentSamples(t_lo, t_hi)

                n_total = feats_full.shape[0]
                bs = min(data_cfg.batch_size, n_total)
                # Collocation sampled once per segment, reused across steps — LHS already covers τ uniformly
                # // 세그먼트당 1회 샘플: 20스텝 재사용. LHS로 균등 커버리지 보장 → 품질 영향 없음
                if not phase1:
                    colloc = trainer.SampleCollocation(t_lo, t_hi)

                for i in range(t_params["steps_per_segment"]):
                    idx = torch.randint(0, n_total, (bs,), device=device)
                    batch = (feats_full[idx], theta_t[idx], omega_t[idx])

                    # Phase 1: data+ic만 / Phase 2: 풀 손실
                    if phase1:
                        losses = trainer.ComputeDataICLosses(batch, ic)
                    else:
                        losses = trainer.ComputeAllLosses(batch, colloc, ic)

                    # rollout-aware: 예측 IC 체인으로 윈도우 핸드오프 교정 // pushforward
                    if roll_on:
                        sel = trainer.active_cases[
                            torch.randint(0, len(trainer.active_cases), (ROLL_CASES,), device=device)
                        ]
                        depth = int(torch.randint(1, ROLL_DEPTH_MAX + 1, (1,)).item())
                        losses["roll"] = trainer.RolloutLoss(sel, depth, ROLL_POINTS)

                    total = trainer.NormalizedTotal(losses)

                    # Single sync to check NaN instead of per-loss checks // NaN 검사를 total 하나로 통합 (sync 5-6회 → 1회)
                    if not total.isfinite():
                        trainer.optimizer.zero_grad()
                        nan_keys = [k for k, v in losses.items() if not v.isfinite()]
                        print(f"[NaN] skipping step — affected: {nan_keys}")
                        step += 1
                        continue

                    trainer.optimizer.zero_grad()
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        trainer.parameters(), train_cfg.grad_clip
                    )
                    trainer.optimizer.step()

                    metric_t = total.detach()
                    step += 1

                    for k in losses:
                        epoch_sums[k] = epoch_sums[k] + losses[k].detach()
                    epoch_steps += 1

            lam = trainer.LambdaAt()
            lr = trainer.optimizer.param_groups[0]["lr"]
            metric = float(metric_t)
            # Deferred .item(): one sync per key at epoch end // 에폭 끝 1회 sync
            avg = {k: float(epoch_sums[k]) / epoch_steps if epoch_steps > 0 else float("nan")
                   for k in epoch_sums}
            if phase1:
                avg["kin"] = avg["phys"] = avg["energy"] = float("nan")
            if not roll_on:
                avg["roll"] = float("nan")

            val_loss = float("nan")
            val_omega_loss = float("nan")
            if epoch % t_params["val_interval"] == 0:
                # val은 고정 케이스로 일관성 확보 // consistent metric for val tracking
                trainer.active_cases = trainer.val_cases
                val_loss, val_omega_loss = ComputeVal(trainer, device, data_cfg.batch_size)
                if val_loss < val_best:
                    val_best = val_loss

            if not phase1 and epoch_steps > 0:
                trainer.UpdateReLoBRaLo(avg)
                ode_sched.Step(avg["phys"], epoch)

            rollout_loss = float("nan")
            rollout_omega_loss = float("nan")
            extrap_loss = float("nan")
            extrap_omega_loss = float("nan")
            if epoch % t_params["rollout_interval"] == 0:
                rollout_loss, rollout_omega_loss = ComputeRollout(trainer, device, n_cases=t_params["rollout_cases"])
                if rollout_loss < rollout_best:
                    rollout_best = rollout_loss
                tqdm.write(
                    f"[rollout] epoch {epoch}  theta={rollout_loss:.3e}"
                    f"  omega={rollout_omega_loss:.3e}  best={rollout_best:.3e}"
                )

                # 외삽 평가 (보조 지표) // 마칭을 t_data_max 밖까지 돌려 RK4 GT와 비교
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
                # 유효 roll 가중치 = λ_relo · roll_ramp(0→1) // 램프 진행 가시화
                "Lambda_Roll":    lam["roll"] * trainer.roll_ramp if roll_on else float("nan"),
            })
            # 최신 에폭이 위로 오도록 역순 재작성 // most recent epoch first
            with open(log_path, "w", newline="") as _f:
                _w = csv.DictWriter(_f, fieldnames=CSV_HEADER)
                _w.writeheader()
                for _r in reversed(log_rows):
                    _w.writerow(_r)

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
                if not math.isnan(val_loss):
                    if trainer.MaybeSaveBest(ckpt_dir, step, val_loss, ode_sched.StateDict()):
                        tqdm.write(f"[best] step {step}  val={val_loss:.3e}")

            if not phase1 and epoch_steps > 0:
                if trainer.MaybeSaveBestODE(ckpt_dir, step, avg["phys"], ode_sched.StateDict()):
                    tqdm.write(f"[best_ode] step {step}  ode={avg['phys']:.3e}")

            if not math.isnan(extrap_loss):
                if trainer.MaybeSaveBestExtrap(ckpt_dir, step, extrap_loss, ode_sched.StateDict()):
                    tqdm.write(f"[best_extrap] step {step}  extrap={extrap_loss:.3e}")

    except KeyboardInterrupt:
        trainer.SaveLatest(ckpt_dir, step, metric, ode_sched.StateDict())
        trainer.MaybeSaveBest(ckpt_dir, step, metric, ode_sched.StateDict())
        tqdm.write(f"저장 완료: {ckpt_dir}")

    finally:
        pass


def Train(MainFolderPath):
    base_dir = Path(__file__).parent.parent
    cfg_path = Path(__file__).parent / "config_copy.toml"
    net_cfg, train_cfg, colloc_cfg, data_cfg, t_params, ode_s_params = LoadConfig(cfg_path)

    shutil.copy(cfg_path, MainFolderPath / "config.toml")

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = PINNTrainer(net_cfg, train_cfg, colloc_cfg, data_cfg, device)
    trainer.Setup(base_dir, MainFolderPath)

    max_cases = t_params["max_cases"]
    n_val = min(t_params.get("n_val", 2000), trainer.n_case - max_cases)
    trainer.max_cases = max_cases

    # 앞쪽 n_val개는 held-out val 전용, 나머지가 학습 풀 // held-out split
    all_idx = torch.arange(trainer.n_case, device=trainer.device)
    trainer.val_cases    = all_idx[:n_val]
    trainer.train_pool   = all_idx[n_val:]
    trainer.active_cases = trainer.train_pool[:max_cases]

    _TrainLoop(trainer, device, train_cfg, data_cfg, t_params,
               start_epoch=0, step=0, val_best=float("inf"), rollout_best=float("inf"),
               extrap_best=float("inf"),
               log_path=MainFolderPath / "log.csv", log_rows=[],
               ckpt_dir=MainFolderPath, ode_s_params=ode_s_pamainrams)


def Resume(ckpt_path, new_lr=None, ckpt_name="latest"):
    base_dir = Path(__file__).parent.parent
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = base_dir / ckpt_path
    ckpt_dir = ckpt_path.parent
    cfg_path = ckpt_dir / "config.toml"  # 학습 때 저장해둔 config
    net_cfg, train_cfg, colloc_cfg, data_cfg, t_params, ode_s_params = LoadConfig(cfg_path)

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = PINNTrainer(net_cfg, train_cfg, colloc_cfg, data_cfg, device)
    trainer.Setup(base_dir, ckpt_dir)

    max_cases = t_params["max_cases"]
    n_val = min(t_params.get("n_val", 2000), trainer.n_case - max_cases)
    trainer.max_cases = max_cases

    # 앞쪽 n_val개는 held-out val 전용, 나머지가 학습 풀 // held-out split
    all_idx = torch.arange(trainer.n_case, device=trainer.device)
    trainer.val_cases    = all_idx[:n_val]
    trainer.train_pool   = all_idx[n_val:]
    trainer.active_cases = trainer.train_pool[:max_cases]

    # 체크포인트 로드 후 LR 덮어쓰기 — AdamW momentum은 유지
    # grad_scale·_phys_balanced는 LoadCheckpoint에서 복원됨 (B는 25에폭마다만 갱신) // resume 시 1.0 리셋 방지
    start_step, _, sched_state = trainer.LoadCheckpoint(ckpt_path)

    old_lr = trainer.optimizer.param_groups[0]["lr"]
    changes = {}
    if new_lr is not None:
        changes["lr"] = (old_lr, new_lr)
        for pg in trainer.optimizer.param_groups:
            pg["lr"] = new_lr

    steps_per_epoch = len(trainer.segments) * t_params["steps_per_segment"]
    start_epoch = start_step // steps_per_epoch
    _AppendResumeLog(cfg_path, start_epoch, start_step, ckpt_name, changes)

    log_path = ckpt_dir / "log.csv"
    val_best, rollout_best, extrap_best, ode_best, ode_last = _ReadLogBests(log_path)

    # best 게이트를 log 전체 기록 기준으로 복원 — 오래된 ckpt에서 재시작해도 best.pt/best_ode.pt 보호
    if val_best < float("inf"):
        trainer.best_metric = val_best
    if ode_best < float("inf"):
        trainer.best_ode_metric = ode_best
    if extrap_best < float("inf"):
        trainer.best_extrap_metric = extrap_best

    tqdm.write(f"[resume] epoch {start_epoch}  step {start_step}  lr {trainer.optimizer.param_groups[0]['lr']:.2e}  val_best {val_best:.3e}")

    # start_epoch 이전 행만 유지 — 재학습 구간 중복 방지 // strip rows from epochs being re-trained to avoid duplicate log entries
    prior_rows = [r for r in _ReadLogRows(log_path) if int(r["Epoch"]) < start_epoch]
    _TrainLoop(trainer, device, train_cfg, data_cfg, t_params,
               start_epoch=start_epoch, step=start_step, val_best=val_best, rollout_best=rollout_best,
               extrap_best=extrap_best,
               log_path=log_path, log_rows=prior_rows,
               ckpt_dir=ckpt_dir, ode_s_params=ode_s_params, ode_ema_init=ode_last,
               ode_sched_state=sched_state)


if __name__ == "__main__":
    # ex) python3 train.py --resume model/2026_06_03_22_15_48 --ckpt ode
    _CKPT_FILES = {"latest": "latest.pt", "val": "best.pt", "ode": "best_ode.pt", "extrap": "best_extrap.pt"}
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="재시작할 모델 폴더 경로")
    parser.add_argument("--ckpt",   type=str, default="latest", choices=_CKPT_FILES,
                        help="불러올 체크포인트 종류: latest / val / ode (기본: latest)")
    parser.add_argument("--lr",     type=float, default=None, help="재시작 시 LR 덮어쓰기")
    args = parser.parse_args()

    if args.resume:
        ckpt_path = Path(args.resume) / _CKPT_FILES[args.ckpt]
        Resume(ckpt_path, new_lr=args.lr, ckpt_name=args.ckpt)
    else:
        MainFolderPath = Init()
        Train(MainFolderPath)
