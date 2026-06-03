import argparse
import csv
import shutil
from pathlib import Path
import time

import torch
from tqdm import tqdm

from PINNsTrainer import LoadConfig, PINNTrainer

CSV_HEADER = [
    "Epoch", "LR",
    "AvgLoss_Data", "AvgLoss_ODE", "AvgLoss_Energy", "AvgLoss_IC",
    "Val", "RolloutBestVal", "BestVal",
    "BestMetric",
    "Lambda_ODE", "Lambda_Energy",
]


def Init():
    ts = time.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = Path(__file__).parent / "model" / ts
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def ComputeVal(trainer, device, batch_size):
    trainer.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for t_lo, t_hi in trainer.segments:
            feats, theta_t, _, _ = trainer.SegmentSamples(t_lo, t_hi)
            n_total = feats.shape[0]
            for start in range(0, n_total, batch_size):
                end = min(start + batch_size, n_total)
                theta_pred = trainer(feats[start:end])
                loss = torch.mean((theta_pred - theta_t[start:end]) ** 2)
                total_loss += loss.item()
                total_steps += 1

    trainer.train()
    return total_loss / total_steps if total_steps > 0 else float("nan")


def ComputeRollout(trainer, device, n_cases=100):
    trainer.eval()
    n = min(n_cases, trainer.n_case)
    t_grid = trainer.t_grid  # (T,)
    T = t_grid.shape[0]

    case_idx = torch.arange(n, device=device)
    t_flat = t_grid.unsqueeze(0).expand(n, -1).reshape(-1, 1)   # (n*T, 1)
    case_flat = case_idx.unsqueeze(1).expand(-1, T).reshape(-1)  # (n*T,)
    feats = trainer._BuildFeats(t_flat, case_flat)               # (n*T, 9)

    chunks = []
    with torch.no_grad():
        for start in range(0, n * T, 4096):
            end = min(start + 4096, n * T)
            chunks.append(trainer(feats[start:end]))

    theta_pred = torch.cat(chunks, dim=0).reshape(n, T, 2)  # (n, T, 2)
    theta_true = trainer.data[:n, :, [1, 3]]                # (n, T, 2) [θ1, θ2]

    mse = torch.mean((theta_pred - theta_true) ** 2).item()
    trainer.train()
    return mse


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
    ode_best = float("inf")
    ode_last = None
    try:
        with open(log_path, newline="") as f:
            reader = csv.DictReader(f, fieldnames=CSV_HEADER)
            for row in reader:
                if row.get("Epoch") in (None, "", "Epoch"):
                    continue
                try:
                    v = float(row["Val"])
                    if v == v and v < val_best:
                        val_best = v
                except (ValueError, KeyError):
                    pass
                try:
                    r = float(row["RolloutBestVal"])
                    if r == r and r < rollout_best:
                        rollout_best = r
                except (ValueError, KeyError):
                    pass
                try:
                    o = float(row["AvgLoss_ODE"])
                    if o == o:
                        if o < ode_best:
                            ode_best = o
                        ode_last = o
                except (ValueError, KeyError):
                    pass
    except FileNotFoundError:
        pass
    return val_best, rollout_best, ode_best, ode_last


def _TrainLoop(trainer, device, train_cfg, data_cfg, t_params, scheduler,
               start_epoch, step, val_best, rollout_best,
               csv_file, writer, ckpt_dir,
               ode_s_params=None, ode_ema_init=None):
    last_save = time.time()
    metric = float("inf")
    losses = {}

    # ODE EMA scheduler state // val scheduler와 독립적으로 동작
    ode_ema = ode_ema_init
    ode_best_ema = ode_ema_init if ode_ema_init is not None else float("inf")
    ode_stall = 0

    pbar = tqdm(range(start_epoch, t_params["max_epochs"]), desc="Training", dynamic_ncols=True)

    try:
        for epoch in pbar:
            epoch_sums = {"data": 0.0, "phys": 0.0, "energy": 0.0, "ic": 0.0}
            epoch_steps = 0

            ic = trainer.ICSamples()
            for t_lo, t_hi in trainer.segments:
                feats_full, theta_t, omega_t, _ = trainer.SegmentSamples(t_lo, t_hi)

                n_total = feats_full.shape[0]
                bs = min(data_cfg.batch_size, n_total)

                for i in range(t_params["steps_per_segment"]):
                    idx = torch.randint(0, n_total, (bs,), device=device)
                    batch = (feats_full[idx], theta_t[idx], omega_t[idx])
                    colloc = trainer.SampleCollocation(t_lo, t_hi)

                    losses = trainer.ComputeAllLosses(batch, colloc, ic)

                    if trainer.IsNaN(losses):
                        step += 1
                        continue

                    if trainer.AccumulateWarmup(losses):
                        step += 1
                        continue

                    total = trainer.NormalizedTotal(losses, step)

                    trainer.optimizer.zero_grad()
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        trainer.parameters(), train_cfg.grad_clip
                    )
                    trainer.optimizer.step()

                    metric = total.item()
                    step += 1

                    for k in epoch_sums:
                        epoch_sums[k] += losses[k].item()
                    epoch_steps += 1

            lam = trainer.LambdaAt(step)
            lr = trainer.optimizer.param_groups[0]["lr"]
            avg = {k: epoch_sums[k] / epoch_steps if epoch_steps > 0 else float("nan")
                   for k in epoch_sums}

            val_loss = float("nan")
            if epoch % t_params["val_interval"] == 0:
                val_loss = ComputeVal(trainer, device, data_cfg.batch_size)
                if val_loss < val_best:
                    val_best = val_loss
                scheduler.step(val_loss)

            # ODE EMA scheduler // val scheduler와 독립, 매 에폭 갱신
            if epoch_steps > 0 and ode_s_params is not None:
                if ode_ema is None:
                    ode_ema = avg["phys"]
                    ode_best_ema = ode_ema
                else:
                    b = ode_s_params["ema_beta"]
                    ode_ema = b * ode_ema + (1.0 - b) * avg["phys"]
                    # activate_threshold 아래일 때만 stall 카운터 진행 // 초기 고loss 구간에서 LR 조기 감소 방지
                    if ode_ema < ode_s_params["activate_threshold"]:
                        if ode_ema < ode_best_ema * (1.0 - ode_s_params["rel_tol"]):
                            ode_best_ema = ode_ema
                            ode_stall = 0
                        else:
                            ode_stall += 1
                    if ode_stall >= ode_s_params["patience"]:
                        cur_lr = trainer.optimizer.param_groups[0]["lr"]
                        new_lr = max(cur_lr * ode_s_params["factor"], ode_s_params["min_lr"])
                        if new_lr < cur_lr:
                            for pg in trainer.optimizer.param_groups:
                                pg["lr"] = new_lr
                            tqdm.write(
                                f"[ode_sched] epoch {epoch}  ode_ema={ode_ema:.3e}"
                                f"  lr {cur_lr:.2e} -> {new_lr:.2e}"
                            )
                        ode_stall = 0

            rollout_loss = float("nan")
            if epoch % t_params["rollout_interval"] == 0:
                rollout_loss = ComputeRollout(trainer, device, n_cases=t_params["rollout_cases"])
                if rollout_loss < rollout_best:
                    rollout_best = rollout_loss
                tqdm.write(f"[rollout] epoch {epoch}  mse={rollout_loss:.3e}  best={rollout_best:.3e}")

            writer.writerow({
                "Epoch":          epoch,
                "LR":             lr,
                "AvgLoss_Data":   avg["data"],
                "AvgLoss_ODE":    avg["phys"],
                "AvgLoss_Energy": avg["energy"],
                "AvgLoss_IC":     avg["ic"],
                "Val":            val_loss,
                "RolloutBestVal": rollout_best,
                "BestVal":        val_best,
                "BestMetric":     trainer.best_metric,
                "Lambda_ODE":     lam["phys"],
                "Lambda_Energy":  lam["energy"],
            })
            csv_file.flush()

            pbar.set_postfix({
                "step":    step,
                "data":    f"{avg['data']:.3e}"  if epoch_steps else "-",
                "phys":    f"{avg['phys']:.3e}"  if epoch_steps else "-",
                "val":     f"{val_best:.3e}"     if val_best < float("inf") else "-",
                "rollout": f"{rollout_best:.3e}" if rollout_best < float("inf") else "-",
            })

            if time.time() - last_save >= t_params["save_interval_sec"]:
                trainer.SaveLatest(ckpt_dir, step, metric)
                last_save = time.time()
                tqdm.write(f"[save] latest @ step {step}")

            if epoch % t_params["val_interval"] == 0:
                save_metric = val_loss if not (val_loss != val_loss) else metric
                if trainer.MaybeSaveBest(ckpt_dir, step, save_metric):
                    tqdm.write(f"[best] step {step}  val={val_loss:.3e}")

            if epoch_steps > 0:
                if trainer.MaybeSaveBestODE(ckpt_dir, step, avg["phys"]):
                    tqdm.write(f"[best_ode] step {step}  ode={avg['phys']:.3e}")

    except KeyboardInterrupt:
        trainer.SaveLatest(ckpt_dir, step, metric)
        trainer.MaybeSaveBest(ckpt_dir, step, metric)
        tqdm.write(f"저장 완료: {ckpt_dir}")

    finally:
        csv_file.close()


def Train(MainFolderPath):
    base_dir = Path(__file__).parent
    cfg_path = base_dir / "config_copy.toml"
    net_cfg, train_cfg, colloc_cfg, data_cfg, t_params, s_params, ode_s_params = LoadConfig(cfg_path)

    shutil.copy(cfg_path, MainFolderPath / "config.toml")

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = PINNTrainer(net_cfg, train_cfg, colloc_cfg, data_cfg, device)
    trainer.Setup(base_dir, MainFolderPath)

    max_cases = t_params["max_cases"]
    if trainer.n_case > max_cases:
        trainer.data = trainer.data[:max_cases].clone()  # drop full-dataset storage // 전체 storage 해제
        trainer.params_raw = trainer.params_raw[:max_cases]
        trainer.ic_trig = trainer.ic_trig[:max_cases]
        trainer.n_case = max_cases

    # val_loss 기준으로 개선 없으면 lr 감소 // patience는 val_interval 에폭 단위
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        mode="min",
        factor=s_params["factor"],
        patience=s_params["patience"],
        min_lr=s_params["min_lr"],
    )

    csv_path = MainFolderPath / "log.csv"
    csv_file = csv_path.open("w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
    writer.writeheader()
    csv_file.flush()

    _TrainLoop(trainer, device, train_cfg, data_cfg, t_params, scheduler,
               start_epoch=0, step=0, val_best=float("inf"), rollout_best=float("inf"),
               csv_file=csv_file, writer=writer, ckpt_dir=MainFolderPath,
               ode_s_params=ode_s_params)


def Resume(ckpt_path, new_lr=None, ckpt_name="latest"):
    ckpt_path = Path(ckpt_path)
    ckpt_dir = ckpt_path.parent
    cfg_path = ckpt_dir / "config.toml"  # 학습 때 저장해둔 config

    base_dir = Path(__file__).parent
    net_cfg, train_cfg, colloc_cfg, data_cfg, t_params, s_params, ode_s_params = LoadConfig(cfg_path)

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = PINNTrainer(net_cfg, train_cfg, colloc_cfg, data_cfg, device)
    trainer.Setup(base_dir, ckpt_dir)

    max_cases = t_params["max_cases"]
    if trainer.n_case > max_cases:
        trainer.data = trainer.data[:max_cases].clone()  # drop full-dataset storage // 전체 storage 해제
        trainer.params_raw = trainer.params_raw[:max_cases]
        trainer.ic_trig = trainer.ic_trig[:max_cases]
        trainer.n_case = max_cases

    # 체크포인트 로드 후 LR 덮어쓰기 — AdamW momentum은 유지
    start_step, _ = trainer.LoadCheckpoint(ckpt_path)

    # resume에는 warmup 불필요 — l0는 이미 복원됨 // 재실행 시 l0가 덮어써지는 것 방지
    trainer._warm_cnt = trainer.trainCfg.warmup_steps

    old_lr = trainer.optimizer.param_groups[0]["lr"]
    changes = {}
    if new_lr is not None:
        changes["lr"] = (old_lr, new_lr)
        for pg in trainer.optimizer.param_groups:
            pg["lr"] = new_lr

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        mode="min",
        factor=s_params["factor"],
        patience=s_params["patience"],
        min_lr=s_params["min_lr"],
    )

    steps_per_epoch = colloc_cfg.seg_count * t_params["steps_per_segment"]
    start_epoch = start_step // steps_per_epoch
    _AppendResumeLog(cfg_path, start_epoch, start_step, ckpt_name, changes)

    log_path = ckpt_dir / "log.csv"
    val_best, rollout_best, ode_best, ode_last = _ReadLogBests(log_path)

    # best 게이트를 log 전체 기록 기준으로 복원 — 오래된 ckpt에서 재시작해도 best.pt/best_ode.pt 보호
    if val_best < float("inf"):
        trainer.best_metric = val_best
    if ode_best < float("inf"):
        trainer.best_ode_metric = ode_best

    csv_file = log_path.open("a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
    csv_file.flush()

    tqdm.write(f"[resume] epoch {start_epoch}  step {start_step}  lr {trainer.optimizer.param_groups[0]['lr']:.2e}  val_best {val_best:.3e}")

    _TrainLoop(trainer, device, train_cfg, data_cfg, t_params, scheduler,
               start_epoch=start_epoch, step=start_step, val_best=val_best, rollout_best=rollout_best,
               csv_file=csv_file, writer=writer, ckpt_dir=ckpt_dir,
               ode_s_params=ode_s_params, ode_ema_init=ode_last)


if __name__ == "__main__":
    # ex) python3 train.py --resume model/2026_05_26_23_23_12 --ckpt ode --lr 5e-04
    _CKPT_FILES = {"latest": "latest.pt", "val": "best.pt", "ode": "best_ode.pt"}

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
