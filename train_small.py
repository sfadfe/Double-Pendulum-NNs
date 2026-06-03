import csv
import shutil
from pathlib import Path
import os
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
    # θ만 비교 — ω 계산 불필요하므로 no_grad 유지 // 학습에 영향 없는 순수 평가
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
    # n_cases 케이스의 전체 t_grid에서 θ 예측 vs 정답 MSE // 롤아웃 에러 계산
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
    

def Train(MainFolderPath):
    base_dir = Path(__file__).parent
    cfg_path = base_dir / "config.toml"
    net_cfg, train_cfg, colloc_cfg, data_cfg, t_params, s_params, _ = LoadConfig(cfg_path)

    # config 파일을 모델 폴더에 그대로 복사 // 실험 재현을 위한 설정 보존
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

    last_save = time.time()
    step = 0
    metric = float("inf")
    losses = {}
    val_best = float("inf")
    rollout_best = float("inf")

    pbar = tqdm(range(t_params["max_epochs"]), desc="Training", dynamic_ncols=True)

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
                "step":     step,
                "data":     f"{avg['data']:.3e}"   if epoch_steps else "-",
                "phys":     f"{avg['phys']:.3e}"   if epoch_steps else "-",
                "val":      f"{val_best:.3e}"      if val_best < float("inf") else "-",
                "rollout":  f"{rollout_best:.3e}"  if rollout_best < float("inf") else "-",
            })

            if time.time() - last_save >= t_params["save_interval_sec"]:
                trainer.SaveLatest(MainFolderPath, step, metric)
                last_save = time.time()
                tqdm.write(f"[save] latest @ step {step}")

            if epoch % t_params["val_interval"] == 0:
                save_metric = val_loss if not (val_loss != val_loss) else metric
                if trainer.MaybeSaveBest(MainFolderPath, step, save_metric):
                    tqdm.write(f"[best] step {step}  val={val_loss:.3e}")

            if epoch_steps > 0:
                if trainer.MaybeSaveBestODE(MainFolderPath, step, avg["phys"]):
                    tqdm.write(f"[best_ode] step {step}  ode={avg['phys']:.3e}")

    except KeyboardInterrupt:
        trainer.SaveLatest(MainFolderPath, step, metric)
        trainer.MaybeSaveBest(MainFolderPath, step, metric)
        tqdm.write(f"저장 완료: {MainFolderPath}")

    finally:
        csv_file.close()


if __name__ == "__main__":
    MainFolderPath = Init()
    Train(MainFolderPath)
