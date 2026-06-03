import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from PINNsTrainer import LoadConfig, PINNTrainer

CSV_HEADER = [
    "Epoch", "Step",
    "Loss_Data", "Loss_ODE", "Loss_Energy", "Loss_IC", "Loss_Total",
    "Lambda_Data", "Lambda_ODE", "Lambda_Energy", "Lambda_IC",
    "Val", "RolloutVal",
]

_CKPT_FILES = {
    "latest":    "latest.pt",
    "val":       "best.pt",
    "ode":       "best_ode.pt",
    "ft_latest": "ft_latest.pt",
    "ft_val":    "ft_best.pt",
    "ft_ode":    "ft_best_ode.pt",
}


def ComputeVal(trainer, device, batch_size):
    trainer.eval()
    t_min = float(trainer.t_grid[0])
    t_max = float(trainer.t_grid[-1])
    feats, theta_t, _, _ = trainer.SegmentSamples(t_min, t_max)
    n_total = feats.shape[0]

    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
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
    t_grid = trainer.t_grid
    T = t_grid.shape[0]

    case_idx = torch.arange(n, device=device)
    t_flat = t_grid.unsqueeze(0).expand(n, -1).reshape(-1, 1)
    case_flat = case_idx.unsqueeze(1).expand(-1, T).reshape(-1)
    feats = trainer._BuildFeats(t_flat, case_flat)

    chunks = []
    with torch.no_grad():
        for start in range(0, n * T, 4096):
            end = min(start + 4096, n * T)
            chunks.append(trainer(feats[start:end]))

    theta_pred = torch.cat(chunks, dim=0).reshape(n, T, 2)
    theta_true = trainer.data[:n, :, [1, 3]]

    mse = torch.mean((theta_pred - theta_true) ** 2).item()
    trainer.train()
    return mse


def _ReadFtLog(log_path):
    ft_best_val = float("inf")
    ft_best_ode = float("inf")
    last_epoch  = -1
    last_step   = 0
    try:
        with open(log_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    v = float(row["Val"])
                    if v == v and v < ft_best_val:
                        ft_best_val = v
                except (ValueError, KeyError):
                    pass
                try:
                    o = float(row["Loss_ODE"])
                    if o == o and o < ft_best_ode:
                        ft_best_ode = o
                except (ValueError, KeyError):
                    pass
                try:
                    last_epoch = int(row["Epoch"])
                    last_step  = int(row["Step"])
                except (ValueError, KeyError):
                    pass
    except FileNotFoundError:
        pass
    return ft_best_val, ft_best_ode, last_epoch, last_step


def Finetune(ckpt_dir, ckpt_name, lbfgs_epochs=None, cfg_path=None):
    ckpt_dir = Path(ckpt_dir)
    ckpt_path = ckpt_dir / _CKPT_FILES[ckpt_name]

    base_dir = Path(__file__).parent
    if cfg_path is None:
        cfg_path = ckpt_dir / "config.toml"
    cfg_path = Path(cfg_path)

    net_cfg, train_cfg, colloc_cfg, data_cfg, t_params, _, _ode = LoadConfig(cfg_path)

    if lbfgs_epochs is None:
        lbfgs_epochs = t_params.get("lbfgs_epochs", 50)

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

    ckpt = torch.load(ckpt_path, map_location=device)
    trainer.load_state_dict(ckpt["model_state"])
    trainer.l0 = ckpt.get("l0", trainer.l0)
    # warmup skip — l0 이미 복원됨 // resume 시 l0 덮어쓰기 방지
    trainer._warm_cnt = trainer.trainCfg.warmup_steps
    # sigmoid lambda는 체크포인트 step 기준 포화값 고정
    final_step = ckpt.get("step", t_params["max_epochs"] * colloc_cfg.seg_count
                          * t_params["steps_per_segment"])

    ft_log_path = ckpt_dir / "ft_log.csv"
    ft_best_val, ft_best_ode, last_epoch, last_step = _ReadFtLog(ft_log_path)

    is_resume = ckpt_name.startswith("ft_")
    epoch_start = last_epoch + 1 if is_resume else 0

    csv_file = ft_log_path.open("a" if is_resume else "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
    if not is_resume:
        writer.writeheader()
    csv_file.flush()

    step = ckpt.get("step", 0)
    ic = trainer.ICSamples()
    accum = train_cfg.lbfgs_accum

    t_min = float(trainer.t_grid[0])
    t_max = float(trainer.t_grid[-1])

    # feats_full 1회 빌드 — 에폭마다 랜덤 idx로 batch 재샘플링 // colloc도 매 에폭 재샘플
    feats_full, theta_t, omega_t, _ = trainer.SegmentSamples(t_min, t_max)
    n_total = feats_full.shape[0]
    bs = min(train_cfg.lbfgs_batch, n_total)

    def _Resample():
        idx = torch.randint(0, n_total, (bs,), device=device)
        new_batch = (feats_full[idx], theta_t[idx], omega_t[idx])
        new_colloc = trainer.SampleCollocation(t_min, t_max, n=colloc_cfg.lbfgs_n_colloc)
        new_n_c = new_colloc[0].shape[0]
        new_chunk = (new_n_c + accum - 1) // accum
        return new_batch, new_colloc, new_n_c, new_chunk

    batch, colloc, n_c, chunk = _Resample()

    box = {}
    lam = trainer.LambdaAt(final_step)   # lambda 고정 — finetune 동안 불변

    # 파인튜닝 시작 시점 손실값으로 정규화 기준 계산 — warmup l0는 랜덤 초기화 기준이라 부적합
    # 정규화 기준값: 소규모 probe로만 계산 — order-of-magnitude만 필요, full batch 불필요
    _np = min(1024, batch[0].shape[0])
    _nc = min(1024, n_c)
    ft_l0 = {}
    ld = trainer.DataLoss(batch[0][:_np], batch[1][:_np], batch[2][:_np])
    li = trainer.ICLoss(ic[0], ic[1])
    lp, le = trainer.PhysicsEnergyLoss(colloc[0][:_nc], colloc[1][:_nc], colloc[2][:_nc])
    ft_l0["data"]   = max(ld.item(), 1e-12)
    ft_l0["ic"]     = max(li.item(), 1e-12)
    ft_l0["phys"]   = max(lp.item(), 1e-12)
    ft_l0["energy"] = max(le.item(), 1e-12)
    del ld, li, lp, le
    trainer.zero_grad()
    tqdm.write(f"[ft_l0] data={ft_l0['data']:.3e}  ic={ft_l0['ic']:.3e}  phys={ft_l0['phys']:.3e}  energy={ft_l0['energy']:.3e}")
    trainer.SwitchToLBFGS()

    def closure():
        trainer.optimizer.zero_grad()
        d0 = ft_l0["data"]
        i0 = ft_l0["ic"]
        p0 = ft_l0["phys"]
        e0 = ft_l0["energy"]
        wd = trainer.trainCfg.weight_decay

        # data + ic: FP32 1회
        l_data = trainer.DataLoss(*batch)
        l_ic   = trainer.ICLoss(ic[0], ic[1])
        box["data"] = l_data.item()
        box["ic"]   = l_ic.item()

        # weight decay: L2 on parameters — AdamW와 동일한 정규화 이어받기
        l_wd = 0.5 * wd * sum(p.pow(2).sum() for p in trainer.parameters())

        t_di = lam["data"] * l_data / d0 + lam["ic"] * l_ic / i0 + l_wd
        t_di.backward()
        total_val = t_di.item()

        # phys + energy: FP64, 청크별 forward+backward // peak VRAM 절감
        phys_v, energy_v = 0.0, 0.0
        for s in range(0, n_c, chunk):
            e = min(s + chunk, n_c)
            frac = (e - s) / n_c
            l_phys, l_energy = trainer.PhysicsEnergyLoss(
                colloc[0][s:e], colloc[1][s:e], colloc[2][s:e]
            )
            t_pe = (lam["phys"] * l_phys / p0 + lam["energy"] * l_energy / e0) * frac
            t_pe.backward()
            total_val += t_pe.item()
            phys_v   += l_phys.item() * frac
            energy_v += l_energy.item() * frac

        box["phys"]   = phys_v
        box["energy"] = energy_v
        box["total"]  = total_val

        return torch.tensor(total_val, device=device)

    epoch_end = epoch_start + lbfgs_epochs
    pbar = tqdm(range(epoch_start, epoch_end), desc="L-BFGS",
                total=lbfgs_epochs, dynamic_ncols=True)
    try:
        for epoch in pbar:
            # 매 에폭 재샘플 — optimizer 리셋 없이 Hessian 유지 (large batch: 에폭 간 분산 낮음)
            batch, colloc, n_c, chunk = _Resample()
            trainer.optimizer.step(closure)
            step += 1

            if box.get("total", 0.0) != box.get("total", 0.0):
                tqdm.write("[L-BFGS] NaN 발생 — 중단")
                break

            pbar.set_postfix({
                "total": f"{box['total']:.3e}",
                "data":  f"{box['data']:.3e}",
                "phys":  f"{box['phys']:.3e}",
            })

            is_last = epoch == epoch_end - 1
            val_loss = float("nan")
            rollout_loss = float("nan")
            if epoch % t_params["val_interval"] == 0 or is_last:
                val_loss = ComputeVal(trainer, device, data_cfg.batch_size)
            if epoch % t_params["rollout_interval"] == 0 or is_last:
                rollout_loss = ComputeRollout(trainer, device, n_cases=t_params["rollout_cases"])
            ode_loss = box.get("phys", float("inf"))

            writer.writerow({
                "Epoch":       epoch,
                "Step":        step,
                "Loss_Data":   box.get("data",   float("nan")),
                "Loss_ODE":    ode_loss,
                "Loss_Energy": box.get("energy", float("nan")),
                "Loss_IC":     box.get("ic",     float("nan")),
                "Loss_Total":  box.get("total",  float("nan")),
                "Lambda_Data":    lam["data"],
                "Lambda_ODE":     lam["phys"],
                "Lambda_Energy":  lam["energy"],
                "Lambda_IC":      lam["ic"],
                "Val":         val_loss,
                "RolloutVal":  rollout_loss,
            })
            csv_file.flush()

            if val_loss == val_loss and val_loss < ft_best_val:
                ft_best_val = val_loss
                trainer._SaveCheckpoint(str(ckpt_dir / "ft_best.pt"), step, val_loss)
                tqdm.write(f"[ft_best] epoch{epoch}  val={val_loss:.3e}")

            if ode_loss < ft_best_ode:
                ft_best_ode = ode_loss
                trainer._SaveCheckpoint(str(ckpt_dir / "ft_best_ode.pt"), step, ode_loss)
                tqdm.write(f"[ft_best_ode] epoch{epoch}  ode={ode_loss:.3e}")

    except KeyboardInterrupt:
        tqdm.write("[L-BFGS] 중단 — 체크포인트 저장 중...")

    trainer._SaveCheckpoint(str(ckpt_dir / "ft_latest.pt"), step, box.get("total", float("inf")))
    csv_file.close()
    tqdm.write(f"파인튜닝 완료: {ckpt_dir}")


if __name__ == "__main__":
    # 신규: python3 finetune.py <폴더> --ckpt ode --epochs 100
    # resume: python3 finetune.py <폴더> --ckpt ode --resume --epochs 50

    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", type=str,
                        help="체크포인트 폴더 경로")
    parser.add_argument("--ckpt", type=str, default="ode", choices=["latest", "val", "ode"],
                        help="불러올 체크포인트 종류 (기본: ode)")
    parser.add_argument("--resume", action="store_true",
                        help="ft_ 체크포인트에서 재시작 (--ckpt ode ft_best_ode.pt 로드)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="L-BFGS 반복 수 (기본: config의 lbfgs_epochs)")
    parser.add_argument("--cfg", type=str, default=None,
                        help="config.toml 경로 (기본: ckpt_dir/config.toml)")
    args = parser.parse_args()

    ckpt_name = f"ft_{args.ckpt}" if args.resume else args.ckpt
    Finetune(args.ckpt_dir, ckpt_name, args.epochs, args.cfg) 
