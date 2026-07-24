#phase 1 vram peak 1.6gb, phase2 vram 2.3gb ->1.7gb


import argparse
import shutil
from pathlib import Path
import sys  

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from PINNsTrainer import LoadConfig, OdeScheduler, PINNTrainer
from Train.loop_common import (
    AppendResumeLog,
    InitCkptDir,
    ReadLogBests,
    ReadLogRows,
    RunTrainLoop,
    SetupCaseSplit,
)


def Train(MainFolderPath):
    base_dir = Path(__file__).parent.parent
    cfg_path = Path(__file__).parent / "config.toml"
    net_cfg, train_cfg, colloc_cfg, data_cfg, t_params, ode_s_params = LoadConfig(cfg_path)

    shutil.copy(cfg_path, MainFolderPath / "config.toml")

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"     

    trainer = PINNTrainer(net_cfg, train_cfg, colloc_cfg, data_cfg, device)
    trainer.Setup(base_dir, MainFolderPath)
    SetupCaseSplit(trainer, t_params)

    RunTrainLoop(
        trainer, device, train_cfg, data_cfg, t_params,
        start_epoch=0, step=0, val_best=float("inf"), rollout_best=float("inf"),
        extrap_best=float("inf"),
        log_path=MainFolderPath / "log.csv", log_rows=[],
        ckpt_dir=MainFolderPath, ode_sched_cls=OdeScheduler,
        ode_s_params=ode_s_params,
    )


def Resume(ckpt_path, new_lr=None, ckpt_name="latest"):
    base_dir = Path(__file__).parent.parent
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = base_dir / ckpt_path
    ckpt_dir = ckpt_path.parent
    cfg_path = ckpt_dir / "config.toml"
    net_cfg, train_cfg, colloc_cfg, data_cfg, t_params, ode_s_params = LoadConfig(cfg_path)

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = PINNTrainer(net_cfg, train_cfg, colloc_cfg, data_cfg, device)
    trainer.Setup(base_dir, ckpt_dir)
    SetupCaseSplit(trainer, t_params)

    start_step, _, sched_state = trainer.LoadCheckpoint(ckpt_path)

    old_lr = trainer.optimizer.param_groups[0]["lr"]
    changes = {}
    if new_lr is not None:
        changes["lr"] = (old_lr, new_lr)
        for pg in trainer.optimizer.param_groups:
            pg["lr"] = new_lr

    steps_per_epoch = len(trainer.segments) * t_params["steps_per_segment"]
    start_epoch = start_step // steps_per_epoch
    AppendResumeLog(cfg_path, start_epoch, start_step, ckpt_name, changes)

    log_path = ckpt_dir / "log.csv"
    val_best, rollout_best, extrap_best, ode_best, ode_last = ReadLogBests(
        log_path, warmup_epochs=t_params.get("warmup_epochs", 0)
    )

    if val_best < float("inf"):
        trainer.best_metric = val_best
    if ode_best < float("inf"):
        trainer.best_ode_metric = ode_best
    if extrap_best < float("inf"):
        trainer.best_extrap_metric = extrap_best

    tqdm.write(
        f"[resume] epoch {start_epoch}  step {start_step}  "
        f"lr {trainer.optimizer.param_groups[0]['lr']:.2e}  val_best {val_best:.3e}"
    )

    prior_rows = [r for r in ReadLogRows(log_path) if int(r["Epoch"]) < start_epoch]
    RunTrainLoop(
        trainer, device, train_cfg, data_cfg, t_params,
        start_epoch=start_epoch, step=start_step, val_best=val_best, rollout_best=rollout_best,
        extrap_best=extrap_best,
        log_path=log_path, log_rows=prior_rows,
        ckpt_dir=ckpt_dir, ode_sched_cls=OdeScheduler,
        ode_s_params=ode_s_params, ode_ema_init=ode_last,
        ode_sched_state=sched_state,
    )


if __name__ == "__main__":
    _CKPT_FILES = {"latest": "latest.pt", "val": "best.pt", "ode": "best_ode.pt", "extrap": "best_extrap.pt"}

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="재시작할 모델 폴더 경로")
    parser.add_argument("--ckpt", type=str, default="latest", choices=_CKPT_FILES,
                        help="불러올 체크포인트 종류: latest / val / ode (기본: latest)")
    parser.add_argument("--lr", type=float, default=None, help="재시작 시 LR 덮어쓰기")
    args = parser.parse_args()

    if args.resume:
        ckpt_path = Path(args.resume) / _CKPT_FILES[args.ckpt]
        Resume(ckpt_path, new_lr=args.lr, ckpt_name=args.ckpt)
    else:
        MainFolderPath = InitCkptDir()
        Train(MainFolderPath)
