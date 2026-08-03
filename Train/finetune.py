#!/usr/bin/env python3
import argparse
import shutil
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from PINNsTrainer import LoadConfig, OdeScheduler, PINNTrainer
from Train.loop_common import FinetuneHooks, InitCkptDir, RunTrainLoop


_CKPT_FILES = {
    "latest": "latest.pt",
    "val": "best.pt",
    "ode": "best_ode.pt",
    "extrap": "best_extrap.pt",
}


def Init(out=None):
    if out is not None:
        ckpt_dir = Path(out)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir
    return InitCkptDir(suffix="_finetune")


def Finetune(
    run_dir,
    pretrain_dir,
    cfg_path,
    ckpt_name="extrap",
    seed=None,
    max_epochs=None,
    data_path=None,
    max_cases=None,
    n_val=None,
):
    base_dir = Path(__file__).parent.parent
    cfg_path = Path(cfg_path)
    net_cfg, train_cfg, colloc_cfg, data_cfg, t_params, ode_s_params = LoadConfig(cfg_path)
    if max_epochs is not None:
        t_params["max_epochs"] = max_epochs
    if data_path is not None:
        data_cfg.data_path = data_path
    if max_cases is not None:
        t_params["max_cases"] = max_cases
    if n_val is not None:
        t_params["n_val"] = n_val

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    dest_cfg = run_dir / "config.toml"
    if cfg_path.resolve() != dest_cfg.resolve():
        shutil.copy(cfg_path, dest_cfg)

    pretrain_dir = Path(pretrain_dir)
    if not pretrain_dir.is_absolute():
        pretrain_dir = base_dir / pretrain_dir
    ckpt_path = pretrain_dir / _CKPT_FILES[ckpt_name]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"pretrain checkpoint not found: {ckpt_path}")

    scaler_src = pretrain_dir / data_cfg.scaler_name
    if not scaler_src.exists():
        raise FileNotFoundError(f"pretrain scaler not found: {scaler_src}")

    torch.set_float32_matmul_precision(train_cfg.matmul_precision)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = PINNTrainer(net_cfg, train_cfg, colloc_cfg, data_cfg, device)
    trainer.SetupFinetune(base_dir, run_dir, pretrain_dir, seed=seed or 42)

    # 빌드 시 확정된 val/train 분할 사용 (replay 버퍼는 val nonflip 소스를 이미 제외) // 재현·누수차단
    n_val_set, n_train = trainer.SetSplitFromMeta()
    max_cases_req = int(t_params["max_cases"])
    trainer.max_cases = min(max_cases_req, n_train)
    trainer.active_cases = trainer.train_pool[: trainer.max_cases]

    pretrain_step = trainer.LoadWeights(ckpt_path)
    trainer.SetOptimizerAdamW()

    tqdm.write(
        f"[finetune] pretrain={pretrain_dir.name} ckpt={ckpt_path.name} "
        f"step={pretrain_step}  train={n_train} val={n_val_set}  lr={train_cfg.lr:.2e}"
    )

    RunTrainLoop(
        trainer,
        device,
        train_cfg,
        data_cfg,
        t_params,
        start_epoch=0,
        step=0,
        val_best=float("inf"),
        rollout_best=float("inf"),
        extrap_best=float("inf"),
        log_path=run_dir / "log.csv",
        log_rows=[],
        ckpt_dir=run_dir,
        ode_sched_cls=OdeScheduler,
        ode_s_params=ode_s_params,
        hooks=FinetuneHooks(),
        save_rollout_best=t_params.get("save_rollout_best", True),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flip fine-tuning with replay + flip-biased colloc")
    parser.add_argument("--pretrain", type=str, required=True, help="pretrain model 폴더")
    parser.add_argument("--ckpt", type=str, default="extrap", choices=list(_CKPT_FILES),
                        help="불러올 pretrain 체크포인트")
    parser.add_argument("--config", type=str, default=None,
                        help="config 경로 (기본: Train/config_finetune.toml)")
    parser.add_argument("--out", type=str, default=None, help="결과 저장 폴더")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None, help="에폭 수 오버라이드 (스모크용)")
    parser.add_argument("--data", type=str, default=None, help="finetune 데이터 경로 오버라이드")
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--n_val", type=int, default=None)
    args = parser.parse_args()

    cfg = args.config or str(Path(__file__).parent / "config_finetune.toml")
    if args.out is None:
        _, _, _, _, t_params, _ = LoadConfig(cfg)
        run_name = t_params.get("name", "_ft_adapter")
        run_dir = InitCkptDir(suffix=run_name)
    else:
        run_dir = Path(args.out)
        run_dir.mkdir(parents=True, exist_ok=True)
    Finetune(run_dir, args.pretrain, cfg, ckpt_name=args.ckpt, seed=args.seed,
             max_epochs=args.max_epochs, data_path=args.data,
             max_cases=args.max_cases, n_val=args.n_val)
