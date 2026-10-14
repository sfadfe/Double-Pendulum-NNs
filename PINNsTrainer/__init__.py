import os
from pathlib import Path

import torch

# hipBLASLt's default kernels for the FiLM last-layer backward / weight-grad GEMMs are slow on gfx1201; pin the faster
# shipped ones before the first GEMM. Step 65.2 -> 62.5 ms (R9700, 2026-10-12), gradients 2.4e-7 from the default.
# Rows are ignored on another library build; HIPBLASLT_TUNING_OVERRIDE_FILE=/dev/null switches it off.
# // gfx1201에서 FiLM 마지막 층 backward / weight grad GEMM의 느린 기본 커널을 빠른 내장 커널로 고정 (첫 GEMM 전에 설정)
os.environ.setdefault("HIPBLASLT_TUNING_OVERRIDE_FILE", str(Path(__file__).with_name("hipblaslt_gfx1201.txt")))

# Run backward on the calling thread. With the device worker thread every backward() (7 per step) hands the work
# over and sleeps; the scheduler then moves the main thread between cores and the step alternates between 62.2 and
# 64.7 ms. Same gradients. // backward를 호출 스레드에서 실행: 스레드 넘김(스텝당 7회)과 그에 따른 코어 이동을 없앤다
torch.autograd.set_multithreading_enabled(False)

from .config import CollocCfg, DataCfg, LoadConfig, NetCfg, TrainCfg
from .Trainer import PINNTrainer
from .Scheduler import OdeScheduler

__all__ = ["NetCfg", "TrainCfg", "CollocCfg", "DataCfg", "LoadConfig", "PINNTrainer", "OdeScheduler"]
    