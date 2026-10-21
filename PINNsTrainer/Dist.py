import datetime
import os
import signal
import tempfile

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# Data parallel over GPUs, launched by torchrun (Train/train.py). Each rank draws its own 1/W share of every per-step
#   sample (ShardConfig); each loss term's gradient is averaged across ranks before its clip (Trainstep._FlushTerm),
#   so every rank takes the same optimizer step and the weights stay identical without a broadcast (CheckSync).
#   Rank 0 alone evaluates, logs and saves, and broadcasts the values the LR scheduler reads.
#   World 1 (plain python): every helper is a no-op.
# // torchrun 데이터 병렬: rank별 표본 1/W, 항별 grad를 rank 평균한 뒤 clip → 모든 rank가 같은 스텝. 평가·저장은 rank 0

_stop = [False]


def World():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def Rank():
    return dist.get_rank() if World() > 1 else 0


def IsMain():
    return Rank() == 0


def _OnSigint(signum, frame):
    # kill -INT on any rank: every rank stops at the next step (StepVote) — a lone KeyboardInterrupt would leave the
    #   peer blocked in a collective. // 다음 스텝에서 모든 rank가 함께 멈춘다
    _stop[0] = True


def InitDist():
    # torchrun sets WORLD_SIZE / LOCAL_RANK. Before any CUDA work, so device "cuda" is the rank's GPU.
    # // torchrun이면 프로세스 그룹 초기화 — 이후 "cuda" = 이 rank의 GPU
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return 1
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    # 60 min: a rank may wait out the other's first compile or rank 0's evaluation // 컴파일·평가 대기 허용
    dist.init_process_group("nccl", device_id=torch.device("cuda", local), timeout=datetime.timedelta(minutes=60))
    signal.signal(signal.SIGINT, _OnSigint)
    return World()


def Shutdown():
    if World() > 1:
        dist.destroy_process_group()


def SeedRank(seed, start_epoch):
    # Own sample stream per rank, after init / split / eval-case selection (those stay identical across ranks)
    # // rank별 표본 RNG — 초기화·분할·평가 케이스 선택 뒤에
    if World() > 1:
        torch.manual_seed(int(seed) + 1_000_003 * Rank() + int(start_epoch))


def ScalerDir(run_dir):
    # scaler.npy is written by rank 0 only; the others write theirs to a scratch dir // rank 0만 런 폴더에 쓴다
    return run_dir if IsMain() else tempfile.mkdtemp(prefix=f"scaler_rank{Rank()}_")


def ShardConfig(train_cfg, colloc_cfg, data_cfg, t_params):
    # Per-rank share of every per-step sample (summed over ranks = the config's count). Eval sizes stay whole.
    #   colloc_chunk is re-picked to divide the rank's n_colloc (one compiled chunk shape), nearest the configured one.
    # // rank당 per-step 표본 = config 값 / W. 평가 크기는 그대로. colloc_chunk는 rank n_colloc의 약수로 재선택
    w = World()
    if w <= 1:
        return

    def Cut(name, n):
        if n % w:
            raise ValueError(f"{name} {n} must be a multiple of the world size {w}")
        return n // w

    data_cfg.batch_size = Cut("batch_size", data_cfg.batch_size)
    if data_cfg.batch_size_phase2 > 0:
        data_cfg.batch_size_phase2 = Cut("batch_size_phase2", data_cfg.batch_size_phase2)
    colloc_cfg.n_colloc = Cut("n_colloc", colloc_cfg.n_colloc)
    t_params["roll_cases"] = Cut("roll_cases", t_params.get("roll_cases", 256))
    train_cfg.roll_balance_cases = Cut("roll_balance_cases", train_cfg.roll_balance_cases)
    train_cfg.warp_sup_cases = Cut("warp_sup_cases", train_cfg.warp_sup_cases)
    train_cfg.route_sup_cases = Cut("route_sup_cases", train_cfg.route_sup_cases)
    if train_cfg.ic_max_n > 0:
        train_cfg.ic_max_n = Cut("ic_max_n", train_cfg.ic_max_n)
    n, c, g = colloc_cfg.n_colloc, data_cfg.colloc_chunk, max(train_cfg.case_group, 1)
    if c and n % c:
        sizes = [n // k for k in range(1, n + 1) if n % k == 0 and (n // k) % g == 0]
        data_cfg.colloc_chunk = min(sizes, key=lambda s: (abs(s - c), -s))
    print(f"[dist] world {w}: per-rank batch {data_cfg.batch_size}/{data_cfg.batch_size_phase2} "
          f"n_colloc {colloc_cfg.n_colloc} (chunk {data_cfg.colloc_chunk}) roll {t_params['roll_cases']} "
          f"warp_sup {train_cfg.warp_sup_cases} route_sup {train_cfg.route_sup_cases}")


def AllReduceMean_(tensors):
    # In-place mean across ranks, one collective over the flattened list // 평균 all-reduce 1회
    if World() <= 1 or not tensors:
        return
    flat = _flatten_dense_tensors(tensors)
    dist.all_reduce(flat)
    flat.div_(World())
    torch._foreach_copy_(tensors, _unflatten_dense_tensors(flat, tensors))


def AverageGrads(params):
    # Mean .grad of the trainable params across ranks. A missing .grad becomes zeros so every rank packs the same
    #   layout (a route absent from one rank's batch). // 빠진 grad는 0으로 채워 rank 간 배치를 맞춘다
    if World() <= 1:
        return
    ps = [p for p in params if p.requires_grad]
    for p in ps:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    AllReduceMean_([p.grad for p in ps])


def StepVote(metric_t, gnorm_t):
    # (metric, gnorm, stop) of a step in its one host sync: metric averaged over ranks (gnorm is already equal),
    #   NaN on any rank → NaN on all (same skip decision), stop = any rank got SIGINT.
    # // 스텝 판정값을 sync 1회로: NaN·종료 요청이 모든 rank에 같이 걸린다
    v = torch.stack([metric_t.detach().float(), gnorm_t.detach().float(),
                     torch.full((), float(_stop[0]), device=metric_t.device)])
    if World() > 1:
        dist.all_reduce(v)
        v.div_(World())
    m, g, s = v.tolist()
    return m, g, s > 0


def BroadcastFloats(vals, src=0):
    # rank src's floats on every rank (NaN / inf pass through) // rank 0 값 공유
    if World() <= 1:
        return vals
    t = torch.tensor(vals, dtype=torch.float64, device="cuda")
    dist.broadcast(t, src)
    return t.tolist()


def BroadcastObject(obj, src=0):
    if World() <= 1:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src)
    return box[0]


def CheckSync(module, tag):
    # Raise if any parameter differs between ranks (per-tensor fp64 sums, MAX vs MIN) // rank 간 가중치 일치 확인
    if World() <= 1:
        return
    with torch.no_grad():
        s = torch.stack([p.detach().double().sum() for p in module.parameters()])
        hi, lo = s.clone(), s.clone()
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        bad = int((hi != lo).sum())
    if bad:
        raise RuntimeError(f"[dist] {tag}: {bad} parameter tensors differ between ranks")
