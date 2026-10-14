import argparse
import multiprocessing
import sys
from pathlib import Path

import numpy as np
from numba import jit
from scipy.interpolate import CubicSpline
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import state.Double_pendulum as Dp

BASE_DIR = Path(__file__).parent.parent

G = 9.81
DT = 5e-06
T_MAX = 5
N_WORKERS = 28
CHUNK = 100


@jit(nopython=True, cache=True)
def Trajectory(state0, m1, m2, L1, L2, g, dt, steps, save_interval):
    # Same RK4 step as state/dfss.py, loop kept inside numba // dfss.py와 같은 RK4 스텝, 루프만 numba 안으로
    out = np.empty((steps // save_interval + 1, 4))
    state = state0.copy()
    out[0] = state
    k = 1
    for i in range(1, steps + 1):
        state = Dp.RK4(state, dt, m1, m2, L1, L2, g)
        if i % save_interval == 0:
            out[k] = state
            k += 1
    return out


def SimulateChunk(args):
    rows, save_interval = args
    steps = round(T_MAX / DT)
    out = np.empty((len(rows), steps // save_interval + 1, 4), dtype=np.float32)
    for n, r in enumerate(rows):
        m1, m2, L1, L2 = r[0], r[1], r[2], r[3]
        out[n] = Trajectory(np.array(r[4:8], dtype=np.float64), m1, m2, L1, L2, G, DT, steps, save_interval)
    return out


def MatchStates(corpus, states_paths, n_each):
    # corpus case k ↔ states line k (float32 equality on IC and params) // 코퍼스 case와 states 줄의 대응 확인
    rows = []
    for part, p in enumerate(states_paths):
        lines = np.loadtxt(p, delimiter=",", dtype=np.float64)[:n_each]     # m1 m2 L1 L2 th1 w1 th2 w2
        ref = np.asarray(corpus[part * n_each:(part + 1) * n_each, 0, 1:9])  # th1 w1 th2 w2 m1 m2 L1 L2
        got = np.concatenate([lines[:, 4:8], lines[:, 0:4]], axis=1).astype(np.float32)
        bad = int((got != ref).any(axis=1).sum())
        if bad:
            raise RuntimeError(f"{p}: {bad} of {n_each} cases do not match the corpus row 0")
        rows.append(lines)
    return np.concatenate(rows, axis=0)


def BuildFine(corpus_path, states_paths, out_path, dt_save, limit=None):
    corpus = np.load(corpus_path, mmap_mode="r")
    n_case = corpus.shape[0]
    rows = MatchStates(corpus, states_paths, n_case // len(states_paths))
    if limit:
        rows = rows[:limit]
    save_interval = round(dt_save / DT)
    steps = round(T_MAX / DT)
    Trajectory(np.zeros(4), 1.0, 1.0, 1.0, 1.0, G, DT, 10, 5)               # compile before fork // fork 전 컴파일
    chunks = [(rows[i:i + CHUNK], save_interval) for i in range(0, len(rows), CHUNK)]
    fine = np.empty((len(rows), steps // save_interval + 1, 4), dtype=np.float32)
    print(f"[gen_fine] cases={len(rows)}  dt_save={dt_save}  workers={N_WORKERS}")
    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        for i, res in enumerate(tqdm(pool.imap(SimulateChunk, chunks), total=len(chunks), desc="Simulating")):
            fine[i * CHUNK:i * CHUNK + len(res)] = res

    # The coarse corpus must be reproduced at its own save times // 기존 코퍼스 저장 시점에서 일치해야 한다
    ratio = round(float(corpus[0, 1, 0]) / dt_save)
    diff = np.abs(fine[:, ::ratio] - np.asarray(corpus[:len(rows), :, 1:5])).max()
    print(f"[gen_fine] max |fine[::{ratio}] - corpus| = {diff:.3e}")
    if diff > 1e-5:
        raise RuntimeError("fine trajectories do not reproduce the corpus")
    np.save(out_path, fine)
    print(f"[gen_fine] {fine.shape} -> {out_path}")


def BuildSkipScore(corpus_path, out_path, skip=2, batch=2000):
    # Per-case error of rebuilding the dropped labels from every skip-th one (cubic spline, std units)
    # // skip칸마다 남긴 라벨로 빠진 라벨을 복원한 case별 오차(3차 스플라인, std 단위)
    corpus = np.load(corpus_path, mmap_mode="r")
    n_case, n_step, _ = corpus.shape
    t = np.asarray(corpus[0, :, 0], dtype=np.float64)
    keep = np.arange(0, n_step, skip)
    miss = np.setdiff1d(np.arange(n_step), keep)
    sd = np.asarray(corpus[::20, :, 1:5], dtype=np.float64).reshape(-1, 4).std(0)
    score = np.empty(n_case, dtype=np.float32)
    for i in tqdm(range(0, n_case, batch), desc="SkipScore"):
        x = np.asarray(corpus[i:i + batch, :, 1:5], dtype=np.float64)
        e = (CubicSpline(t[keep], x[:, keep], axis=1)(t[miss]) - x[:, miss]) / sd
        score[i:i + batch] = np.sqrt((e ** 2).mean((1, 2)))
    np.save(out_path, score)
    q = np.quantile(score, [0.5, 0.9, 0.95, 0.99])
    print(f"[gen_fine] skip score p50={q[0]:.2e} p90={q[1]:.2e} p95={q[2]:.2e} p99={q[3]:.2e} -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fine-step labels and label-skip score for the nonflip corpus")
    parser.add_argument("mode", choices=["fine", "score"])
    parser.add_argument("--corpus", default=str(BASE_DIR / "data/nonflip_RK4_0_5s_70k.npy"))
    parser.add_argument("--states", nargs="+", default=[str(BASE_DIR / "data/states.txt"),
                                                        str(BASE_DIR / "data/states_b.txt")])
    parser.add_argument("--out", default=None)
    parser.add_argument("--dt_save", type=float, default=0.005)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "fine":
        BuildFine(args.corpus, args.states, args.out or str(BASE_DIR / "data/nonflip_RK4_0_5s_70k_dt005.npy"),
                  args.dt_save, args.limit)
    else:
        BuildSkipScore(args.corpus, args.out or str(BASE_DIR / "data/nonflip_RK4_0_5s_70k_skipscore.npy"))
