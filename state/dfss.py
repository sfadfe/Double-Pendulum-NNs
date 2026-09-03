import argparse
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import state.Double_pendulum as Dp

BASE_DIR = Path(__file__).parent.parent

G = 9.81
DT = 5e-06
T_MAX = 5
SAVE_INTERVAL = 2000
ENERGY_TOL = 1e-4
FLIP_LIMIT = np.pi
N_WORKERS = 21

N_TOTAL = 35000
FLIP_FRAC = 0.25
N_VAL = 2000

PATHS = {
    "nonflip": {
        "states": BASE_DIR / "data/states.txt",
        "out": BASE_DIR / "data/nonflip_RK4_0_5s.npy",
    },
    "mixed": {
        "nf_states": BASE_DIR / "data/finetune_nf_states.txt",
        "fl_states": BASE_DIR / "data/finetune_fl_states.txt",
        "nf_staging": BASE_DIR / "data/_build_nf_staging.npy",
        "fl_staging": BASE_DIR / "data/_build_fl_staging.npy",
        "out": BASE_DIR / "data/mixed_RK4_0_5s_25flip.npy",
    },
}


def Energy(state, m1, m2, L1, L2):
    th1, w1, th2, w2 = state
    v1_sq = (L1 * w1) ** 2
    v2_sq = v1_sq + (L2 * w2) ** 2 + 2.0 * L1 * L2 * w1 * w2 * np.cos(th1 - th2)
    K = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
    y1 = -L1 * np.cos(th1)
    y2 = y1 - L2 * np.cos(th2)
    V = m1 * G * y1 + m2 * G * y2
    return K + V


def GetDataRow(state, m1, m2, L1, L2, current_time):
    s = state.tolist()
    theta1, theta2 = s[0], s[2]
    return (
        [current_time]
        + s
        + [m1, m2, L1, L2]
        + [np.sin(theta1), np.cos(theta1), np.sin(theta2), np.cos(theta2)]
    )


def SimulateSingleTrajectory(args):
    line, want_flip = args
    if not line.strip():
        return None

    parts = line.replace(",", " ").split()
    m1, m2, L1, L2, th1, w1, th2, w2 = map(float, parts)
    initial_state = [th1, w1, th2, w2]

    dp = Dp.Double_pendulum(m1, m2, L1=L1, L2=L2, initial_state=initial_state, g=G)
    e0 = Energy(dp.state, m1, m2, L1, L2)
    trajectory = [GetDataRow(dp.state, m1, m2, L1, L2, 0.0)]

    steps = round(T_MAX / DT)
    flipped = False
    for i in range(1, steps + 1):
        dp.RK4(DT)
        if i % SAVE_INTERVAL == 0:
            e = Energy(dp.state, m1, m2, L1, L2)
            if abs(e - e0) / abs(e0) > ENERGY_TOL:
                return None
            if abs(dp.state[0]) > FLIP_LIMIT or abs(dp.state[2]) > FLIP_LIMIT:
                flipped = True
            trajectory.append(GetDataRow(dp.state, m1, m2, L1, L2, i * DT))

    if want_flip and not flipped:
        return None
    if not want_flip and flipped:
        return None

    return np.array(trajectory, dtype=np.float32)


def RunSimulation(IsFlip, input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    worker_args = [(line, IsFlip) for line in lines]
    print(f"[dfss] cores={N_WORKERS}  IsFlip={IsFlip}  input={len(lines)} lines")

    results = []
    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        for res in tqdm(
            pool.imap(SimulateSingleTrajectory, worker_args),
            total=len(lines),
            desc="Simulating",
        ):
            if res is not None:
                results.append(res)

    if not results:
        raise RuntimeError(f"[dfss] no trajectories passed filters: {input_path}")

    print("[dfss] stacking...")
    final_data = np.array(results, dtype=np.float32)
    np.save(output_path, final_data)
    print(f"[dfss] {final_data.shape} -> {output_path}")
    return final_data.shape


def StratifiedValSplitIdx(is_flip, n_val):
    rng = np.random.default_rng()
    n_total = is_flip.shape[0]
    n_val = min(int(n_val), n_total - 1)
    flip_pos = np.where(is_flip)[0]
    nf_pos = np.where(~is_flip)[0]
    rng.shuffle(flip_pos)
    rng.shuffle(nf_pos)
    flip_frac = float(is_flip.mean())
    n_vf = min(len(flip_pos), max(1, int(round(flip_frac * n_val))))
    n_vnf = min(len(nf_pos), n_val - n_vf)
    n_vf = min(len(flip_pos), n_val - n_vnf)
    val_idx = np.concatenate([flip_pos[:n_vf], nf_pos[:n_vnf]])
    rng.shuffle(val_idx)
    mask = np.ones(n_total, dtype=bool)
    mask[val_idx] = False
    return val_idx, np.where(mask)[0]


def BuildNonflip(n_total=N_TOTAL, states_path=None, out_path=None):
    # Overrides for extra corpora (case-count lever) // 케이스 증량용 추가 코퍼스 — 기본값은 종전 동작
    paths = PATHS["nonflip"]
    states_path = paths["states"] if states_path is None else Path(states_path)
    out_path = paths["out"] if out_path is None else Path(out_path)
    print(f"[dfss] mode=nonflip  n_total={n_total}  states={states_path}  out={out_path}")
    RunSimulation(False, states_path, out_path)

    data = np.load(out_path, mmap_mode="r")
    if data.shape[0] < n_total:
        raise RuntimeError(
            f"nonflip: got {data.shape[0]} trajectories, need {n_total} — "
            f"re-run state/states.py nonflip"
        )
    trimmed = np.array(data[:n_total], dtype=np.float32)
    np.save(out_path, trimmed)
    print(f"[dfss] saved {trimmed.shape} -> {out_path}")
    return trimmed.shape


def BuildMixed():
    rng = np.random.default_rng()
    paths = PATHS["mixed"]
    n_flip = int(round(FLIP_FRAC * N_TOTAL))
    n_nonflip = N_TOTAL - n_flip

    print(
        f"[dfss] mode=mixed  n_total={N_TOTAL}  flip={n_flip}  "
        f"nonflip={n_nonflip}  n_val={N_VAL}"
    )

    RunSimulation(False, paths["nf_states"], paths["nf_staging"])
    RunSimulation(True, paths["fl_states"], paths["fl_staging"])

    nf_data = np.load(paths["nf_staging"], mmap_mode="r")
    fl_data = np.load(paths["fl_staging"], mmap_mode="r")
    if nf_data.shape[0] < n_nonflip or fl_data.shape[0] < n_flip:
        raise RuntimeError(
            f"mixed: nf={nf_data.shape[0]}/{n_nonflip}  "
            f"flip={fl_data.shape[0]}/{n_flip} — re-run state/states.py mixed"
        )

    nf_pick = rng.choice(nf_data.shape[0], size=n_nonflip, replace=False)
    fl_pick = rng.choice(fl_data.shape[0], size=n_flip, replace=False)
    nf_rows = np.array(nf_data[nf_pick], dtype=np.float32)
    fl_rows = np.array(fl_data[fl_pick], dtype=np.float32)

    corpus = np.concatenate([nf_rows, fl_rows], axis=0)
    is_flip = np.array([False] * n_nonflip + [True] * n_flip, dtype=bool)
    perm = rng.permutation(N_TOTAL)
    corpus = corpus[perm]
    is_flip = is_flip[perm]

    val_idx, train_idx = StratifiedValSplitIdx(is_flip, N_VAL)
    replay_idx = [int(i) for i in train_idx if not is_flip[i]]

    np.save(paths["out"], corpus)
    meta_path = paths["out"].with_suffix(".meta.json")
    meta = {
        "mode": "mixed",
        "flip_frac": FLIP_FRAC,
        "n_total": N_TOTAL,
        "n_flip": n_flip,
        "n_nonflip": n_nonflip,
        "is_flip": is_flip.tolist(),
        "shape": list(corpus.shape),
        "n_val": int(len(val_idx)),
        "val_idx": val_idx.tolist(),
        "train_idx": train_idx.tolist(),
        "replay_idx": replay_idx,
        "n_replay": len(replay_idx),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[dfss] saved {corpus.shape} -> {paths['out']}")
    print(f"[dfss] meta -> {meta_path}")
    print(
        f"[dfss] val={len(val_idx)} (flip {int(is_flip[val_idx].sum())})  "
        f"replay={len(replay_idx)}"
    )
    return corpus.shape


def Main():
    parser = argparse.ArgumentParser(
        description="RK4 build from IC files: nonflip or mixed"
    )
    parser.add_argument(
        "mode", choices=["nonflip", "mixed"],
        help="nonflip=pretrain, mixed=finetune",
    )
    parser.add_argument("--n_total", type=int, default=N_TOTAL, help="nonflip only")
    parser.add_argument("--states", type=str, default=None, help="nonflip only: IC file")
    parser.add_argument("--out", type=str, default=None, help="nonflip only: output .npy")
    args = parser.parse_args()
    if args.mode == "nonflip":
        BuildNonflip(n_total=args.n_total, states_path=args.states, out_path=args.out)
    else:
        BuildMixed()


if __name__ == "__main__":
    Main()
