# Joint corpus for flip + nonflip training from scratch (2026-10-19, branch time-warp) // flip+nonflip 처음부터 공동 학습용 코퍼스
#   corpus = nonflip 70k corpus (unchanged, same order) + the flip cases of the mixed corpus (mixed index order).
#   labels = nonflip dt_f labels + flip fine labels re-simulated from the float64 states of finetune_fl_states.txt
#     (gen_fine.SimulateChunk, same RK4 step and DT), checked against the mixed corpus at its own save times.
#   skip score = nonflip scores + one value above every nonflip score for all flip cases → label_dense_frac 0.2 gives
#     dense = all 8750 flip + the same top 7000 nonflip as the nonflip-only run (0.10 × 70k).
#   meta = is_flip, val_idx = nonflip [0, 5000) (SetupCaseSplit's val of the nonflip run) + the flip cases of the mixed
#     val, train_idx = the rest. Read by DataCfg.split_meta (loop_common.SetupCaseSplit).
# // nonflip 70k 그대로 + mixed의 flip 케이스. flip 촘촘한 라벨은 float64 states에서 재적분, val은 nonflip 런과 같은 앞 5000 + mixed val의 flip
# Usage: .venv/bin/python state/build_joint.py
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from state import gen_fine  # noqa: E402

BASE_DIR = Path(__file__).parent.parent
NF_CORPUS = BASE_DIR / "data/nonflip_RK4_0_5s_70k.npy"
NF_LABEL = BASE_DIR / "data/nonflip_RK4_0_5s_70k_dt005.npy"
NF_SCORE = BASE_DIR / "data/nonflip_RK4_0_5s_70k_skipscore.npy"
MIXED = BASE_DIR / "data/mixed_RK4_0_5s_25flip.npy"
FL_STATES = BASE_DIR / "data/finetune_fl_states.txt"
N_VAL_NF = 5000          # SetupCaseSplit val of the nonflip run: min(n_val 5000, 70000 − max_cases 65000)
DT_SAVE = 0.005
OUT = BASE_DIR / "data/joint_RK4_0_5s_70k_8750flip"


def MatchFlipStates(mixed, flip_pos):
    # mixed flip case ↔ float64 states line (float32 equality of row 0) // mixed flip 케이스의 float64 초기상태 찾기
    lines = np.loadtxt(FL_STATES, delimiter=",", dtype=np.float64)                  # m1 m2 L1 L2 th1 w1 th2 w2
    keys = np.concatenate([lines[:, 4:8], lines[:, 0:4]], axis=1).astype(np.float32)   # th1 w1 th2 w2 m1 m2 L1 L2
    index = {}
    for i, k in enumerate(map(bytes, keys)):
        index.setdefault(k, i)
    rows = []
    for p in flip_pos:
        k = bytes(np.asarray(mixed[p, 0, 1:9], dtype=np.float32))
        if k not in index:
            raise RuntimeError(f"mixed case {p}: no matching line in {FL_STATES.name}")
        rows.append(lines[index[k]])
    return np.stack(rows)


def SimulateFine(rows):
    save_interval = round(DT_SAVE / gen_fine.DT)
    steps = round(gen_fine.T_MAX / gen_fine.DT)
    gen_fine.Trajectory(np.zeros(4), 1.0, 1.0, 1.0, 1.0, gen_fine.G, gen_fine.DT, 10, 5)   # compile before fork
    chunks = [(rows[i:i + gen_fine.CHUNK], save_interval) for i in range(0, len(rows), gen_fine.CHUNK)]
    fine = np.empty((len(rows), steps // save_interval + 1, 4), dtype=np.float32)
    with multiprocessing.Pool(processes=gen_fine.N_WORKERS) as pool:
        for i, res in enumerate(tqdm(pool.imap(gen_fine.SimulateChunk, chunks), total=len(chunks), desc="Simulating")):
            fine[i * gen_fine.CHUNK:i * gen_fine.CHUNK + len(res)] = res
    return fine


def Main():
    nf = np.load(NF_CORPUS, mmap_mode="r")
    nf_lab = np.load(NF_LABEL, mmap_mode="r")
    nf_score = np.load(NF_SCORE)
    mixed = np.load(MIXED, mmap_mode="r")
    with open(MIXED.with_suffix(".meta.json"), encoding="utf-8") as f:
        meta_mixed = json.load(f)
    is_flip_mixed = np.asarray(meta_mixed["is_flip"], dtype=bool)
    flip_pos = np.nonzero(is_flip_mixed)[0]                                          # mixed index order
    val_mixed = set(int(i) for i in meta_mixed["val_idx"])
    n_nf, n_fl = nf.shape[0], len(flip_pos)
    print(f"[joint] nonflip {nf.shape}  mixed {mixed.shape}  flip {n_fl}  "
          f"flip in mixed val {sum(int(p) in val_mixed for p in flip_pos)}")
    if nf.shape[1:] != mixed.shape[1:] or not np.allclose(nf[0, :, 0], mixed[0, :, 0]):
        raise ValueError("nonflip and mixed corpora have different time grids")

    rows = MatchFlipStates(mixed, flip_pos)
    fine = SimulateFine(rows)
    ratio = round(float(mixed[0, 1, 0]) / DT_SAVE)
    fl_corpus = np.asarray(mixed[flip_pos], dtype=np.float32)
    diff = float(np.abs(fine[:, ::ratio] - fl_corpus[:, :, 1:5]).max())
    print(f"[joint] flip fine labels {fine.shape}  max |fine[::{ratio}] - mixed| = {diff:.3e}")
    if diff > 1e-5:
        raise RuntimeError("flip fine labels do not reproduce the mixed corpus")
    if nf_lab.shape[1:] != fine.shape[1:]:
        raise ValueError(f"label shape {nf_lab.shape} vs {fine.shape}")

    corpus = np.lib.format.open_memmap(f"{OUT}.npy", mode="w+", dtype=np.float32, shape=(n_nf + n_fl, *nf.shape[1:]))
    for s in range(0, n_nf, 10000):
        corpus[s:s + 10000] = nf[s:s + 10000]
    corpus[n_nf:] = fl_corpus
    corpus.flush()
    label = np.lib.format.open_memmap(f"{OUT}_dt005.npy", mode="w+", dtype=np.float32,
                                      shape=(n_nf + n_fl, *nf_lab.shape[1:]))
    for s in range(0, n_nf, 10000):
        label[s:s + 10000] = nf_lab[s:s + 10000]
    label[n_nf:] = fine
    label.flush()
    score = np.concatenate([nf_score, np.full(n_fl, float(nf_score.max()) * 2.0 + 1.0, dtype=np.float32)])
    np.save(f"{OUT}_skipscore.npy", score.astype(np.float32))

    is_flip = np.concatenate([np.zeros(n_nf, dtype=bool), np.ones(n_fl, dtype=bool)])
    fl_val = [n_nf + r for r, p in enumerate(flip_pos) if int(p) in val_mixed]
    val_idx = list(range(N_VAL_NF)) + fl_val
    val_set = set(val_idx)
    train_idx = [i for i in range(n_nf + n_fl) if i not in val_set]
    meta = {
        "mode": "joint",
        "sources": {"nonflip": NF_CORPUS.name, "flip": MIXED.name, "flip_states": FL_STATES.name},
        "n_total": n_nf + n_fl, "n_flip": n_fl, "n_nonflip": n_nf,
        "is_flip": is_flip.tolist(),
        "shape": list(corpus.shape),
        "n_val": len(val_idx), "val_idx": val_idx, "train_idx": train_idx,
        "flip_src_idx": flip_pos.tolist(),
    }
    with open(f"{OUT}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"[joint] corpus {corpus.shape} -> {OUT}.npy")
    print(f"[joint] labels {label.shape} -> {OUT}_dt005.npy   score -> {OUT}_skipscore.npy")
    print(f"[joint] val {len(val_idx)} (flip {len(fl_val)})  train {len(train_idx)} "
          f"(flip {n_fl - len(fl_val)})  -> {OUT}.meta.json")


if __name__ == "__main__":
    Main()
