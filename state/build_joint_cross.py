# Joint corpus with more cross windows (2026-10-21, gate MoE: per-window route balance)
# // cross 창 증량 joint 코퍼스 — 게이트 MoE의 창 단위 경로 비율 보정
#   corpus = the flipx corpus (build_joint_flipx.py, 140000 cases, same order) + n_new new flip cases appended.
#   New flip ICs: states.SampleParamsUnified + IcFlipMask (same IC distribution as every flip case), seeded, kept only if
#     they pass the flipx filters (energy drift, flipped) and their corpus path has ≥ k_min cross windows of the 20
#     march windows (window j = corpus rows 25j..25j+25, a θ1 or θ2 passing an odd multiple of π — RouteLabels' rule).
#   flipx flip cases: 2.32 cross windows of 20 on average, 16.8 % have ≥ 4 (mean 4.65 among them) — the selection
#     keeps the IC distribution and doubles the cross windows (161k → ~300k with n_new 30000).
#   labels / skip score / meta extended as build_joint_flipx.py, val_idx unchanged.
# // flipx 그대로 + cross 창 k_min개 이상인 새 flip을 뒤에 붙임. IC 분포 동일, val 불변
# Usage: .venv/bin/python state/build_joint_cross.py [n_new] [k_min] [seed] [workers]
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from state import gen_fine, states  # noqa: E402
from state.build_joint_flipx import CHUNK, COARSE, CorpusRows, SimulateChunk  # noqa: E402

BASE_DIR = Path(__file__).parent.parent
SRC = BASE_DIR / "data/joint_RK4_0_5s_70k_70000flip"
WIN = 25                    # corpus rows per march window (0.25 s / 0.01 s) // 창당 코퍼스 행


def CrossWindows(fine):
    # fine (n, 1001, 4) → (n,) cross windows of the 20 march windows on the corpus-rate path // 창별 꼭대기 통과 수
    th = fine[:, ::COARSE, 0::2].astype(np.float64)
    kk = np.floor((th + np.pi) / (2.0 * np.pi))
    n_win = (th.shape[1] - 1) // WIN
    w = np.stack([kk[:, WIN * j:WIN * j + WIN + 1] for j in range(n_win)], 1)
    return (w.max(2) != w.min(2)).any(-1).sum(1)


def NewCross(n_new, k_min, pool):
    # Draw flip ICs in rounds until n_new pass the filters and have ≥ k_min cross windows // 조건 통과 n_new개까지 추첨
    rows_out, fine_out, trig_out, n_sim, rate = [], [], [], 0, 0.09
    while sum(len(r) for r in rows_out) < n_new:
        need = n_new - sum(len(r) for r in rows_out)
        cand = []
        while sum(len(c) for c in cand) < int(need / rate * 1.05) + CHUNK:
            b = states.SampleParamsUnified(20000)
            cand.append(b[states.IcFlipMask(b)])
        cand = np.concatenate(cand)[:int(need / rate * 1.05) + CHUNK]
        chunks = [cand[i:i + CHUNK] for i in range(0, len(cand), CHUNK)]
        got = 0
        for c, (fine, trig, keep) in zip(chunks, tqdm(pool.imap(SimulateChunk, chunks), total=len(chunks),
                                                      desc=f"Simulating (need {need})")):
            keep = keep & (CrossWindows(fine) >= k_min)
            rows_out.append(c[keep])
            fine_out.append(fine[keep])
            trig_out.append(trig[keep])
            got += int(keep.sum())
        n_sim += len(cand)
        rate = max(got / len(cand), 0.01)
        print(f"[cross] round: {len(cand)} ICs simulated, {got} kept ({rate:.4f})")
    rows, fine, trig = (np.concatenate(x)[:n_new] for x in (rows_out, fine_out, trig_out))
    print(f"[cross] {n_new} new flip cases (≥ {k_min} cross windows) from {n_sim} ICs")
    return rows, fine, trig


def Main():
    n_new = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    k_min = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20261021
    n_work = int(sys.argv[4]) if len(sys.argv) > 4 else 28
    old = np.load(f"{SRC}.npy", mmap_mode="r")
    old_lab = np.load(f"{SRC}_dt005.npy", mmap_mode="r")
    old_score = np.load(f"{SRC}_skipscore.npy")
    with open(f"{SRC}.meta.json", encoding="utf-8") as f:
        meta_old = json.load(f)
    is_flip_old = np.asarray(meta_old["is_flip"], dtype=bool)
    n_old = old.shape[0]
    out = BASE_DIR / f"data/{SRC.name}_cross{n_new}k{k_min}"
    print(f"[cross] source {old.shape}  flip {int(is_flip_old.sum())}  new {n_new} (≥ {k_min} cross windows)  "
          f"seed {seed}  -> {out.name}")

    np.random.seed(seed)
    gen_fine.Trajectory(np.zeros(4), 1.0, 1.0, 1.0, 1.0, gen_fine.G, gen_fine.DT, 10, 5)   # compile before fork
    st_path = Path(f"{out}_new_states.txt")
    with multiprocessing.Pool(processes=n_work) as pool:
        if st_path.exists():   # kept ICs of an earlier run (float64 repr) → simulate only those // 저장된 통과 IC만 재적분
            rows = np.loadtxt(st_path, delimiter=",", dtype=np.float64)
            if len(rows) != n_new:
                raise ValueError(f"{st_path.name}: {len(rows)} lines, expected {n_new}")
            chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
            res = list(tqdm(pool.imap(SimulateChunk, chunks), total=len(chunks), desc="Simulating (saved ICs)"))
            fine, trig, keep = (np.concatenate([r[j] for r in res]) for j in range(3))
            if not (keep & (CrossWindows(fine) >= k_min)).all():
                raise RuntimeError("saved ICs fail the filters on re-simulation")
            print(f"[cross] {n_new} saved flip ICs re-simulated from {st_path.name}")
        else:
            rows, fine, trig = NewCross(n_new, k_min, pool)
            states.WriteStates(rows, st_path)
    corp = CorpusRows(rows, fine, trig, np.asarray(old[0, :, 0]))
    n_cr = CrossWindows(fine)
    print(f"[cross] new cases: cross windows mean {n_cr.mean():.3f} / 20, hist {np.bincount(n_cr).tolist()}")

    corpus = np.lib.format.open_memmap(f"{out}.npy", mode="w+", dtype=np.float32, shape=(n_old + n_new, *old.shape[1:]))
    for s in range(0, n_old, 10000):
        corpus[s:min(s + 10000, n_old)] = old[s:s + 10000]
    corpus[n_old:] = corp
    corpus.flush()
    label = np.lib.format.open_memmap(f"{out}_dt005.npy", mode="w+", dtype=np.float32,
                                      shape=(n_old + n_new, *old_lab.shape[1:]))
    for s in range(0, n_old, 10000):
        label[s:min(s + 10000, n_old)] = old_lab[s:s + 10000]
    label[n_old:] = fine
    label.flush()
    fl_score = float(old_score[is_flip_old][0])
    np.save(f"{out}_skipscore.npy", np.concatenate([old_score, np.full(n_new, fl_score, dtype=np.float32)]))

    is_flip = np.concatenate([is_flip_old, np.ones(n_new, dtype=bool)])
    new_idx = list(range(n_old, n_old + n_new))
    meta = dict(meta_old)
    meta.update({
        "sources": {**meta_old["sources"], "cross_base": SRC.name, "cross_new_states": f"{out.name}_new_states.txt",
                    "cross_new_seed": seed, "cross_k_min": k_min},
        "n_total": n_old + n_new, "n_flip": int(is_flip.sum()),
        "is_flip": is_flip.tolist(),
        "shape": list(corpus.shape),
        "train_idx": meta_old["train_idx"] + new_idx,
    })
    with open(f"{out}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"[cross] corpus {corpus.shape} -> {out}.npy")
    print(f"[cross] labels {label.shape} -> {out}_dt005.npy   score -> {out}_skipscore.npy (flip value {fl_score:.3e})")
    print(f"[cross] val {meta['n_val']} (flip {int(is_flip[meta['val_idx']].sum())})  train {len(meta['train_idx'])} "
          f"(flip {int(is_flip[meta['train_idx']].sum())})  -> {out}.meta.json")


if __name__ == "__main__":
    Main()
