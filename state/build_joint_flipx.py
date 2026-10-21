# Joint corpus with more flip cases (2026-10-20, route3: the over-the-top route starved of windows)
# // flip 케이스 증량 joint 코퍼스 — route3의 OTT 경로 데이터 부족
#   corpus = the build_joint.py corpus (78750 cases, same order) + n_new new flip cases appended.
#   New flip ICs: states.SampleParamsUnified + IcFlipMask (same IC distribution as finetune_fl_states.txt), seeded.
#     One fine simulation per IC (gen_fine.Trajectory, DT 5e-6, saved every 0.005 s); kept only if it passes the dfss.py
#     filters at the corpus save points (every 0.01 s): relative energy drift ≤ ENERGY_TOL and |θ1| or |θ2| > π
#     (flipped). Corpus rows = the fine trajectory at every other save (= dfss.py rows; build_joint checked
#     fine[::2] == corpus at 0 diff) with dfss's 13 columns.
#   labels = old labels + the new fine trajectories; skip score = old + the old flip value (all flip stay dense).
#   meta = is_flip extended, val_idx unchanged (same val as every joint run), train_idx + the new cases.
#   New IC lines (float64 repr) are written to <out>_new_states.txt.
# // 기존 joint 그대로 + 새 flip을 뒤에 붙임. IC 분포는 기존 flip과 동일, fine 1회 적분 후 dfss 필터. val 불변
# Usage: .venv/bin/python state/build_joint_flipx.py [n_new] [seed]
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from state import gen_fine, states  # noqa: E402

BASE_DIR = Path(__file__).parent.parent
SRC = BASE_DIR / "data/joint_RK4_0_5s_70k_8750flip"
DT_SAVE = 0.005
COARSE = 2                  # corpus save 0.01 s = every 2nd fine save // 코퍼스 저장 간격 = fine 2칸
ENERGY_TOL = 1e-4           # dfss.py
FLIP_LIMIT = np.pi          # dfss.py
N_WORKERS = 30
CHUNK = 100


def Energy(x, m1, m2, L1, L2):
    # dfss.Energy over rows x (n, 4) [θ1, ω1, θ2, ω2] // dfss와 같은 에너지
    th1, w1, th2, w2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    v1_sq = (L1 * w1) ** 2
    v2_sq = v1_sq + (L2 * w2) ** 2 + 2.0 * L1 * L2 * w1 * w2 * np.cos(th1 - th2)
    k = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
    y1 = -L1 * np.cos(th1)
    y2 = y1 - L2 * np.cos(th2)
    return k + m1 * gen_fine.G * y1 + m2 * gen_fine.G * y2


def SimulateChunk(rows):
    # rows (n, 8) m1 m2 L1 L2 th1 w1 th2 w2 → fine (n, 1001, 4) float32, trig (n, 501, 4) float32 = sin/cos of the
    #   float64 θ at the corpus saves (as dfss), keep (n,) bool // fine 적분 + 코퍼스 sin/cos + dfss 필터
    save_interval = round(DT_SAVE / gen_fine.DT)
    steps = round(gen_fine.T_MAX / gen_fine.DT)
    n_f = steps // save_interval + 1
    fine = np.empty((len(rows), n_f, 4), dtype=np.float32)
    trig = np.empty((len(rows), (n_f - 1) // COARSE + 1, 4), dtype=np.float32)
    keep = np.zeros(len(rows), dtype=bool)
    for n, r in enumerate(rows):
        m1, m2, L1, L2 = r[0], r[1], r[2], r[3]
        tr = gen_fine.Trajectory(np.array(r[4:8], dtype=np.float64), m1, m2, L1, L2, gen_fine.G, gen_fine.DT, steps,
                                 save_interval)
        e0 = Energy(tr[:1], m1, m2, L1, L2)[0]
        c = tr[COARSE::COARSE]
        drift_ok = bool((np.abs(Energy(c, m1, m2, L1, L2) - e0) / abs(e0) <= ENERGY_TOL).all())
        flipped = bool(((np.abs(c[:, 0]) > FLIP_LIMIT) | (np.abs(c[:, 2]) > FLIP_LIMIT)).any())
        keep[n] = drift_ok and flipped
        fine[n] = tr
        th = tr[::COARSE, 0::2]
        trig[n] = np.stack([np.sin(th[:, 0]), np.cos(th[:, 0]), np.sin(th[:, 1]), np.cos(th[:, 1])], axis=1)
    return fine, trig, keep


def NewFlip(n_new, pool):
    # Draw flip ICs in rounds until n_new pass the filters, IC order kept // 필터 통과 n_new개가 될 때까지 추첨
    rows_out, fine_out, trig_out, n_sim, rate = [], [], [], 0, 0.55
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
            rows_out.append(c[keep])
            fine_out.append(fine[keep])
            trig_out.append(trig[keep])
            got += int(keep.sum())
        n_sim += len(cand)
        rate = max(got / len(cand), 0.05)
        print(f"[flipx] round: {len(cand)} ICs simulated, {got} kept ({rate:.3f})")
    rows, fine, trig = (np.concatenate(x)[:n_new] for x in (rows_out, fine_out, trig_out))
    print(f"[flipx] {n_new} new flip cases from {n_sim} ICs")
    return rows, fine, trig


def CorpusRows(rows, fine, trig, t):
    # dfss.GetDataRow columns: t, θ1 ω1 θ2 ω2, m1 m2 L1 L2, sin θ1 cos θ1 sin θ2 cos θ2 // dfss 13열
    out = np.empty((len(rows), len(t), 13), dtype=np.float32)
    out[:, :, 0] = t
    out[:, :, 1:5] = fine[:, ::COARSE]
    out[:, :, 5:9] = rows[:, None, 0:4].astype(np.float32)
    out[:, :, 9:13] = trig
    return out


def Main():
    n_new = int(sys.argv[1]) if len(sys.argv) > 1 else 26250
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261020
    old = np.load(f"{SRC}.npy", mmap_mode="r")
    old_lab = np.load(f"{SRC}_dt005.npy", mmap_mode="r")
    old_score = np.load(f"{SRC}_skipscore.npy")
    with open(f"{SRC}.meta.json", encoding="utf-8") as f:
        meta_old = json.load(f)
    is_flip_old = np.asarray(meta_old["is_flip"], dtype=bool)
    n_old, n_fl_old = old.shape[0], int(is_flip_old.sum())
    out = BASE_DIR / f"data/joint_RK4_0_5s_70k_{n_fl_old + n_new}flip"
    print(f"[flipx] source {old.shape}  flip {n_fl_old}  new {n_new}  seed {seed}  -> {out.name}")

    np.random.seed(seed)
    gen_fine.Trajectory(np.zeros(4), 1.0, 1.0, 1.0, 1.0, gen_fine.G, gen_fine.DT, 10, 5)   # compile before fork
    st_path = Path(f"{out}_new_states.txt")
    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        if st_path.exists():   # kept ICs of an earlier run (float64 repr) → simulate only those // 저장된 통과 IC만 재적분
            rows = np.loadtxt(st_path, delimiter=",", dtype=np.float64)
            if len(rows) != n_new:
                raise ValueError(f"{st_path.name}: {len(rows)} lines, expected {n_new}")
            chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
            res = list(tqdm(pool.imap(SimulateChunk, chunks), total=len(chunks), desc="Simulating (saved ICs)"))
            fine, trig, keep = (np.concatenate([r[j] for r in res]) for j in range(3))
            if not keep.all():
                raise RuntimeError(f"{int((~keep).sum())} saved ICs fail the filters on re-simulation")
            print(f"[flipx] {n_new} saved flip ICs re-simulated from {st_path.name}")
        else:
            rows, fine, trig = NewFlip(n_new, pool)
            states.WriteStates(rows, st_path)
    corp = CorpusRows(rows, fine, trig, np.asarray(old[0, :, 0]))

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
        "sources": {**meta_old["sources"], "base": SRC.name, "flip_new_states": f"{out.name}_new_states.txt",
                    "flip_new_seed": seed},
        "n_total": n_old + n_new, "n_flip": n_fl_old + n_new,
        "is_flip": is_flip.tolist(),
        "shape": list(corpus.shape),
        "train_idx": meta_old["train_idx"] + new_idx,
    })
    with open(f"{out}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"[flipx] corpus {corpus.shape} -> {out}.npy")
    print(f"[flipx] labels {label.shape} -> {out}_dt005.npy   score -> {out}_skipscore.npy (flip value {fl_score:.3e})")
    print(f"[flipx] val {meta['n_val']} (flip {int(is_flip[meta['val_idx']].sum())})  train {len(meta['train_idx'])} "
          f"(flip {int(is_flip[meta['train_idx']].sum())})  -> {out}.meta.json")


if __name__ == "__main__":
    Main()
