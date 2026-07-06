import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent

# Unified IC sampling ranges // 에너지 분류로 flip/nonflip 구분
THETA_RANGE = (0.1, 2.5)
OMEGA_RANGE = (-2.0, 2.0)
M_RANGE = (0.5, 5.0)
L_RANGE = (0.5, 2.0)
G = 9.81
ALPHA_NONFLIP = 0.6
EQUAL_ML = False

N_TOTAL = 35000
FLIP_FRAC = 0.25
OVERSAMPLE_NF = 1.5
OVERSAMPLE_FL = 3.0

PATHS = {
    "nonflip": BASE_DIR / "data/states.txt",
    "mixed_nf": BASE_DIR / "data/finetune_nf_states.txt",
    "mixed_fl": BASE_DIR / "data/finetune_fl_states.txt",
}


def SampleLogUniform(lo, hi, size):
    return np.exp(np.random.uniform(np.log(lo), np.log(hi), size))


def SampleParamsUnified(n):
    theta = np.random.uniform(*THETA_RANGE, (n, 2))
    omega = np.random.uniform(*OMEGA_RANGE, (n, 2))
    m1 = SampleLogUniform(*M_RANGE, n)
    L1 = SampleLogUniform(*L_RANGE, n)
    if EQUAL_ML:
        m2, L2 = m1.copy(), L1.copy()
    else:
        m2 = SampleLogUniform(*M_RANGE, n)
        L2 = SampleLogUniform(*L_RANGE, n)
    return np.stack(
        [m1, m2, L1, L2, theta[:, 0], omega[:, 0], theta[:, 1], omega[:, 1]], axis=1
    )


def TotalEnergy(p):
    m1, m2, L1, L2, th1, w1, th2, w2 = p.T
    v1_sq = (L1 * w1) ** 2
    v2_sq = v1_sq + (L2 * w2) ** 2 + 2.0 * L1 * L2 * w1 * w2 * np.cos(th1 - th2)
    K = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
    y1 = -L1 * np.cos(th1)
    y2 = y1 - L2 * np.cos(th2)
    V = m1 * G * y1 + m2 * G * y2
    return K + V


def RelEnergyAndBarrier(p):
    m1, m2, L1, L2 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    v_min = -G * ((m1 + m2) * L1 + m2 * L2)
    e_rel = TotalEnergy(p) - v_min
    e_flip = np.minimum(2.0 * G * m2 * L2, 2.0 * G * L1 * (m1 + m2))
    return e_rel, e_flip


def IcFlipMask(batch):
    e_rel, e_flip = RelEnergyAndBarrier(batch)
    return e_rel > e_flip


def IcNonflipMask(batch):
    e_rel, e_flip = RelEnergyAndBarrier(batch)
    return e_rel < ALPHA_NONFLIP * e_flip


def WriteStates(params, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in params:
            f.write(", ".join(f"{v}" for v in row) + "\n")
    return len(params)


def GenerateIcPool(n_flip, n_nonflip, flip_path=None, nonflip_path=None):
    flip_rows = []
    nf_rows = []
    while len(flip_rows) < n_flip or len(nf_rows) < n_nonflip:
        need = (n_flip - len(flip_rows)) + (n_nonflip - len(nf_rows))
        batch = SampleParamsUnified(max(512, need * 2))
        flip_mask = IcFlipMask(batch)
        nf_mask = IcNonflipMask(batch)
        for i in range(batch.shape[0]):
            if len(flip_rows) < n_flip and flip_mask[i]:
                flip_rows.append(batch[i])
            elif len(nf_rows) < n_nonflip and nf_mask[i]:
                nf_rows.append(batch[i])

    n_f = n_g = 0
    if n_flip > 0 and flip_path is not None:
        n_f = WriteStates(np.array(flip_rows[:n_flip]), flip_path)
    if n_nonflip > 0 and nonflip_path is not None:
        n_g = WriteStates(np.array(nf_rows[:n_nonflip]), nonflip_path)
    print(f"[states] flip={n_f}  nonflip={n_g}")
    return n_f, n_g


def BuildNonflip():
    n_gen = max(int(N_TOTAL * OVERSAMPLE_NF), N_TOTAL + 50)
    print(f"[states] mode=nonflip  n_total={N_TOTAL}  ic_lines={n_gen}")
    GenerateIcPool(0, n_gen, nonflip_path=PATHS["nonflip"])


def BuildMixed():
    n_flip = int(round(FLIP_FRAC * N_TOTAL))
    n_nonflip = N_TOTAL - n_flip
    nf_gen = max(int(n_nonflip * OVERSAMPLE_NF), n_nonflip + 50)
    fl_gen = max(int(n_flip * OVERSAMPLE_FL), n_flip + 100)
    print(
        f"[states] mode=mixed  n_total={N_TOTAL}  flip={n_flip}  "
        f"nonflip={n_nonflip}  ic_nf={nf_gen}  ic_flip={fl_gen}"
    )
    GenerateIcPool(
        fl_gen, nf_gen,
        flip_path=PATHS["mixed_fl"],
        nonflip_path=PATHS["mixed_nf"],
    )


def Main():
    parser = argparse.ArgumentParser(
        description="Generate IC files: nonflip (100%%) or mixed (25%% flip)"
    )
    parser.add_argument(
        "mode", choices=["nonflip", "mixed"],
        help="nonflip=pretrain, mixed=finetune",
    )
    args = parser.parse_args()
    if args.mode == "nonflip":
        BuildNonflip()
    else:
        BuildMixed()


if __name__ == "__main__":
    Main()
