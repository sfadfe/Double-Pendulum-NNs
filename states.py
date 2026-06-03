from pathlib import Path

import numpy as np

none_filp_theta = (0.1, 0.8)
flip_theta = (0.1, 2.5)

none_filp_omega = (-1.0, 1.0)
flip_omega = (-2.0, 2.0)

m_range = (0.5, 5.0)
L_range = (0.5, 2.0)

G = 9.81
ALPHA_NONFLIP = 0.6
EQUAL_ML = False


def SampleLogUniform(lo, hi, size):
    # Log-scale sampling for m, L // m, L 로그 스케일 샘플링
    return np.exp(np.random.uniform(np.log(lo), np.log(hi), size))


def SampleParams(n, IsFlip):
    # Full IC set [m1, m2, L1, L2, th1, w1, th2, w2]
    if IsFlip:
        theta = np.random.uniform(*flip_theta, (n, 2))
        omega = np.random.uniform(*flip_omega, (n, 2))
    else:
        theta = np.random.uniform(*none_filp_theta, (n, 2))
        omega = np.random.uniform(*none_filp_omega, (n, 2))

    m1 = SampleLogUniform(*m_range, n)
    L1 = SampleLogUniform(*L_range, n)
    if EQUAL_ML:
        m2, L2 = m1.copy(), L1.copy()
    else:
        m2 = SampleLogUniform(*m_range, n)
        L2 = SampleLogUniform(*L_range, n)

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
    v_min = -G * ((m1 + m2) * L1 + m2 * L2)  # both bobs hanging down // 둘 다 아래로 정지
    e_rel = TotalEnergy(p) - v_min
    # Lowest barrier: invert outer bob vs inner bob // 최소 반전 장벽 (바깥/안쪽 중 작은 쪽)
    e_flip = np.minimum(2.0 * G * m2 * L2, 2.0 * G * L1 * (m1 + m2))
    return e_rel, e_flip


def GenerateStates(n, filename, IsFlip):
    collected = []
    count = 0
    while count < n:
        batch = SampleParams(n, IsFlip)
        e_rel, e_flip = RelEnergyAndBarrier(batch)
        if IsFlip:
            mask = e_rel > e_flip
        else:
            mask = e_rel < ALPHA_NONFLIP * e_flip
        collected.append(batch[mask])
        count += len(batch[mask])

    params = np.concatenate(collected, axis=0)[:n]

    with open(filename, "w", encoding="utf-8") as f:
        for r in params:
            f.write(", ".join(f"{v}" for v in r) + "\n")


n = 35000
IsFlip = False

BASE_DIR = Path(__file__).parent

if IsFlip:
    path = BASE_DIR / "data" / "hardstates.txt"
else:
    path = BASE_DIR / "data" / "states.txt"

GenerateStates(n, path, IsFlip)
