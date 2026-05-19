import multiprocessing
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

import Double_pendulum as Dp

IsFlip = False

G = 9.81
DT = 5e-06            # integration step, float64 (design 2.1) // 적분 스텝
T_MAX = 3             # 0~3s only for now; 0~5 decided after t=3 results // 우선 0~3초만
SAVE_INTERVAL = 2000  # 2000 step = 0.01s record // 0.01초 간격 기록
ENERGY_TOL = 1e-4     # drop if |E(t)-E0|/|E0| exceeds (design 2.2/2.3) // 에너지 보존 필터
FLIP_LIMIT = np.pi    # |theta| over pi within [0,3s] => flipped // 플립 판정 (연속 θ 기준)


def Energy(state, m1, m2, L1, L2):
    # float64 energy, same convention as PINNsTrainer/ODE.py GetEnergy // float64 총에너지
    th1, w1, th2, w2 = state
    v1_sq = (L1 * w1) ** 2
    v2_sq = v1_sq + (L2 * w2) ** 2 + 2.0 * L1 * L2 * w1 * w2 * np.cos(th1 - th2)
    K = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq

    y1 = -L1 * np.cos(th1)
    y2 = y1 - L2 * np.cos(th2)
    V = m1 * G * y1 + m2 * G * y2
    return K + V


def GetDataRow(state, m1, m2, L1, L2, current_time):
    # [t] + [th1,w1,th2,w2] + [m1,m2,L1,L2] + [sin t1,cos t1,sin t2,cos t2] (design 2.4) // 13 채널
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

    # Full IC from states.py: m1,m2,L1,L2,th1,w1,th2,w2 // 전체 초기조건 파싱
    parts = line.replace(",", " ").split()
    m1, m2, L1, L2, th1, w1, th2, w2 = map(float, parts)
    initial_state = [th1, w1, th2, w2]

    dp = Dp.Double_pendulum(m1, m2, L1=L1, L2=L2, initial_state=initial_state, g=G)

    e0 = Energy(dp.state, m1, m2, L1, L2)  # float64, conserved // 기준 총에너지
    trajectory = [GetDataRow(dp.state, m1, m2, L1, L2, 0.0)]

    steps = int(T_MAX / DT)
    flipped = False
    for i in range(1, steps + 1):
        dp.RK4(DT)

        if i % SAVE_INTERVAL == 0:
            # Energy conservation filter in float64 before float32 cast // float32 변환 전 float64 에너지 필터
            e = Energy(dp.state, m1, m2, L1, L2)
            if abs(e - e0) / abs(e0) > ENERGY_TOL:
                return None

            if abs(dp.state[0]) > FLIP_LIMIT or abs(dp.state[2]) > FLIP_LIMIT:
                flipped = True

            trajectory.append(GetDataRow(dp.state, m1, m2, L1, L2, i * DT))

    # Pool consistency // 풀 일관성 확정
    if want_flip and not flipped:
        return None
    if not want_flip and flipped:
        return None

    # float64로 계산 후 float32로 변환 해 저장
    return np.array(trajectory, dtype=np.float32)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    output_dir = BASE_DIR / "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if IsFlip:
        input_path = output_dir / "hardstates.txt"
        output_path = output_dir / "flip_RK4_0_3s.npy"
    else:
        input_path = output_dir / "states.txt"
        output_path = output_dir / "nonflip_RK4_0_3s.npy"

    with open(input_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    worker_args = [(line, IsFlip) for line in lines]

    NumCores = multiprocessing.cpu_count()
    UsingCore = max(1, NumCores - 3)

    print(f"Using {UsingCore} core, IsFlip={IsFlip}, cases={len(lines)}")

    results = []

    with multiprocessing.Pool(processes=UsingCore) as pool:
        for res in tqdm(
            pool.imap(SimulateSingleTrajectory, worker_args),
            total=len(lines),
            desc="Simulating",
        ):
            if res is not None:
                results.append(res)

    print("Stacking data...")
    final_data = np.array(results, dtype=np.float32)  # (N, 301, 13)

    np.save(output_path, final_data)

    print(f"Done... {final_data.shape} -> {output_path.name}")
