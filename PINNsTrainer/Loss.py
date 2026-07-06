import torch
import torch.func as tf

class Loss:
    def StateDerivs(self, feats):
        # net outputs state [θ, ω]; one forward-mode pass gives [dθ/dt, dω/dt]
        # feats: (M, 11) ; col 0 = t(seconds), col 1: = case-constant
        # FP32 jvp: network params are FP32 anyway — no precision gain from FP64 functional_call
        # // FP32 jvp로 FP32 코어 활용 (검증: domega/dt floor 1.8e-5 << 잔차 O(100)). AngularAccel/GetEnergy만 FP64 유지
        t = feats[:, 0:1].float()
        rest = feats[:, 1:].float()
        ones = torch.ones_like(t)

        def f(tc):                                        # (M, 1) -> ((M,2), (M,2))
            x = torch.cat([tc, rest], dim=1)
            out = self(x)                                 # (M, 4) [θ1, θ2, ω1, ω2]
            return out[:, :2], out[:, 2:]                 # theta, omega

        # single jvp // d/dt [θ, ω] = [dθ/dt, dω/dt]
        (theta, omega), (dtheta_dt, domega_dt) = tf.jvp(f, (t,), (ones,))
        return theta, omega, dtheta_dt, domega_dt

    def DataLoss(self, feats, theta_true, omega_true):
        out = self(feats)                                 # (N, 4)
        theta, omega = out[:, :2], out[:, 2:]
        return torch.mean((theta - theta_true) ** 2) + torch.mean(
            (omega - omega_true) ** 2
        )

    def PhysicsEnergyLoss(self, feats, params, e0):
        # FP32 jvp for network derivatives (검증: loss rel_diff < 4e-6) // FP32로 dθ/dt, dω/dt — FP32 코어 활용
        theta, omega, dtheta_dt, domega_dt = self.StateDerivs(feats.float())

        # Kinematic residual: dθ/dt = ω // 출력 ω와 θ의 시간미분 일치
        l_kin = torch.mean((dtheta_dt - omega) ** 2)

        # EOM + Energy stay FP64 — formula cancellation guard // 수식 상쇄 보호: AngularAccel/GetEnergy만 FP64
        # grad는 .double() 캐스트를 타고 FP32 망 파라미터로 환원. 수식 backward만 FP64(작음), 망 backward는 FP32
        # detach 금지: phys의 f_eom·energy의 state는 망 출력 theta/omega가 유일한 grad 경로 (끊으면 energy grad 소멸)
        with torch.autocast(device_type=self.device_type, enabled=False):
            p64 = params.double()
            e0_64 = e0.double()
            th64 = theta.double()
            om64 = omega.double()
            dom64 = domega_dt.double()

            # EOM residual: dω/dt = AngularAccel(θ, ω) // 운동방정식 잔차
            f_eom = self.AngularAccel(
                th64[:, 0], om64[:, 0], th64[:, 1], om64[:, 1], p64
            )
            # Per-sample relative residual: each point normalized by its own EOM scale // 샘플별 가속도 스케일로 무차원화: 저가속/고가속 구간 동등 비중
            denom = f_eom.abs() + self.dataCfg.phys_eps
            l_phys = torch.mean(((dom64 - f_eom) / denom) ** 2).float()

            # Energy residual (theta/omega 재사용)
            state = torch.stack(
                [th64[:, 0], om64[:, 0], th64[:, 1], om64[:, 1]], dim=1
            )
            e = self.GetEnergy(state, p64)
            denom_e = e0_64.abs() + self.dataCfg.energy_eps
            l_energy = torch.mean(((e - e0_64) / denom_e) ** 2).float()

        return l_kin, l_phys, l_energy

    def ICLoss(self, feats_ic, theta0_true, omega0_true):
        # τ=0 must reproduce the window-start state (θ and ω) // 윈도우 연속성: 시작 상태 전체 일치
        out = self(feats_ic)
        theta, omega = out[:, :2], out[:, 2:]
        return torch.mean((theta - theta0_true) ** 2) + torch.mean(
            (omega - omega0_true) ** 2
        )

    def RolloutLoss(self, case_idx, depth, n_points):
        # Pushforward (Brandstetter+ 2022): roll `depth` windows under no_grad to reach the
        # off-manifold IC the net actually produces, then one differentiable window matched to
        # true RK4 data — teaches the map to contract its own ω hand-off error back to truth.
        # // 예측 IC를 물려 만든 off-manifold 상태에서 다음 윈도우를 참 궤적으로 끌어오도록 학습
        dev = self.device
        md = self.dataCfg.march_dt
        dt = self.dt
        n_win = len(self.segments)

        # grad window index kg = k0 + depth must stay inside the data range // 데이터 구간 안에 들어오게
        depth = max(1, min(depth, n_win - 1))
        k0 = int(torch.randint(0, n_win - depth, (1,)).item())

        # true IC at window k0 start // 참 시작 상태
        i0 = int(round(k0 * md / dt))
        state = self.data[case_idx, i0][:, [1, 2, 3, 4]]

        # pushforward: only the end state is needed to hand off // 끝상태만 다음 IC로 전달
        tau_end = torch.full((1, 1), md, device=dev)
        with torch.no_grad():
            for _ in range(depth):
                out = self._RollWindow(case_idx, state, tau_end)   # (n,1,4)
                state = self._NextIC(out[:, -1, :])
        state = state.detach()                                     # fixed off-manifold input

        # differentiable window kg, matched to stored trajectory // grad 윈도우 = 참 데이터 매칭
        kg = k0 + depth
        i_start = int(round(kg * md / dt))
        i_end = int(round((kg + 1) * md / dt))
        grid = torch.arange(i_start + 1, i_end + 1, device=dev)    # exclude τ=0, include end
        if grid.numel() > n_points:
            sel = torch.linspace(0, grid.numel() - 1, n_points, device=dev).round().long()
            grid = grid[sel]
        tau = (self.t_grid[grid] - self.t_grid[i_start]).reshape(-1, 1).float()

        out = self._RollWindow(case_idx, state, tau)               # (n,P,4) with grad
        seg = self.data[case_idx][:, grid]                         # (n,P,13)
        theta_true = seg[:, :, [1, 3]]
        omega_true = seg[:, :, [2, 4]]
        return torch.mean((out[:, :, :2] - theta_true) ** 2) + torch.mean(
            (out[:, :, 2:] - omega_true) ** 2
        )

    def ComputeAllLosses(self, batch, colloc, ic):
        l_data = self.DataLoss(*batch)
        l_kin, l_phys, l_energy = self.PhysicsEnergyLoss(colloc[0], colloc[1], colloc[2])
        l_ic = self.ICLoss(ic[0], ic[1], ic[2])
        return {
            "data": l_data,
            "kin": l_kin,
            "phys": l_phys,
            "energy": l_energy,
            "ic": l_ic,
        }

    def ComputeDataICLosses(self, batch, ic):
        l_data = self.DataLoss(*batch)
        l_ic = self.ICLoss(ic[0], ic[1], ic[2])
        return {"data": l_data, "ic": l_ic}