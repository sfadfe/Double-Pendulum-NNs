import torch
import torch.func as tf

class Loss:
    def StateDerivs(self, feats):
        # net outputs [Δθ, ω]; one forward-mode pass gives [dΔθ/dt, dω/dt]
        # dΔθ/dt = dθ/dt since θ_IC is constant → kin residual (dΔθ/dt = ω) unchanged
        # feats: (M, feat_dim) ; col 0 = τ, col 1: = case-constant (IC + param_embed)
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

    def _KinTerm(self, omega, dtheta_dt):
        # kin = 두 헤드를 잇는 커플러 (ω := dθ/dτ). // structural glue, not a physics law
        # 상대 정규화: 분모 = mean(dθ/dτ²).detach() → 미학습 랜덤 init의 거대 magnitude를 상쇄해
        #   gradient가 스케일 무관하게 bounded. raw면 init grad가 data를 4000× 지배(측정) → data피팅 잠식.
        #   phys loss의 상대잔차 철학과 동일 // [[phys-loss-relative-norm]]
        # _kin_detach=True(Phase 2): dθ/dτ 기준 고정, ω_head가 추종 → rollout 핸드오프 ω 드리프트 억제.
        # _kin_detach=False(Phase 1): 대칭 — 두 헤드가 함께 커플링 학습.
        scale = dtheta_dt.detach().pow(2).mean() + self.dataCfg.kin_eps
        if getattr(self, "_kin_detach", False):
            return torch.mean((omega - dtheta_dt.detach()) ** 2) / scale
        return torch.mean((dtheta_dt - omega) ** 2) / scale

    def KinLoss(self, feats):
        # Phase 1 커플링 전용: kin만 (jvp, FP64 EOM/energy 없음) // cheap structural coupling
        _, omega, dtheta_dt, _ = self.StateDerivs(feats.float())
        return self._KinTerm(omega, dtheta_dt)

    def DataLoss(self, feats, theta_true, omega_true):
        out = self(feats)                                 # (N, 4)
        theta, omega = out[:, :2], out[:, 2:]
        return torch.mean((theta - theta_true) ** 2) + torch.mean(
            (omega - omega_true) ** 2
        )

    def _PhysicsEnergySlice(self, feats, params, e0, ic_theta):
        # Single colloc slice // 콜로케이션 청크 1개분 물리+에너지 손실
        theta, omega, dtheta_dt, domega_dt = self.StateDerivs(feats.float())
        l_kin = self._KinTerm(omega, dtheta_dt)

        with torch.autocast(device_type=self.device_type, enabled=False):
            p64 = params.double()
            e0_64 = e0.double()
            th64 = ic_theta.double() + theta.double()
            om64 = omega.double()
            dom64 = domega_dt.double()
            f_eom = self.AngularAccel(
                th64[:, 0], om64[:, 0], th64[:, 1], om64[:, 1], p64
            )
            denom = f_eom.abs() + self.dataCfg.phys_eps
            l_phys = torch.mean(((dom64 - f_eom) / denom) ** 2).float()
            state = torch.stack(
                [th64[:, 0], om64[:, 0], th64[:, 1], om64[:, 1]], dim=1
            )
            e = self.GetEnergy(state, p64)
            # max(|E0|,|E|) scale: E0≈0 zero-crossing 시 |E0|만 분모로 쓰면 폭발 // symmetric energy scale
            denom_e = torch.maximum(e0_64.abs(), e.abs()) + self.dataCfg.energy_eps
            l_energy = torch.mean(((e - e0_64) / denom_e) ** 2).float()

        return l_kin, l_phys, l_energy

    def PhysicsEnergyLoss(self, feats, params, e0, ic_theta):
        l_kin, l_phys, l_energy = self._PhysicsEnergySlice(feats, params, e0, ic_theta)
        return l_kin, l_phys, l_energy

    def _ICLossSlice(self, feats_ic, theta0_true, omega0_true):
        # τ=0 window-start match on a slice // IC 손실 청크
        out = self(feats_ic)
        theta, omega = out[:, :2], out[:, 2:]
        return torch.mean((theta - theta0_true) ** 2) + torch.mean(
            (omega - omega0_true) ** 2
        )

    def ICLoss(self, feats_ic, theta0_true, omega0_true):
        return self._ICLossSlice(feats_ic, theta0_true, omega0_true)

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
        data, _ = self._ActiveSource()
        if data.device != self.device:
            ci = case_idx.cpu()
        else:
            ci = case_idx
        state = data[ci, i0][:, [1, 2, 3, 4]]
        if state.device != self.device:
            state = state.to(self.device)
        params = self._ParamsAt(case_idx)

        # pushforward: net outputs Δθ; absolute θ_next = θ_start + Δθ_end // 감김수 누적 핸드오프
        tau_end = torch.full((1, 1), md, device=dev)
        with torch.no_grad():
            for _ in range(depth):
                out = self._RollWindow(case_idx, params, state, tau_end)   # (n,1,4) [Δθ,ω]
                last = out[:, -1, :]
                th_abs = state[:, [0, 2]] + last[:, :2]                    # 절대각 복원
                state = torch.stack(
                    [th_abs[:, 0], last[:, 2], th_abs[:, 1], last[:, 3]], dim=1
                )
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

        out = self._RollWindow(case_idx, params, state, tau)       # (n,P,4) [Δθ,ω] with grad
        theta_pred = state[:, [0, 2]].unsqueeze(1) + out[:, :, :2]  # 절대각 = 윈도우 시작각 + Δθ
        omega_pred = out[:, :, 2:]
        grid_idx = grid.cpu() if data.device != self.device else grid
        seg = data[ci][:, grid_idx]
        if seg.device != self.device:
            seg = seg.to(self.device)
        theta_true = seg[:, :, [1, 3]]
        omega_true = seg[:, :, [2, 4]]
        return torch.mean((theta_pred - theta_true) ** 2) + torch.mean(
            (omega_pred - omega_true) ** 2
        )

    def ComputeAllLosses(self, batch, colloc, ic):
        l_data = self.DataLoss(*batch)
        l_kin, l_phys, l_energy = self.PhysicsEnergyLoss(
            colloc[0], colloc[1], colloc[2], colloc[3]
        )
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