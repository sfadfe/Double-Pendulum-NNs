import contextlib

import torch


class Loss:
    def ThetaDerivs(self, feats, create_graph=True):
        # theta = net(t, IC); w = dθ/dt, a = d²θ/dt² via autograd (design 2.4, 3.1)
        # feats: (M, 9) ; col 0 = t(seconds, leaf), col 1:9 = case-constant
        # 시간에 대한 1·2차 autograd 미분으로 ω, 각가속도 유도
        t = feats[:, 0:1].clone().requires_grad_(True)
        rest = feats[:, 1:9]
        full = torch.cat([t, rest], dim=1)

        theta = self(full)                                # (M, 2) [θ1, θ2]
        omega_cols, accel_cols = [], []
        for i in range(theta.shape[1]):
            g = torch.autograd.grad(
                theta[:, i].sum(), t, create_graph=True, retain_graph=True
            )[0]                                          # (M, 1) dθ_i/dt
            a = torch.autograd.grad(
                g.sum(), t, create_graph=create_graph, retain_graph=True
            )[0]                                          # (M, 1) d²θ_i/dt²
            omega_cols.append(g)
            accel_cols.append(a)
        omega = torch.cat(omega_cols, dim=1)              # (M, 2)
        accel = torch.cat(accel_cols, dim=1)              # (M, 2)
        return theta, omega, accel

    def DataLoss(self, feats, theta_true, omega_true):
        # θ에 직접, ω는 autograd 유도값에 데이터 손실 (design 2.4) // θ + 유도 ω MSE
        theta, omega, _ = self.ThetaDerivs(feats, create_graph=False)
        return torch.mean((theta - theta_true) ** 2) + torch.mean(
            (omega - omega_true) ** 2
        )

    def EnergyLoss(self, feats, params, e0):
        # relative energy error |E(t)-E0|/|E0| (design 4.3) // 상대 에너지 보존 오차
        theta, omega, _ = self.ThetaDerivs(feats, create_graph=False)
        state = torch.stack(
            [theta[:, 0], omega[:, 0], theta[:, 1], omega[:, 1]], dim=1
        )
        e = self.GetEnergy(state, params)
        denom = e0.abs() + self.config.energy_eps
        return torch.mean(((e - e0) / denom) ** 2)

    def PhysicsLoss(self, feats, params):
        # EOM residual: d²θ/dt² - f(θ, ω, params) (design 3.1) // 물리 잔차
        # design 4.6 (A): 잔차 경로 forward 전체 float64
        ctx = self._Float64Path() if self.config.physics_float64 else _NullCtx()
        with ctx:
            f = feats.double() if self.config.physics_float64 else feats
            p = params.double() if self.config.physics_float64 else params
            theta, omega, accel = self.ThetaDerivs(f, create_graph=True)
            f_eom = self.AngularAccel(
                theta[:, 0], omega[:, 0], theta[:, 1], omega[:, 1], p
            )                                              # (M, 2)
            res = accel - f_eom
            loss = torch.mean(res**2)
        return loss.float()

    def ICLoss(self, feats_ic, theta0_true):
        # θ(t=0) = IC truth (design 4.3 L_ic) // 초기조건 손실
        theta = self(feats_ic)
        return torch.mean((theta - theta0_true) ** 2)

    @contextlib.contextmanager
    def _Float64Path(self):
        # Full forward incl. weights in float64, restore after (design 4.6 fix)
        # 가중치 포함 전체 경로 float64 → 복원 (메모리상 콜로케이션 배치 작게)
        self.double()
        try:
            with torch.autocast(device_type=self.device_type, enabled=False):
                yield
        finally:
            self.float()

    def ComputeAllLosses(self, batch, colloc, ic):
        # batch=(feats, θ_true, ω_true) ; colloc=(feats, params, e0) ; ic=(feats, θ0)
        # 손실 항만 반환; λ 가중·정규화는 LambdaBalance, 합산은 학습 루프(미구현)
        l_data = self.DataLoss(*batch)
        l_phys = self.PhysicsLoss(colloc[0], colloc[1])
        l_energy = self.EnergyLoss(colloc[0], colloc[1], colloc[2])
        l_ic = self.ICLoss(*ic)
        return {
            "data": l_data,
            "phys": l_phys,
            "energy": l_energy,
            "ic": l_ic,
        }


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
