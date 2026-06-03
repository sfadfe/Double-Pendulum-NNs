import contextlib
import torch
import torch.func as tf

class Loss:
    def ThetaDerivs(self, feats, need_accel=True):
        # theta = net(t, IC); w = dθ/dt, a = d²θ/dt² via forward-mode AD (jvp)
        # feats: (M, 9) ; col 0 = t(seconds), col 1:9 = case-constant
        # t가 행마다 스칼라라 forward-mode가 효율적 // jvp 중첩으로 1·2차 도함수 유도
        t = feats[:, 0:1]
        rest = feats[:, 1:9]
        ones = torch.ones_like(t)

        def f(tc):                                        # (M, 1) -> (M, 2) [θ1, θ2]
            return self(torch.cat([tc, rest], dim=1))

        # Data path skips 2nd derivative // accel 불필요 시 jvp 1회로 단축
        if not need_accel:
            theta, omega = tf.jvp(f, (t,), (ones,))       # theta, dθ/dt
            return theta, omega, None

        def f_d(tc):                                      # (theta, dθ/dt)
            return tf.jvp(f, (tc,), (ones,))

        # forward-over-forward // d²θ/dt² = jvp의 jvp
        (theta, omega), (_, accel) = tf.jvp(f_d, (t,), (ones,))
        return theta, omega, accel

    def DataLoss(self, feats, theta_true, omega_true):
        theta, omega, _ = self.ThetaDerivs(feats, need_accel=False)
        return torch.mean((theta - theta_true) ** 2) + torch.mean(
            (omega - omega_true) ** 2
        )

    def PhysicsEnergyLoss(self, feats, params, e0):
        # Physics + Energy share the same forward // 동일 forward 재사용 (FP64)
        with self._Float64Path():
            f64 = feats.double()
            p64 = params.double()
            e0_64 = e0.double()

            theta, omega, accel = self.ThetaDerivs(f64)

            # Physics residual
            f_eom = self.AngularAccel(
                theta[:, 0], omega[:, 0], theta[:, 1], omega[:, 1], p64
            )
            l_phys = torch.mean((accel - f_eom) ** 2)

            # Energy residual (theta/omega 재사용)
            state = torch.stack(
                [theta[:, 0], omega[:, 0], theta[:, 1], omega[:, 1]], dim=1
            )
            e = self.GetEnergy(state, p64)
            denom = e0_64.abs() + self.dataCfg.energy_eps
            l_energy = torch.mean(((e - e0_64) / denom) ** 2)

        return l_phys.float(), l_energy.float()

    def ICLoss(self, feats_ic, theta0_true):
        theta = self(feats_ic)
        return torch.mean((theta - theta0_true) ** 2)
    

    def ComputeAllLosses(self, batch, colloc, ic):
        l_data = self.DataLoss(*batch)
        l_phys, l_energy = self.PhysicsEnergyLoss(colloc[0], colloc[1], colloc[2])
        l_ic = self.ICLoss(ic[0], ic[1])
        return {
            "data": l_data,
            "phys": l_phys,
            "energy": l_energy,
            "ic": l_ic,
        }

    @contextlib.contextmanager
    def _Float64Path(self):
        # Using FP64 in Calculate and return FP32 // FP64로 계산 후 FP32로 반환
        self.double()
        try:
            with torch.autocast(device_type=self.device_type, enabled=False):
                yield
        finally:
            self.float()

# Dataloss 2차 미분 제거랑, FP64 path self.double() 구조 변경 먼저 해봐야함