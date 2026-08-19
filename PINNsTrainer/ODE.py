import torch


class Physics:
    def ODE(self, state, params, trigs=None):
        # state: (N, 4) -> [θ1, ω1, θ2, ω2]
        # params: (N, 4) -> [m1, m2, L1, L2]
        if trigs is None:
            trigs = self.GetTrigs(state)

        t1, w1, t2, w2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        m1, m2, L1, L2 = params[:, 0], params[:, 1], params[:, 2], params[:, 3]

        sin_t1, cos_t1 = trigs[:, 0], trigs[:, 1]
        sin_t2, cos_t2 = trigs[:, 2], trigs[:, 3]

        s_delta = sin_t1 * cos_t2 - cos_t1 * sin_t2  # sin(θ1-θ2)
        c_delta = cos_t1 * cos_t2 + sin_t1 * sin_t2  # cos(θ1-θ2)

        cos_2_delta = 2.0 * (c_delta**2) - 1.0
        den = 2.0 * m1 + m2 - m2 * cos_2_delta

        sin_t1_minus_2t2 = s_delta * cos_t2 - c_delta * sin_t2
        num1 = (
            -self.g * (2.0 * m1 + m2) * sin_t1
            - m2 * self.g * sin_t1_minus_2t2
            - 2.0 * s_delta * m2 * (w2**2 * L2 + w1**2 * L1 * c_delta)
        )
        domega1 = num1 / (L1 * den)

        num2 = (
            2.0
            * s_delta
            * (
                w1**2 * L1 * (m1 + m2)
                + self.g * (m1 + m2) * cos_t1
                + w2**2 * L2 * m2 * c_delta
            )
        )
        domega2 = num2 / (L2 * den)

        return torch.stack([w1, domega1, w2, domega2], dim=1)

    def AngularAccel(self, th1, w1, th2, w2, params):
        # EOM angular accelerations for physics residual // 물리 잔차용 각가속도
        # th*, w*: (N,) ; params: (N, 4) ; returns (N, 2) = [dω1/dt, dω2/dt]
        state = torch.stack([th1, w1, th2, w2], dim=1)
        d = self.ODE(state, params)   # (N, 4) = [w1, dω1/dt, w2, dω2/dt]
        return d[:, [1, 3]]

    def GetTrigs(self, state):
        return torch.stack(
            [
                torch.sin(state[:, 0]),
                torch.cos(state[:, 0]),
                torch.sin(state[:, 2]),
                torch.cos(state[:, 2]),
            ],
            dim=1,
        )

    def GetEnergy(self, state, params):
        th1, w1, th2, w2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        m1, m2, L1, L2 = params[:, 0], params[:, 1], params[:, 2], params[:, 3]

        # 운동에너지 // Kinetic
        v1_sq = (L1 * w1) ** 2
        v2_sq = v1_sq + (L2 * w2) ** 2 + 2.0 * L1 * L2 * w1 * w2 * torch.cos(th1 - th2)
        K = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq

        # 위치에너지 // Potential
        y1 = -L1 * torch.cos(th1)
        y2 = y1 - L2 * torch.cos(th2)
        V = m1 * self.g * y1 + m2 * self.g * y2

        return K + V

