import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def equations(state, m1, m2, L1, L2, g):
    t1, w1, t2, w2 = state[0], state[1], state[2], state[3]

    delta = t1 - t2
    s_delta = np.sin(delta)
    c_delta = np.cos(delta)

    # 공통 분모
    den = 2 * m1 + m2 - m2 * np.cos(2 * delta)

    # 가속도 1
    num1 = (
        -g * (2 * m1 + m2) * np.sin(t1)
        - m2 * g * np.sin(t1 - 2 * t2)
        - 2 * s_delta * m2 * (w2**2 * L2 + w1**2 * L1 * c_delta)
    )
    domega1 = num1 / (L1 * den)

    # 가속도 2
    num2 = (
        2
        * s_delta
        * (
            w1**2 * L1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(t1)
            + w2**2 * L2 * m2 * c_delta
        )
    )
    domega2 = num2 / (L2 * den)

    return np.array([w1, domega1, w2, domega2])


@jit(nopython=True, cache=True)
def RK4(state, dt, m1, m2, L1, L2, g):
    k1 = equations(state, m1, m2, L1, L2, g)
    k2 = equations(state + 0.5 * dt * k1, m1, m2, L1, L2, g)
    k3 = equations(state + 0.5 * dt * k2, m1, m2, L1, L2, g)
    k4 = equations(state + dt * k3, m1, m2, L1, L2, g)

    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class Double_pendulum:
    def __init__(self, m1, m2, L1, L2, initial_state, g=9.81):
        self.m1 = m1
        self.m2 = m2
        self.L1 = L1
        self.L2 = L2
        self.g = g
        # state: [theta1, omega1, theta2, omega2]
        self.state = np.array(initial_state, dtype=np.float64)

    def RK4(self, dt):

        self.state = RK4(self.state, dt, self.m1, self.m2, self.L1, self.L2, self.g)
