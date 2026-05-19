import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    # Fixed geometric-spaced Fourier features on t (design 3.2) // t에만 기하 간격 푸리에
    def __init__(self, fourier_l, f_min, f_max):
        super().__init__()
        # f_k = f_min * (f_max/f_min)^(k/(L-1)), non-learnable // 고정 주파수
        exponent = torch.linspace(0.0, 1.0, fourier_l)
        freqs = f_min * (f_max / f_min) ** exponent
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        # t: (N, 1) seconds -> (N, 2L) // 초 단위 시간 → 사인/코사인
        proj = 2.0 * torch.pi * t * self.freqs  # (N, L), broadcast
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.act = nn.Tanh()

    def forward(self, h, u, v):
        z = self.act(self.fc1(h))
        return (1.0 - z) * u + z * v + h  # gating + residual


class Networks(nn.Module):
    # Continuous-time PINN: (t, IC) -> [th1, th2]; w via autograd (design 3.1) // 연속시간 PINN
    def __init__(self, config):
        super().__init__()
        self.t_data_max = config.t_data_max

        self.fourier = FourierFeatures(config.fourier_l, config.f_min, config.f_max)

        gx = config.gx_dim
        hidden = config.hidden
        self.proj_u = nn.Sequential(nn.Linear(gx, hidden), nn.Tanh())
        self.proj_v = nn.Sequential(nn.Linear(gx, hidden), nn.Tanh())
        self.proj_in = nn.Sequential(nn.Linear(gx, hidden), nn.Tanh())

        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden) for _ in range(config.n_blocks)]
        )
        self.head = nn.Linear(hidden, config.out_dim)

        # Xavier init // Tanh 계열 안정 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feats):
        # feats: (N, 9) = [t, sin t1, cos t1, sin t2, cos t2, m1n, m2n, L1n, L2n]
        #   t: seconds (Fourier 입력), 각도 trig은 t=0 케이스 상수, m/L은 log z-score
        t = feats[:, 0:1]
        emb_t = self.fourier(t)  # (N, 2L)
        t_norm = 2.0 * t / self.t_data_max - 1.0  # raw monotone ramp (design 3.2)
        x = torch.cat([emb_t, t_norm, feats[:, 1:9]], dim=-1)  # (N, gx_dim)

        u = self.proj_u(x)
        v = self.proj_v(x)
        h = self.proj_in(x)
        for blk in self.blocks:
            h = blk(h, u, v)
        return self.head(h)  # (N, 2) -> [th1, th2]
