import torch
import torch.nn as nn

"""
- 입력: (N, 9) = [t, sin t1, cos t1, sin t2, cos t2, m1, m2, L1, L2]
- t는 Fourier Features로 변환되어 주파수 공간으로 매핑됨, t는 또한 [0, t_data_max]에서 [-1, 1]로 정규화됨.
- 각도 trig은 t=0 케이스 상수, m/L은 log z-score
"""


class FourierFeatures(nn.Module):
    # Fourier features on t  // 시간 t에 대해서만 fourier features 적용
    def __init__(self, fourier_l, f_min, f_max):
        super().__init__()
        # f_k = f_min * (f_max/f_min)^(k/(L-1)), non-learnable NeRF Style // 고정 주파수 NeRF 방식 채용
        exponent = torch.linspace(0.0, 1.0, fourier_l)
        freqs = f_min * (f_max / f_min) ** exponent
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        proj = 2.0 * torch.pi * t * self.freqs
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.act = nn.Tanh()

    def forward(self, h, u, v):
        z = self.act(self.fc1(h))
        return (1.0 - z) * u + z * v + h  # U, V gating + residual connection


class Networks(nn.Module):
    def __init__(self, netCfg, dataCfg):
        super().__init__()
        self.t_data_max = dataCfg.t_data_max

        self.fourier = FourierFeatures(netCfg.fourier_l, netCfg.f_min, netCfg.f_max)

        gx = netCfg.gx_dim
        width = netCfg.width
        self.proj_u = nn.Sequential(nn.Linear(gx, width), nn.Tanh())
        self.proj_v = nn.Sequential(nn.Linear(gx, width), nn.Tanh())
        self.proj_in = nn.Sequential(nn.Linear(gx, width), nn.Tanh())

        self.blocks = nn.ModuleList(
            [ResidualBlock(width) for i in range(netCfg.n)]
        )
        self.out = nn.Linear(width, 2)

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feats):

        t = feats[:, 0:1]
        emb_t = self.fourier(t)  # (N, 2L)
        t_norm = 2.0 * t / self.t_data_max - 1.0
        x = torch.cat([emb_t, t_norm, feats[:, 1:9]], dim=-1)  # (N, gx_dim)

        u = self.proj_u(x)
        v = self.proj_v(x)
        h = self.proj_in(x)

        for block in self.blocks:  # Pass all residual blocks through U and V gates. // 모든 residual block을 U, V 게이트에 통과시킴
            h = block(h, u, v)

        return self.out(h)  # [th1, th2]
