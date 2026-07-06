import torch
import torch.nn as nn

"""
- 입력: (N, 11) = [τ, sin θ1, cos θ1, sin θ2, cos θ2, ω1, ω2, m1, m2, L1, L2]
- time-marching flow map: τ는 윈도우 내 상대시간(τ∈[0, march_dt]), trig/ω는 윈도우 시작 상태(IC)
- τ는 Fourier Features(물리 주파수, Hz)로 매핑되고, 또한 [0, march_dt]에서 [-1, 1]로 정규화됨
- m/L은 log z-score
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
        self.march_dt = dataCfg.march_dt   # 상대시간 정규화 기준 // window duration for τ normalization

        self.fourier = FourierFeatures(netCfg.fourier_l, netCfg.f_min, netCfg.f_max)

        gx = netCfg.gx_dim
        width = netCfg.width
        self.proj_u = nn.Sequential(nn.Linear(gx, width), nn.Tanh())
        self.proj_v = nn.Sequential(nn.Linear(gx, width), nn.Tanh())
        self.proj_in = nn.Sequential(nn.Linear(gx, width), nn.Tanh())

        self.blocks = nn.ModuleList(
            [ResidualBlock(width) for i in range(netCfg.n)]
        )
        self.out = nn.Linear(width, 4)  # state-space output -  [θ1, θ2, ω1, ω2] // 상태공간형, [θ1, θ2, ω1, ω2] 직접 출력

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feats):

        tau = feats[:, 0:1]                              # relative time within window // 윈도우 내 상대시간
        emb_t = self.fourier(tau)  # (N, 2L)
        t_norm = 2.0 * tau / self.march_dt - 1.0
        x = torch.cat([emb_t, t_norm, feats[:, 1:]], dim=-1)  # (N, gx_dim)

        u = self.proj_u(x)
        v = self.proj_v(x)
        h = self.proj_in(x)

        for block in self.blocks:  # Pass all residual blocks through U and V gates. // 모든 residual block을 U, V 게이트에 통과시킴
            h = block(h, u, v)

        return self.out(h)  # (N, 4) [θ1, θ2, ω1, ω2]
        