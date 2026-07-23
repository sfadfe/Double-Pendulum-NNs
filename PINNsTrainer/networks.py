import torch
import torch.nn as nn

"""
- 입력 feats (N, feat_dim): [τ, sinθ1, cosθ1, sinθ2, cosθ2, ω1, ω2, param_embed(32)]
  ParamEmbed는 Dataset._BuildFeats에서 호출 — raw m/L 대신 임베딩 벡터 사용 // option B
- time-marching flow map: τ는 윈도우 내 상대시간(τ∈[0, march_dt]), trig/ω는 윈도우 시작 상태(IC)
- τ는 Fourier Features(물리 주파수, Hz)로 매핑되고, 또한 [0, march_dt]에서 [-1, 1]로 정규화됨
- 출력: (N, 4) = [Δθ1, Δθ2, ω1, ω2]. Δθ = θ(τ) − θ_IC (상대각), ω는 절대값.
  공유 trunk + head_theta / head_omega 분리 // option B1
"""


class FourierFeatures(nn.Module):
    # Fourier features on t  // 시간 t에 대해서만 fourier features 적용
    def __init__(self, fourier_l, f_min, f_max):
        super().__init__()
        # f_k = f_min * (f_max/f_min)^(k/(L-1)), non-learnable NeRF Style // 고정 주파수 NeRF 방식 채택
        exponent = torch.linspace(0.0, 1.0, fourier_l)
        freqs = f_min * (f_max / f_min) ** exponent
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        proj = 2.0 * torch.pi * t * self.freqs
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ParamEmbed(nn.Module):
    # params_norm(4) → dim → dim, SiLU // option B3
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, params_norm):
        return self.net(params_norm)


class ResidualBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.act = nn.SiLU()  # option B2: Tanh → SiLU

    def forward(self, h, u, v):
        z = self.act(self.fc1(h))
        return (1.0 - z) * u + z * v + h  # U, V gating + residual connection


class Networks(nn.Module):
    def __init__(self, netCfg, dataCfg):
        super().__init__()
        self.march_dt = dataCfg.march_dt   # 상대시간 정규화 기준 // window duration for τ normalization

        self.fourier = FourierFeatures(netCfg.fourier_l, netCfg.f_min, netCfg.f_max)
        self.param_embed = ParamEmbed(netCfg.param_embed_dim)

        gx = netCfg.gx_dim
        width = netCfg.width
        self.proj_u = nn.Sequential(nn.Linear(gx, width), nn.SiLU())
        self.proj_v = nn.Sequential(nn.Linear(gx, width), nn.SiLU())
        self.proj_in = nn.Sequential(nn.Linear(gx, width), nn.SiLU())

        self.blocks = nn.ModuleList(
            [ResidualBlock(width) for i in range(netCfg.n)]
        )
        self.head_theta = nn.Linear(width, 2)   # Δθ1, Δθ2 // option B1
        self.head_omega = nn.Linear(width, 2)   # ω1, ω2

        # Kaiming init for SiLU // Tanh용 Xavier 대신 ReLU-family 휴리스틱
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
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

        theta = self.head_theta(h)
        omega = self.head_omega(h)
        return torch.cat([theta, omega], dim=-1)  # (N, 4) [Δθ1, Δθ2, ω1, ω2]
