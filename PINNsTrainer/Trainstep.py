import torch

from .Dataset import C_TH1, C_W1, C_TH2, C_W2


class LambdaBalance:
    # Loss normalization + sigmoid λ curriculum (design 4.3, 4.4) // 손실 정규화·λ 커리큘럼
    def InitLambda(self):
        self.l0 = {"data": None, "phys": None, "energy": None, "ic": None}
        self._warm_sum = {k: 0.0 for k in self.l0}
        self._warm_cnt = 0

    def AccumulateWarmup(self, losses):
        # First N steps: collect mean for L_*_0 (design 4.3) // 워밍업 평균 수집
        if self._warm_cnt >= self.config.warmup_steps:
            return False
        for k, v in losses.items():
            self._warm_sum[k] += float(v.detach())
        self._warm_cnt += 1
        if self._warm_cnt == self.config.warmup_steps:
            for k in self.l0:
                self.l0[k] = max(self._warm_sum[k] / self._warm_cnt, 1e-12)
        return True

    def _Sigmoid(self, step):
        c = self.config
        x = (step - c.lambda_sigmoid_mid) / (c.lambda_sigmoid_steps / 6.0)
        return float(torch.sigmoid(torch.tensor(x)))

    def LambdaAt(self, step):
        # λ_ic, λ_data fixed; λ_phys, λ_energy sigmoid (design 4.4) // 단계별 λ
        c = self.config
        s = self._Sigmoid(step)
        return {
            "ic": c.lambda_ic,
            "data": c.lambda_data,
            "phys": c.lambda_phys_init
            + (c.lambda_phys_final - c.lambda_phys_init) * s,
            "energy": c.lambda_energy_init
            + (c.lambda_energy_final - c.lambda_energy_init) * s,
        }

    def NormalizedTotal(self, losses, step):
        # L = Σ λ · (L / L_0) (design 4.3) // 정규화 가중합
        lam = self.LambdaAt(step)
        total = 0.0
        for k, v in losses.items():
            denom = self.l0[k] if self.l0[k] is not None else 1.0
            total = total + lam[k] * (v / denom)
        return total


class TimeMarching:
    # Overlapping micro-segments + LHS/RAR collocation (design 4.1, 4.2)
    # 겹침 구간 분할 + LHS/RAR 콜로케이션
    def BuildSegments(self, t_min, t_max):
        c = self.config
        span = (t_max - t_min) / c.seg_count
        delta = c.overlap_frac * span
        segs = []
        for i in range(c.seg_count):
            lo = t_min + i * span
            hi = min(lo + span + delta, t_max)
            segs.append((lo, hi))
        self.segments = segs
        return segs

    def SampleCollocation(self, t_lo, t_hi, n=None):
        # LHS over t + random case; e0 = true IC energy (design 4.2) // 콜로케이션 샘플
        n = n or self.config.n_colloc
        dev = self.device

        # Latin Hypercube on t // t축 라틴 하이퍼큐브
        edges = torch.linspace(0.0, 1.0, n + 1, device=dev)
        u = edges[:-1] + torch.rand(n, device=dev) * (1.0 / n)
        u = u[torch.randperm(n, device=dev)]
        t = (t_lo + u * (t_hi - t_lo)).reshape(-1, 1)

        case_idx = torch.randint(0, self.n_case, (n,), device=dev)
        feats = self._BuildFeats(t, case_idx)
        params = self.params_raw[case_idx].to(dev)

        ic = self.data[case_idx.cpu(), 0, :].to(dev)
        e0 = self.GetEnergy(
            torch.stack(
                [ic[:, C_TH1], ic[:, C_W1], ic[:, C_TH2], ic[:, C_W2]], dim=1
            ),
            params,
        ).detach()
        return feats, params, e0

    def RARUpdate(self, feats, params, e0, residual):
        # Drop low-residual, densify near high-residual, keep n fixed (design 4.2)
        # 잔차 하위 제거 + 상위 근방 밀집, 총 포인트 수 유지
        c = self.config
        n = feats.shape[0]
        order = torch.argsort(residual, descending=True)
        n_top = int(c.rar_top_frac * n)
        n_bot = int(c.rar_bot_frac * n)

        keep = order[: n - n_bot]                              # 하위 제거
        feats_k, params_k, e0_k = feats[keep], params[keep], e0[keep]

        # Jitter t near top-residual points to refill removed budget // 상위 근방 재샘플
        top = order[:n_top]
        reps = (n_bot + n_top - 1) // max(n_top, 1)
        src = top.repeat(reps)[:n_bot]
        jitter = (torch.rand(n_bot, 1, device=feats.device) - 0.5) * 2.0
        dt = (self.t_grid[1] - self.t_grid[0]).to(feats.device)
        new_feats = feats[src].clone()
        new_feats[:, 0:1] = new_feats[:, 0:1] + jitter * dt

        feats_n = torch.cat([feats_k, new_feats], dim=0)
        params_n = torch.cat([params_k, params[src]], dim=0)
        e0_n = torch.cat([e0_k, e0[src]], dim=0)
        return feats_n, params_n, e0_n
