import os

import numpy as np
import torch

class Dataset:
    def LoadData(self, path):
        arr = np.load(path)
        self.data = torch.from_numpy(arr).float() # (N, T, 13) CPU
        self.n_case, self.n_step, _ = self.data.shape
        self.t_grid = self.data[0, :, 0].clone()
        # Per-case constants // 케이스 상수: 파라미터, t=0 각도 trig
        self.params_raw = self.data[:, 0, 5:9].clone()  # (N, 4) m1,m2,L1,L2
        self.ic_trig = self.data[:, 0, 9:13].clone()    # (N, 4) sin/cos@t=0
        return self.n_case, self.n_step

    def ComputeScaler(self, scaler_dir):
        # log z-score for m, L over training pool (design 2.4 #3) // 로그 후 z-score
        log_p = torch.log(self.params_raw.double())               # (N, 4)
        mu = log_p.mean(dim=0)
        sigma = log_p.std(dim=0) + 1e-8
        self.register_buffer("param_mu", mu.float())
        self.register_buffer("param_sigma", sigma.float())

        os.makedirs(scaler_dir, exist_ok=True)
        np.save(
            os.path.join(scaler_dir, self.dataCfg.scaler_name),
            {"param_mu": mu.numpy(), "param_sigma": sigma.numpy()},
        )

    def LoadScaler(self, scaler_path):
        # Inference-time reuse of training stats // 추론 시 학습 통계 재사용
        d = np.load(scaler_path, allow_pickle=True).item()
        self.register_buffer("param_mu", torch.tensor(d["param_mu"]).float())
        self.register_buffer("param_sigma", torch.tensor(d["param_sigma"]).float())

    def NormParams(self, params_raw):
        # (N, 4) raw m,L -> log z-score // 정규화
        return (torch.log(params_raw) - self.param_mu) / self.param_sigma

    def _BuildFeats(self, t, case_idx):
        # t: (M, 1) seconds ; case_idx: (M,) -> feats (M, 9)
        trig = self.ic_trig[case_idx]                             # (M, 4) 상수
        pn = self.NormParams(self.params_raw[case_idx])           # (M, 4)
        return torch.cat([t, trig, pn], dim=1)

    def SegmentSamples(self, t_lo, t_hi):
        # Flatten (case, time-in-window) into data-loss samples on GPU // 구간 데이터 샘플
        mask = (self.t_grid >= t_lo) & (self.t_grid <= t_hi)
        ti = torch.nonzero(mask, as_tuple=True)[0]                # (Tw,)
        seg = self.data[:, ti, :]                                  # (N, Tw, 13)
        n, tw, _ = seg.shape

        case_idx = torch.arange(n, device=self.device).repeat_interleave(tw)
        t = seg[:, :, 0].reshape(-1, 1)
        feats = self._BuildFeats(t, case_idx)                     # (N*Tw, 9)

        theta_true = torch.stack(
            [seg[:, :, 1].reshape(-1), seg[:, :, 3].reshape(-1)], dim=1
        )
        omega_true = torch.stack(
            [seg[:, :, 2].reshape(-1), seg[:, :, 4].reshape(-1)], dim=1
        )
        params = self.params_raw[case_idx]                        # (N*Tw, 4) raw
        return feats, theta_true, omega_true, params

    def ICSamples(self, max_n=None):
        # t=0 slice: θ(0)=IC truth (design 4.3 L_ic) // 초기조건 샘플
        n = self.n_case
        if max_n is not None and n > max_n:
            sel = torch.randperm(n, device=self.device)[:max_n]
        else:
            sel = torch.arange(n, device=self.device)
        t0 = torch.zeros(len(sel), 1, device=self.device)
        feats = self._BuildFeats(t0, sel)
        theta0 = torch.stack(
            [self.data[sel, 0, 1], self.data[sel, 0, 3]], dim=1
        )
        params = self.params_raw[sel]
        return feats, theta0, params
