import os

import numpy as np
import torch

class Dataset:
    def LoadData(self, path):
        arr = np.load(path)
        self.data = torch.from_numpy(arr).float() # (N, T, 13) CPU
        self.n_case, self.n_step, _ = self.data.shape
        self.t_grid = self.data[0, :, 0].clone()
        # Per-case constants // 케이스 상수: 파라미터 (m,L). IC는 윈도우 시작 상태에서 동적으로 추출
        self.params_raw = self.data[:, 0, 5:9].clone()  # (N, 4) m1,m2,L1,L2
        return self.n_case, self.n_step

    def ComputeScaler(self, scaler_dir):
        # log z-score for m, L over training pool (design 2.4 #3) // 로그 후 z-score
        log_p = torch.log(self.params_raw.double())               # (N, 4)
        mu = log_p.mean(dim=0)
        sigma = log_p.std(dim=0) + 1e-8
        self.register_buffer("param_mu", mu.float())
        self.register_buffer("param_sigma", sigma.float())

        # RMS scale for ω input over all states // 각속도 입력 무차원화 — IC가 임의 윈도우 시작 상태라 전 구간 ω로 RMS
        omega_all = self.data[:, :, [2, 4]].double()              # (N, T, 2) ω1, ω2
        omega_rms = torch.sqrt((omega_all ** 2).mean()) + 1e-8
        self.register_buffer("omega_rms", omega_rms.float())

        os.makedirs(scaler_dir, exist_ok=True)
        np.save(
            os.path.join(scaler_dir, self.dataCfg.scaler_name),
            {
                "param_mu": mu.numpy(),
                "param_sigma": sigma.numpy(),
                "omega_rms": omega_rms.numpy(),
            },
        )

    def LoadScaler(self, scaler_path):
        # Inference-time reuse of training stats // 추론 시 학습 통계 재사용
        d = np.load(scaler_path, allow_pickle=True).item()
        self.register_buffer("param_mu", torch.tensor(d["param_mu"]).float())
        self.register_buffer("param_sigma", torch.tensor(d["param_sigma"]).float())
        self.register_buffer("omega_rms", torch.tensor(d["omega_rms"]).float())

    def NormParams(self, params_raw):
        # (N, 4) raw m,L -> log z-score // 정규화
        return (torch.log(params_raw) - self.param_mu) / self.param_sigma

    def _BuildFeats(self, tau, case_idx, ic_state):
        # Flow-map features // 플로우맵 입력: 상대시간 τ + 윈도우 시작 상태(IC) + 파라미터
        # tau: (M, 1) relative time in window ; case_idx: (M,) ; ic_state: (M, 4) [θ1,ω1,θ2,ω2] at window start
        trig = self.GetTrigs(ic_state)                            # (M, 4) sin/cos of IC angles
        omega = ic_state[:, [1, 3]] / self.omega_rms              # (M, 2) IC ω RMS 정규화
        pn = self.NormParams(self.params_raw[case_idx])           # (M, 4)
        return torch.cat([tau, trig, omega, pn], dim=1)           # [τ, trig, ω, params]

    def _WindowStartIdx(self, t_lo):
        # grid index of the window start time // 윈도우 시작 시점의 격자 인덱스
        return min(max(int(round(t_lo / self.dt)), 0), self.n_step - 1)

    def SegmentSamples(self, t_lo, t_hi):
        # Flatten (case, time-in-window) into data-loss samples; IC = state at window start // 윈도우 데이터 샘플
        mask = (self.t_grid >= t_lo) & (self.t_grid <= t_hi)
        ti = torch.nonzero(mask, as_tuple=True)[0]                # (Tw,)
        seg = self.data[self.active_cases][:, ti, :]              # (n_active, Tw, 13)
        n, tw, _ = seg.shape

        case_idx = self.active_cases.repeat_interleave(tw)        # global indices (n_active*Tw,)
        tau = (seg[:, :, 0].reshape(-1, 1) - t_lo)                # relative time within window

        i0 = self._WindowStartIdx(t_lo)
        ic_state = self.data[self.active_cases, i0][:, [1, 2, 3, 4]]   # (n_active, 4) window-start state
        ic_flat = ic_state.repeat_interleave(tw, dim=0)               # (n_active*Tw, 4)
        feats = self._BuildFeats(tau, case_idx, ic_flat)              # (n_active*Tw, 11)

        theta_true = torch.stack(
            [seg[:, :, 1].reshape(-1), seg[:, :, 3].reshape(-1)], dim=1
        )
        omega_true = torch.stack(
            [seg[:, :, 2].reshape(-1), seg[:, :, 4].reshape(-1)], dim=1
        )
        params = self.params_raw[case_idx]                        # (n_active*Tw, 4) raw
        return feats, theta_true, omega_true, params

    def ICSamples(self, max_n=None):
        # τ=0 at every window start: predicted state must equal window-start state // 윈도우 연속성 제약 (전 상태)
        sel = self.active_cases
        if max_n is not None and len(sel) > max_n:
            sel = sel[torch.randperm(len(sel), device=self.device)[:max_n]]

        feats_list, theta_list, omega_list = [], [], []
        tau0 = torch.zeros(len(sel), 1, device=self.device)
        for t_lo, _ in self.segments:
            i0 = self._WindowStartIdx(t_lo)
            ic_state = self.data[sel, i0][:, [1, 2, 3, 4]]        # (M, 4) [θ1,ω1,θ2,ω2]
            feats_list.append(self._BuildFeats(tau0, sel, ic_state))
            theta_list.append(ic_state[:, [0, 2]])               # [θ1, θ2]
            omega_list.append(ic_state[:, [1, 3]])               # [ω1, ω2]

        feats = torch.cat(feats_list, dim=0)
        theta0 = torch.cat(theta_list, dim=0)
        omega0 = torch.cat(omega_list, dim=0)
        params = self.params_raw[sel.repeat(len(self.segments))]
        return feats, theta0, omega0, params
