import json
import os
from pathlib import Path

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
        self.is_flip = None
        self._replay_active = False
        return self.n_case, self.n_step

    def LoadMixedData(self, path, meta_path=None):
        # Finetune 단일 코퍼스 + meta 분할/replay 인덱스 // one-file, fresh IC corpus
        self.LoadData(path)
        self.meta = None
        self.replay_idx = None
        self.replay_n_case = 0
        if meta_path is None:
            meta_path = Path(path).with_suffix(".meta.json")
        meta_path = Path(meta_path)
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            flip = torch.tensor(meta["is_flip"], dtype=torch.bool)
            if flip.shape[0] != self.n_case:
                raise ValueError(f"is_flip len {flip.shape[0]} != n_case {self.n_case}")
            self.is_flip = flip
            self.meta = meta
            if "replay_idx" in meta:
                self.replay_idx = torch.tensor(meta["replay_idx"], dtype=torch.long)
                self.replay_n_case = int(self.replay_idx.shape[0])
        return self.n_case, self.n_step

    def SetSplitFromMeta(self):
        # 빌드 시 확정된 stratified 분할 사용 // 런타임 재분할 대신 재현 가능한 val/train
        if self.meta is None or "val_idx" not in self.meta:
            raise RuntimeError(
                "mixed meta에 val_idx가 없음 — state/dfss.py mixed로 재빌드하세요"
            )
        self.val_cases = torch.tensor(self.meta["val_idx"], dtype=torch.long, device=self.device)
        self.train_pool = torch.tensor(self.meta["train_idx"], dtype=torch.long, device=self.device)
        return len(self.val_cases), len(self.train_pool)

    def _ActiveSource(self):
        return self.data, self.params_raw

    def StratifiedValSplit(self, n_val, seed=42):
        # Shuffle while keeping flip fraction in val // is_flip 비율 유지 검증 분할
        if self.is_flip is None:
            raise RuntimeError("StratifiedValSplit requires is_flip metadata")
        n_val = min(int(n_val), self.n_case - 1)
        rng = np.random.default_rng(seed)
        flip_idx = torch.where(self.is_flip)[0].cpu().numpy()
        nf_idx = torch.where(~self.is_flip)[0].cpu().numpy()
        rng.shuffle(flip_idx)
        rng.shuffle(nf_idx)
        flip_frac = float(self.is_flip.float().mean())
        n_vf = min(len(flip_idx), max(1, int(round(flip_frac * n_val))))
        n_vnf = min(len(nf_idx), n_val - n_vf)
        n_vf = min(len(flip_idx), n_val - n_vnf)
        val_np = np.concatenate([flip_idx[:n_vf], nf_idx[:n_vnf]])
        rng.shuffle(val_np)
        val_idx = torch.tensor(val_np, dtype=torch.long, device=self.device)
        mask = torch.ones(self.n_case, dtype=torch.bool, device=self.device)
        mask[val_idx] = False
        self.val_cases = val_idx
        self.train_pool = torch.where(mask)[0]
        return len(self.val_cases), len(self.train_pool)

    def ComputeScaler(self, scaler_dir, extra_omega_paths=None):
        # log z-score for m, L over training pool (design 2.4 #3) // 로그 후 z-score
        log_p = torch.log(self.params_raw.double())               # (N, 4)
        mu = log_p.mean(dim=0)
        sigma = log_p.std(dim=0) + 1e-8
        self.register_buffer("param_mu", mu.float())
        self.register_buffer("param_sigma", sigma.float())

        # RMS scale for ω input over all states // 각속도 입력 무차원화 — IC가 임의 윈도우 시작 상태라 전 구간 ω로 RMS
        # extra_omega_paths(예: mixed corpus) 포함 시 flip ω까지 커버해 추론 입력 OOD 방지 // flip-aware scaler
        sq_sum = (self.data[:, :, [2, 4]].double() ** 2).sum()
        n_elem = self.data[:, :, [2, 4]].numel()
        for p in extra_omega_paths or []:
            if not Path(p).exists():
                continue
            extra = torch.from_numpy(np.load(p)).double()
            sq_sum = sq_sum + (extra[:, :, [2, 4]] ** 2).sum()
            n_elem += extra[:, :, [2, 4]].numel()
        omega_rms = torch.sqrt(sq_sum / n_elem) + 1e-8
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

    def _ParamsAt(self, case_idx):
        return self.params_raw[case_idx]

    def _BuildFeats(self, tau, params_raw, ic_state):
        # Flow-map features // 플로우맵 입력: τ + IC(trig,ω) + ParamEmbed(m,L)
        # params_raw는 호출자가 활성 소스에서 직접 해석해 전달 // explicit params, no hidden _replay_active dep
        trig = self.GetTrigs(ic_state)
        omega = ic_state[:, [1, 3]] / self.omega_rms
        pn = self.NormParams(params_raw)
        p_emb = self.param_embed(pn)  # Networks.ParamEmbed // option B3
        return torch.cat([tau, trig, omega, p_emb], dim=1)

    def _WindowStartIdx(self, t_lo):
        # grid index of the window start time // 윈도우 시작 시점의 격자 인덱스
        return min(max(int(round(t_lo / self.dt)), 0), self.n_step - 1)

    def SegmentFrame(self, t_lo, t_hi):
        # Raw segment cache without param_embed // param_embed 없는 raw만 캐시 — step 간 재사용 가능
        data, _ = self._ActiveSource()
        ac = self.active_cases
        if data.device != self.device:
            ac_cpu = ac.cpu()
        else:
            ac_cpu = ac

        mask = (self.t_grid >= t_lo) & (self.t_grid <= t_hi)
        ti = torch.nonzero(mask, as_tuple=True)[0]
        ti_idx = ti.cpu() if data.device != self.device else ti

        seg = data[ac_cpu][:, ti_idx, :]
        if seg.device != self.device:
            seg = seg.to(self.device)
        n, tw, _ = seg.shape

        case_idx = ac.repeat_interleave(tw)
        tau = (seg[:, :, 0].reshape(-1, 1) - t_lo)

        i0 = self._WindowStartIdx(t_lo)
        ic_state = data[ac_cpu, i0][:, [1, 2, 3, 4]]
        if ic_state.device != self.device:
            ic_state = ic_state.to(self.device)
        ic_flat = ic_state.repeat_interleave(tw, dim=0)
        params = self._ParamsAt(case_idx)

        # Δθ label: absolute seg angle minus window-start angle // 감김수 모순 제거용 상대각 라벨
        theta_abs = torch.stack(
            [seg[:, :, 1].reshape(-1), seg[:, :, 3].reshape(-1)], dim=1
        )
        ic_theta_flat = ic_flat[:, [0, 2]]
        theta_true = theta_abs - ic_theta_flat
        omega_true = torch.stack(
            [seg[:, :, 2].reshape(-1), seg[:, :, 4].reshape(-1)], dim=1
        )
        return {
            "tau": tau,
            "ic_flat": ic_flat,
            "params": params,
            "theta_true": theta_true,
            "omega_true": omega_true,
            "n_total": n * tw,
        }

    def SegmentBatch(self, frame, bs=None):
        # Fresh param_embed graph per step // step마다 param_embed 그래프 새로 구성
        n_total = frame["n_total"]
        bs = min(bs or self.dataCfg.batch_size, n_total)
        idx = torch.randint(0, n_total, (bs,), device=self.device)
        feats = self._BuildFeats(
            frame["tau"][idx], frame["params"][idx], frame["ic_flat"][idx]
        )
        return feats, frame["theta_true"][idx], frame["omega_true"][idx]

    def SegmentSamples(self, t_lo, t_hi):
        frame = self.SegmentFrame(t_lo, t_hi)
        feats = self._BuildFeats(frame["tau"], frame["params"], frame["ic_flat"])
        return feats, frame["theta_true"], frame["omega_true"], frame["params"]

    def ICSamplesRaw(self, max_n=None):
        # IC raw tensors without param_embed // param_embed 없는 IC raw (청크별 feats 생성용)
        data, _ = self._ActiveSource()
        sel = self.active_cases
        if max_n is not None and len(sel) > max_n:
            sel = sel[torch.randperm(len(sel), device=sel.device)[:max_n]]

        if data.device != self.device:
            sel_cpu = sel.cpu()
        else:
            sel_cpu = sel

        params_sel = self._ParamsAt(sel)
        tau0 = torch.zeros(len(sel), 1, device=self.device)
        parts = []
        for t_lo, _ in self.segments:
            i0 = self._WindowStartIdx(t_lo)
            ic_state = data[sel_cpu, i0][:, [1, 2, 3, 4]]
            if ic_state.device != self.device:
                ic_state = ic_state.to(self.device)
            parts.append({
                "tau": tau0,
                "params": params_sel,
                "ic_state": ic_state,
                "theta0": torch.zeros_like(ic_state[:, [0, 2]]),
                "omega0": ic_state[:, [1, 3]],
            })
        return parts

    def ICSamples(self, max_n=None):
        parts = self.ICSamplesRaw(max_n=max_n)
        feats_list, theta_list, omega_list = [], [], []
        for part in parts:
            feats_list.append(self._BuildFeats(part["tau"], part["params"], part["ic_state"]))
            theta_list.append(part["theta0"])
            omega_list.append(part["omega0"])
        feats = torch.cat(feats_list, dim=0)
        theta0 = torch.cat(theta_list, dim=0)
        omega0 = torch.cat(omega_list, dim=0)
        params = parts[0]["params"].repeat(len(self.segments))
        return feats, theta0, omega0, params
