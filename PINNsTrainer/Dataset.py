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
        parts = [tau, trig, omega, p_emb]
        if self.netCfg.energy_gate:
            parts.append(self._EnergyGate(ic_state, params_raw).to(p_emb.dtype))
        return torch.cat(parts, dim=1)

    @torch.no_grad()
    def _EnergyGate(self, ic_state, params_raw):
        # log(e_rel / e_flip) — state/states.py:62-77의 라벨 생성 기준과 동일한 장벽 비
        #   is_flip은 관측이 아니라 이 비로 '정의'된 라벨: flip은 >1, nonflip은 <0.6이고
        #   [0.6, 1.0] 밴드는 샘플링에서 버려져 실측 0건 → 완전 간격 분리(정확도 1.000).
        #   log를 취해 장벽에서 0-중심 + 상단 23배 꼬리 압축. E는 윈도우 보존량 = 케이스 상수라
        #   τ 무관 → Loss.StateDerivs의 jvp 불변. // [[flip-label-is-energy-defined]]
        p = params_raw.double()
        m1, m2, L1, L2 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        v_min = -self.g * ((m1 + m2) * L1 + m2 * L2)     # 전 배위 최소 위치에너지
        e_rel = self.GetEnergy(ic_state.double(), p) - v_min
        e_flip = torch.minimum(
            2.0 * self.g * m2 * L2, 2.0 * self.g * L1 * (m1 + m2)
        )
        log_ratio = torch.log(e_rel.clamp_min(1e-12) / e_flip)
        return log_ratio.unsqueeze(1)

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
