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
        return torch.cat([tau, trig, omega, p_emb], dim=1)

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
        # Returns (feats, θ, ω, n_q). Plain nets: iid rows, n_q = 1. xattn nets: whole cases (all tw grid rows
        # of bs//tw random cases, case-major) so forward(n_q=tw) shares the window tokens per case.
        # // xattn: 행 단위 추첨 대신 케이스 단위(창의 tw행 전부)
        n_total = frame["n_total"]
        bs = min(bs or self.dataCfg.batch_size, n_total)
        if getattr(self, "n_tok", 0) > 0:
            tw = n_total // max(len(self.active_cases), 1)          # SegmentFrame은 case-major (n_case, tw)
            n_c = max(1, bs // tw)
            cases = torch.randint(0, n_total // tw, (n_c, 1), device=self.device)
            idx = (cases * tw + torch.arange(tw, device=self.device)).reshape(-1)
            n_q = tw
        elif self.CaseGroup() > 1:
            # TrainCfg.case_group: bs//K random cases × K distinct grid rows each, case-major
            # // 케이스 bs//K개 × 창 격자에서 서로 다른 K행
            n_q = self.CaseGroup()
            tw = n_total // max(len(self.active_cases), 1)
            if n_q > tw:
                raise ValueError(f"case_group {n_q} exceeds the {tw} grid rows of a window")
            cases = torch.randint(0, n_total // tw, (bs // n_q, 1), device=self.device)
            rows = torch.rand(bs // n_q, tw, device=self.device).argsort(dim=1)[:, :n_q]
            idx = (cases * tw + rows).reshape(-1)
        else:
            idx = torch.randint(0, n_total, (bs,), device=self.device)
            n_q = 1
        feats = self._BuildFeats(
            frame["tau"][idx], frame["params"][idx], frame["ic_flat"][idx]
        )
        return feats, frame["theta_true"][idx], frame["omega_true"][idx], n_q

    def DataBatchSize(self):
        # Data batch rows of the current phase (DataCfg.batch_size_phase2, 0 = batch_size in both phases)
        # // 현재 위상의 data 배치 크기 — _phase1은 loop가 매 에폭 미러
        b2 = self.dataCfg.batch_size_phase2
        return b2 if b2 > 0 and not self._phase1 else self.dataCfg.batch_size

    # --- Fine label source of the data term // data 항의 촘촘한 라벨 소스 (DataCfg.label_path) ---
    def LoadLabels(self, base_dir):
        # Called by Setup / SetupFinetune after self.data is on the device. Every label key at its default → no-op.
        # // 라벨 파일 로드·검증. 키가 전부 기본값이면 아무것도 하지 않는다
        c = self.dataCfg
        if not c.label_path:
            if c.label_score_path or c.label_dt or c.label_dt_phase2 or c.label_dense_frac or c.slide_frac:
                raise ValueError("[data] label_score_path / label_dt / label_dt_phase2 / label_dense_frac / "
                                 "slide_frac need label_path")
            return
        if getattr(self, "n_tok", 0) > 0:
            raise ValueError("label_path: plain nets only (xattn / window-token nets draw whole windows)")
        if not (0.0 <= c.label_dense_frac <= 1.0 and 0.0 <= c.slide_frac <= 1.0):
            raise ValueError("label_dense_frac and slide_frac must be in [0, 1]")
        path = Path(c.label_path)
        arr = np.load(path if path.is_absolute() else Path(base_dir) / path)
        if arr.ndim != 3 or arr.shape[0] != self.n_case or arr.shape[1] < 2 or arr.shape[2] != 4:
            raise ValueError(f"label_path {tuple(arr.shape)}: expected ({self.n_case}, T_f, 4)")
        step = float(self.t_grid[-1] - self.t_grid[0]) / (arr.shape[1] - 1)     # dt_f

        def Steps(x, name):
            k = int(round(x / step))
            if k < 1 or abs(x / step - k) > 1e-6:
                raise ValueError(f"{name} {x:g} is not a positive integer multiple of the label step {step:g}")
            return k

        m_win = Steps(c.march_dt, "march_dt")
        s1 = Steps(c.label_dt, "label_dt")
        s2 = Steps(c.label_dt_phase2, "label_dt_phase2") if c.label_dt_phase2 else s1
        ratio = Steps(self.dt, "corpus dt")
        n_off = (m_win - 1) // max(s1, s2) + 1
        if self.CaseGroup() > n_off:
            raise ValueError(f"case_group {self.CaseGroup()} exceeds the {n_off} label offsets of a window")

        fine = torch.from_numpy(arr).float().to(self.device)
        # The labels must reproduce the corpus at its own save times (roll / colloc / val still read self.data)
        # // 코퍼스 저장 시점에서 같은 값이어야 한다
        diff = 0.0
        for s in range(0, self.n_case, 8192):
            ref = self.data[s:s + 8192, :, 1:5].to(self.device)
            diff = max(diff, float((fine[s:s + 8192, ::ratio] - ref).abs().max()))
        if not diff <= 1e-5:
            raise ValueError(f"label_path does not reproduce the corpus: max |label[::{ratio}] - corpus| = {diff:.3e}")

        self._label_dense = None                                                 # (N,) bool, None = no dense case
        n_dense = int(round(c.label_dense_frac * self.n_case))
        if n_dense > 0:
            if not c.label_score_path:
                raise ValueError("label_dense_frac > 0 needs label_score_path")
            sp = Path(c.label_score_path)
            score = torch.from_numpy(np.load(sp if sp.is_absolute() else Path(base_dir) / sp)).float()
            if tuple(score.shape) != (self.n_case,) or not bool(torch.isfinite(score).all()):
                raise ValueError(f"label_score_path {tuple(score.shape)}: expected ({self.n_case},) finite scores")
            dense = torch.zeros(self.n_case, dtype=torch.bool)
            dense[torch.topk(score, n_dense).indices] = True
            self._label_dense = dense.to(self.device)
        self._label = fine                 # (N, T_f, 4) [θ1, ω1, θ2, ω2]
        self._label_step = step            # dt_f
        self._label_m = m_win              # fine steps per window
        self._label_stride = (s1, s2)      # label stride in fine steps: phase 1, phase 2
        print(f"[label] {c.label_path} {tuple(arr.shape)}  dt_f={step:g}  M={m_win}  stride={s1}/{s2}  "
              f"dense={n_dense}/{self.n_case}  max|label-corpus|={diff:.1e}")

    def _LabelDraw(self, t_lo, bs):
        # Index draw of LabelBatch: case (G,), window start j (G,), label offsets m (G, K) in fine steps; G = bs // K
        # // LabelBatch의 인덱스 추첨 — 케이스, 창 시작 j, 라벨 오프셋 m (촘촘한 격자 단위)
        k = self.CaseGroup()
        g = bs // k
        dev = self.device
        ac = self.active_cases
        cases = ac[torch.randint(0, len(ac), (g,), device=dev)]
        m_win = self._label_m
        j = torch.full((g,), int(round(t_lo / self._label_step)), dtype=torch.long, device=dev)
        n_slide = int(round(self.dataCfg.slide_frac * g))
        if n_slide > 0:
            # slide_frac: the first n_slide groups (cases are iid, so this is a random share) start anywhere with
            #   j + M ≤ T_f − 1 // 앞 n_slide 그룹은 창이 데이터 구간 안에 드는 임의 시작점
            j[:n_slide] = torch.randint(0, self._label.shape[1] - m_win, (n_slide,), device=dev)
        # K distinct offsets from {m : 1 ≤ m ≤ M, (M − m) % s == 0} = {M − s·i, i = 0 .. (M−1)//s}
        s1, s2 = self._label_stride
        s = s1 if self._phase1 else s2
        if self._phase1 or s1 == s2 or self._label_dense is None:
            i = torch.rand(g, (m_win - 1) // s + 1, device=dev).argsort(dim=1)[:, :k]
        else:
            # phase 2: dense cases keep the phase-1 stride — per-group set size, the columns past it sort last
            # // phase 2: dense 케이스는 phase-1 간격 유지. 그룹마다 집합 크기가 달라 초과 열은 뒤로 민다
            s = torch.where(self._label_dense[cases], s1, s2).unsqueeze(1)                    # (G, 1)
            r = torch.rand(g, (m_win - 1) // min(s1, s2) + 1, device=dev)
            r = r + (torch.arange(r.shape[1], device=dev) > (m_win - 1) // s)
            i = r.argsort(dim=1)[:, :k]
        return cases, j, m_win - s * i

    def LabelBatch(self, t_lo, bs=None):
        # Data batch from the fine labels; same return as SegmentBatch: (feats, θ, ω, n_q), case-major groups of K
        #   rows that share one window-start state (K = CaseGroup(), 1 = iid rows). Label = fine[case, j + m],
        #   τ = m·dt_f, Δθ label against the window start (same convention as SegmentFrame). No SegmentFrame.
        # // 촘촘한 라벨에서 data 배치 추첨 — SegmentBatch와 같은 반환, 그룹(K행)은 창 시작 상태를 공유
        cases, j, m = self._LabelDraw(t_lo, bs or self.DataBatchSize())
        k = m.shape[1]
        ic = self._label[cases, j]                                         # (G, 4) window-start state
        lab = self._label[cases.unsqueeze(1), j.unsqueeze(1) + m]          # (G, K, 4)
        tau = (m.double() * self._label_step).float().reshape(-1, 1)
        theta_true = (lab[:, :, [0, 2]] - ic[:, None, [0, 2]]).reshape(-1, 2)
        omega_true = lab[:, :, [1, 3]].reshape(-1, 2)
        feats = self._BuildFeats(
            tau, self._ParamsAt(cases).repeat_interleave(k, dim=0), ic.repeat_interleave(k, dim=0)
        )
        return feats, theta_true, omega_true, k

    def DataDesc(self):
        # Data settings of the current phase for the [data] stdout line. label_dt = the phase's stride, dense =
        #   train-pool cases kept at the phase-1 stride. // [data] 로그용 현재 위상의 data 설정
        c = self.dataCfg
        out = f"batch={self.DataBatchSize()}"
        if not c.label_path:
            return out + " label=off"
        s1, s2 = self._label_stride
        s = s1 if self._phase1 else s2
        n_dense = 0
        if s != s1 and self._label_dense is not None:
            n_dense = int(self._label_dense[self.train_pool].sum())
        return out + f" label_dt={s * self._label_step:g} dense={n_dense} slide={c.slide_frac:g}"

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
