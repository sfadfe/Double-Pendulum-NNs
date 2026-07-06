import torch
import math


class LambdaBalance:
    
    def InitLambda(self):
        c = self.trainCfg
        self.grad_scale = {"data": 1.0, "kin": 1.0, "phys": 1.0, "energy": 1.0, "ic": 1.0}
        self.roll_ramp = 1.0               # rollout 손실 0→1 램프 계수 (grad_scale 슬롯 재활용 폐기 → 명시 분리) // explicit ramp
        self._phys_balanced = False        # Phase 2 진입 시 첫 RebalanceGradScales로 True
        self._colloc_ic_sigma = 0.0        # 물리 콜로케이션 IC 섭동 폭 (Phase 3, 매 에폭 램프) // off-manifold
        self._base_lambda = {
            "ic": c.lambda_ic, "data": c.lambda_data, "kin": c.lambda_kin,
            "phys": c.lambda_phys, "energy": c.lambda_energy, "roll": c.lambda_roll,
        }
        self._lambda_relo = dict(self._base_lambda)  # ReLoBRaLo current weights // 현재 동적 가중치
        self._loss_prev   = {}                        # previous epoch losses // 이전 에폭 손실 (변화율 계산용)

    def _GlobalGradNorm(self):
        # 전 파라미터 grad의 global L2 노름 (단일 sync) // global grad norm over all params
        parts = [p.grad.detach().pow(2).sum() for p in self.parameters() if p.grad is not None]
        if not parts:
            return 0.0
        return float(torch.sqrt(torch.stack(parts).sum()))

    def RebalanceGradScales(self, n_batches=3, eps=1e-12):
        # B: 각 손실항 grad 노름을 측정해 data(anchor) 기준으로 균등화 // Wang et al. 2021 LR-annealing
        # backward K회 × (segments·n_batches)는 무거움 → grad_balance_every마다만 호출
        self.train()
        sums = {"data": 0.0, "kin": 0.0, "phys": 0.0, "energy": 0.0, "ic": 0.0}
        cnt = 0
        ic = self.ICSamples()
        for _ in range(n_batches):
            for t_lo, t_hi in self.segments:
                feats_full, theta_t, omega_t, _ = self.SegmentSamples(t_lo, t_hi)
                n_total = feats_full.shape[0]
                bs = min(self.dataCfg.batch_size, n_total)
                idx = torch.randint(0, n_total, (bs,), device=self.device)
                batch = (feats_full[idx], theta_t[idx], omega_t[idx])
                colloc = self.SampleCollocation(t_lo, t_hi)
                losses = self.ComputeAllLosses(batch, colloc, ic)
                for k, v in losses.items():
                    self.optimizer.zero_grad(set_to_none=True)
                    v.backward(retain_graph=True)
                    sums[k] += self._GlobalGradNorm()
                cnt += 1
                del losses, colloc, batch, feats_full, theta_t, omega_t
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        g = {k: sums[k] / cnt for k in sums}        # 평균 grad 노름
        anchor = g["data"] + eps
        for k in g:
            # EMA 교체: 계단 외란 제거 // smooth update instead of hard replace
            # clamp [0.1, 10]: energy처럼 grad 작은 항 scale 폭발 방지 // prevent blowup on weak-grad terms
            new_scale = max(0.1, min(10.0, anchor / (g[k] + eps)))
            self.grad_scale[k] = 0.7 * self.grad_scale[k] + 0.3 * new_scale
        self._phys_balanced = True
        return g

    def LambdaAt(self):
        return dict(self._lambda_relo)

    def UpdateReLoBRaLo(self, avg_losses):
        # ReLoBRaLo: 수렴 느린 항에 가중치 증가, 빠른 항에 감소 // Bischof & Kraus 2021
        c     = self.trainCfg
        alpha = c.relobralo_alpha
        tau   = c.relobralo_tau

        active = {k: v for k, v in avg_losses.items() if not math.isnan(v)}
        n = len(active)
        if n == 0:
            return  

        # rho_k = L_k(t) / L_k(t-1): 클수록 수렴 안 됨 → softmax에서 높은 가중치 // stagnating loss gets more weight
        rho = {}
        for k, v in active.items():
            prev = self._loss_prev.get(k)
            rho[k] = v / (prev + 1e-30) if (prev is not None and prev > 1e-30) else 1.0

        # Numerically stable softmax // log-sum-exp trick
        vals  = [rho[k] / tau for k in active]
        max_v = max(vals)
        exps  = [math.exp(v - max_v) for v in vals]
        denom = sum(exps)

        # 중립 상태(모두 같은 속도 수렴)에서 base_lambda로 복원 // neutral → base lambda recovered
        lo, hi = self.trainCfg.lambda_min, self.trainCfg.lambda_max
        for i, k in enumerate(active):
            hat_k = n * exps[i] / denom * self._base_lambda[k]
            relo = (1.0 - alpha) * hat_k + alpha * self._lambda_relo[k]
            self._lambda_relo[k] = min(max(relo, lo), hi)   # clamp → 폭주 방지 (B가 절대 스케일 담당)

        for k, v in active.items():
            self._loss_prev[k] = v

    def NormalizedTotal(self, losses):
        # 유효 가중치 w_k = λ_relo_k · grad_scale_k (B 절대스케일 × ReLoBRaLo 상대nudge)
        # roll은 grad_scale 대신 roll_ramp (0→1 선형 램프) // explicit ramp, not grad_scale slot abuse
        lam = self.LambdaAt()
        total = 0.0
        for k, v in losses.items():
            scale = self.roll_ramp if k == "roll" else self.grad_scale[k]
            total = total + lam[k] * scale * v
        return total


class TimeMarching:
    # Non-overlapping march windows + LHS/RAR 콜로케이션
    def BuildSegments(self, t_min, t_max):
        # Windows of duration march_dt; the network sees relative time τ∈[0,march_dt] // 비겹침 마칭 윈도우
        march_dt = self.dataCfg.march_dt
        n_win = max(1, int(round((t_max - t_min) / march_dt)))
        segs = [(t_min + k * march_dt, t_min + (k + 1) * march_dt) for k in range(n_win)]
        self.segments = segs
        return segs

    def SampleCollocation(self, t_lo, t_hi, n=None):
        n = n or self.collocCfg.n_colloc
        dev = self.device
        span = t_hi - t_lo

        # Latin Hypercube on relative time τ∈[0, march_dt] // τ축 라틴 하이퍼큐브
        edges = torch.linspace(0.0, 1.0, n + 1, device=dev)
        u = edges[:-1] + torch.rand(n, device=dev) * (1.0 / n)
        u = u[torch.randperm(n, device=dev)]
        tau = (u * span).reshape(-1, 1)

        local_idx = torch.randint(0, len(self.active_cases), (n,), device=dev)
        case_idx = self.active_cases[local_idx]

        i0 = self._WindowStartIdx(t_lo)
        ic_state = self.data[case_idx, i0][:, [1, 2, 3, 4]]        # (n, 4) window-start state
        # Phase 3: 물리 콜로케이션 전용 IC 섭동 — rollout이 밟을 off-manifold 이웃을 물리로 학습 // data/ic 손실 IC는 불변
        sigma = getattr(self, "_colloc_ic_sigma", 0.0)
        if sigma > 0.0:
            ic_state = ic_state + sigma * torch.randn_like(ic_state)
        feats = self._BuildFeats(tau, case_idx, ic_state)
        params = self.params_raw[case_idx]

        # e0 in FP64 from the window-start state // 윈도우 기준 에너지 (FP64)
        e0 = self.GetEnergy(ic_state.double(), params.double()).detach()
        return feats, params, e0

    @torch.no_grad()
    def MarchRollout(self, case_idx, n_windows):
        # True time-marching: window k's predicted end-state feeds window k+1's IC // 윈도우 연쇄 외삽
        # Returns times (K,), theta_pred (n,K,2)=[θ1,θ2], omega_pred (n,K,2)=[ω1,ω2]
        dev = self.device
        dt = self.dt
        march_dt = self.dataCfg.march_dt
        steps = int(round(march_dt / dt))
        n = case_idx.shape[0]
        tau = (torch.arange(1, steps + 1, device=dev).double() * dt).float().reshape(-1, 1)  # (steps,1)

        state = self.data[case_idx, 0][:, [1, 2, 3, 4]].clone()   # (n,4) [θ1,ω1,θ2,ω2] @ t=0
        th_out = [state[:, [0, 2]].unsqueeze(1)]                  # include t=0 point: (n,1,2)
        om_out = [state[:, [1, 3]].unsqueeze(1)]

        case_flat = case_idx.repeat_interleave(steps)             # (n*steps,)
        tau_flat = tau.repeat(n, 1)                               # (n*steps,1)
        for _ in range(n_windows):
            ic_flat = state.repeat_interleave(steps, dim=0)       # (n*steps,4)
            feats = self._BuildFeats(tau_flat, case_flat, ic_flat)
            preds = []
            for s in range(0, n * steps, 4096):
                preds.append(self(feats[s : s + 4096]))
            out = torch.cat(preds).reshape(n, steps, 4)           # [θ1,θ2,ω1,ω2]
            th_out.append(out[:, :, :2])
            om_out.append(out[:, :, 2:])
            # next IC = end-of-window state, reordered to [θ1,ω1,θ2,ω2] // 다음 윈도우 IC
            last = out[:, -1, :]
            state = torch.stack([last[:, 0], last[:, 2], last[:, 1], last[:, 3]], dim=1)

        theta = torch.cat(th_out, dim=1)                          # (n, n_windows*steps+1, 2)
        omega = torch.cat(om_out, dim=1)
        times = torch.arange(0, n_windows * steps + 1, device=dev).double() * dt
        return times, theta, omega

    def _RollWindow(self, case_idx, state, tau):
        # One flow-map window: predict state at relative times tau from IC `state` // 단일 윈도우 예측
        # state: (n,4) [θ1,ω1,θ2,ω2] ; tau: (P,1) ; returns (n,P,4) [θ1,θ2,ω1,ω2]
        n, p = case_idx.shape[0], tau.shape[0]
        case_flat = case_idx.repeat_interleave(p)
        tau_flat = tau.repeat(n, 1)
        ic_flat = state.repeat_interleave(p, dim=0)
        feats = self._BuildFeats(tau_flat, case_flat, ic_flat)
        return self(feats).reshape(n, p, 4)

    @staticmethod
    def _NextIC(out_last):
        # window-end output [θ1,θ2,ω1,ω2] -> next-window IC [θ1,ω1,θ2,ω2] // 핸드오프 재정렬
        return torch.stack(
            [out_last[:, 0], out_last[:, 2], out_last[:, 1], out_last[:, 3]], dim=1
        )

    def RARUpdate(self, feats, params, e0, residual):
        # 잔차 하위 제거 + 상위 근방 밀집, 총 포인트 수 유지
        c = self.collocCfg
        n = feats.shape[0]
        order = torch.argsort(residual, descending=True)
        n_top = int(c.rar_top_frac * n)
        n_bot = int(c.rar_bot_frac * n)

        keep = order[: n - n_bot]
        feats_k, params_k, e0_k = feats[keep], params[keep], e0[keep]

        # Jitter t near top-residual points to refill removed budget // 상위 근방 재샘플
        top = order[:n_top]
        reps = (n_bot + n_top - 1) // max(n_top, 1)
        src = top.repeat(reps)[:n_bot]
        jitter = (torch.rand(n_bot, 1, device=feats.device) - 0.5) * 2.0
        dt = self.t_grid[1] - self.t_grid[0]
        new_feats = feats[src].clone()
        new_feats[:, 0:1] = new_feats[:, 0:1] + jitter * dt

        feats_n = torch.cat([feats_k, new_feats], dim=0)
        params_n = torch.cat([params_k, params[src]], dim=0)
        e0_n = torch.cat([e0_k, e0[src]], dim=0)
        return feats_n, params_n, e0_n