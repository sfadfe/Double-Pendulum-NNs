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

    def CollocationForBalance(self, t_lo, t_hi):
        # Finetune: fixed colloc_cases pool; pretrain: active_cases LHS // B 측정용 콜로케이션 meta
        if getattr(self, "_colloc_inited", False):
            return self.GetCollocMeta(t_lo, t_hi)
        return self.SampleCollocation(t_lo, t_hi)

    def _LossChunk(self):
        return self.dataCfg.batch_size

    def _GradNormOfLoss(self, loss):
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=False)
        gnorm = self._GlobalGradNorm()
        self.optimizer.zero_grad(set_to_none=True)
        return gnorm

    def _BackwardPhysicsChunks(self, meta, metric_out, losses_out):
        # Chunked colloc: rebuild feats per chunk // 청크마다 feats 새로 구성
        n = meta["tau"].shape[0]
        chunk = self._LossChunk()
        lam = self.LambdaAt()
        kin_acc = phys_acc = en_acc = 0.0
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            sl = slice(s, e)
            feats = self._BuildFeats(meta["tau"][sl], meta["params"][sl], meta["ic_state"][sl])
            ic_theta = meta["ic_state"][sl, [0, 2]]
            lk, lp, le = self._PhysicsEnergySlice(
                feats, meta["params"][sl], meta["e0"][sl], ic_theta
            )
            combo = (
                lam["kin"] * self.grad_scale["kin"] * lk
                + lam["phys"] * self.grad_scale["phys"] * lp
                + lam["energy"] * self.grad_scale["energy"] * le
            ) * ((e - s) / n)
            combo.backward(retain_graph=False)
            kin_acc += float(lk.detach()) * ((e - s) / n)
            phys_acc += float(lp.detach()) * ((e - s) / n)
            en_acc += float(le.detach()) * ((e - s) / n)
            metric_out[0] += float(combo.detach())
        losses_out["kin"] = kin_acc
        losses_out["phys"] = phys_acc
        losses_out["energy"] = en_acc

    def _BackwardICChunks(self, ic_parts, metric_out, losses_out):
        chunk = self._LossChunk()
        lam = self.LambdaAt()
        ic_acc = 0.0
        total_n = sum(p["ic_state"].shape[0] for p in ic_parts)
        for part in ic_parts:
            n = part["ic_state"].shape[0]
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                sl = slice(s, e)
                feats = self._BuildFeats(
                    part["tau"][sl], part["params"][sl], part["ic_state"][sl]
                )
                lic = self._ICLossSlice(
                    feats, part["theta0"][sl], part["omega0"][sl]
                )
                frac = (e - s) / total_n
                weighted = lam["ic"] * self.grad_scale["ic"] * lic * frac
                weighted.backward(retain_graph=False)
                ic_acc += float(lic.detach()) * frac
                metric_out[0] += float(weighted.detach())
        losses_out["ic"] = ic_acc

    def BackwardDataIC(self, batch, ic_parts):
        # Peak VRAM: data then IC, one graph each // data·IC 순차 backward
        self.optimizer.zero_grad(set_to_none=True)
        lam = self.LambdaAt()
        losses = {}
        metric_box = [0.0]

        l_data = self.DataLoss(*batch)
        w_data = lam["data"] * self.grad_scale["data"] * l_data
        w_data.backward(retain_graph=False)
        losses["data"] = float(l_data.detach())
        metric_box[0] += float(w_data.detach())

        self._BackwardICChunks(ic_parts, metric_box, losses)
        return metric_box[0], losses

    def BackwardAll(self, batch, colloc_meta, ic_parts, roll_loss=None):
        # Peak VRAM: data → physics chunks → IC chunks → roll // 항목별 순차 backward
        self.optimizer.zero_grad(set_to_none=True)
        lam = self.LambdaAt()
        losses = {}
        metric_box = [0.0]

        l_data = self.DataLoss(*batch)
        w_data = lam["data"] * self.grad_scale["data"] * l_data
        w_data.backward(retain_graph=False)
        losses["data"] = float(l_data.detach())
        metric_box[0] += float(w_data.detach())

        self._BackwardPhysicsChunks(colloc_meta, metric_box, losses)
        self._BackwardICChunks(ic_parts, metric_box, losses)

        if roll_loss is not None:
            w_roll = lam["roll"] * self.roll_ramp * roll_loss
            w_roll.backward(retain_graph=False)
            losses["roll"] = float(roll_loss.detach())
            metric_box[0] += float(w_roll.detach())

        return metric_box[0], losses

    def RebalanceGradScales(self, n_batches=3, eps=1e-12):
        # B: 항목별 독립 forward-backward로 grad 노름 측정 // retain_graph 없이 peak VRAM 절약
        self.train()
        sums = {"data": 0.0, "kin": 0.0, "phys": 0.0, "energy": 0.0, "ic": 0.0}
        cnt = 0
        chunk = self._LossChunk()
        for _ in range(n_batches):
            for t_lo, t_hi in self.segments:
                frame = self.SegmentFrame(t_lo, t_hi)
                batch = self.SegmentBatch(frame)
                ic_parts = self.ICSamplesRaw()
                colloc_meta = self.CollocationForBalance(t_lo, t_hi)

                sums["data"] += self._GradNormOfLoss(self.DataLoss(*batch))

                e_col = min(chunk, colloc_meta["tau"].shape[0])
                sl = slice(0, e_col)
                col_feats = self._BuildFeats(
                    colloc_meta["tau"][sl], colloc_meta["params"][sl], colloc_meta["ic_state"][sl]
                )
                ic_theta = colloc_meta["ic_state"][sl, [0, 2]]
                p_sl = colloc_meta["params"][sl]
                e0_sl = colloc_meta["e0"][sl]
                for key, idx in (("kin", 0), ("phys", 1), ("energy", 2)):
                    feats_i = self._BuildFeats(
                        colloc_meta["tau"][sl], colloc_meta["params"][sl], colloc_meta["ic_state"][sl]
                    )
                    parts = self._PhysicsEnergySlice(feats_i, p_sl, e0_sl, ic_theta)
                    sums[key] += self._GradNormOfLoss(parts[idx])

                part0 = ic_parts[0]
                e_ic = min(chunk, part0["ic_state"].shape[0])
                ic_feats = self._BuildFeats(
                    part0["tau"][:e_ic], part0["params"][:e_ic], part0["ic_state"][:e_ic]
                )
                sums["ic"] += self._GradNormOfLoss(
                    self._ICLossSlice(ic_feats, part0["theta0"][:e_ic], part0["omega0"][:e_ic])
                )

                cnt += 1
                del colloc_meta, batch, ic_parts, frame
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
    def _InitCollocPools(self):
        if not hasattr(self, "_colloc_pools"):
            self._colloc_pools = {}

    def _CollocCasePool(self):
        # RAR 풀 전용 case 집합; 미설정 시 active_cases 폴백 // finetune은 colloc_cases 고정
        pool = getattr(self, "colloc_cases", None)
        if pool is None:
            return self.active_cases
        return pool

    def _LHSTau(self, n, span):
        dev = self.device
        edges = torch.linspace(0.0, 1.0, n + 1, device=dev)
        u = edges[:-1] + torch.rand(n, device=dev) * (1.0 / n)
        u = u[torch.randperm(n, device=dev)]
        return (u * span).reshape(-1, 1)

    def _SampleCollocMeta(self, t_lo, t_hi, case_pool, n=None):
        n = n or self.collocCfg.n_colloc
        span = t_hi - t_lo
        tau = self._LHSTau(n, span)

        local_idx = torch.randint(0, len(case_pool), (n,), device=self.device)
        pool = case_pool.to(self.device) if case_pool.device != self.device else case_pool
        case_idx = pool[local_idx]

        i0 = self._WindowStartIdx(t_lo)
        data, params_src = self._ActiveSource()
        if data.device != self.device:
            ci = case_idx.cpu()
        else:
            ci = case_idx
        ic_base = data[ci, i0][:, [1, 2, 3, 4]]
        if ic_base.device != self.device:
            ic_base = ic_base.to(self.device)
        params = params_src[ci]
        if params.device != self.device:
            params = params.to(self.device)
        # off-manifold IC 섭동 (Phase 3, 에폭 램프) // ic_base 보존해 RAR 재충전 시 재섭동
        sigma = getattr(self, "_colloc_ic_sigma", 0.0)
        ic_state = ic_base + sigma * torch.randn_like(ic_base) if sigma > 0.0 else ic_base
        e0 = self.GetEnergy(ic_state.double(), params.double()).detach()
        return {
            "tau": tau,
            "case_idx": case_idx,
            "ic_base": ic_base,
            "ic_state": ic_state,
            "params": params,
            "e0": e0,
            "step_birth": getattr(self, "_global_step", 0),
        }

    def _MetaToColloc(self, meta):
        # meta["params"]로 직접 feats 구성 — _replay_active 전역 상태 의존 제거 // params mismatch 버그 수정
        feats = self._BuildFeats(meta["tau"], meta["params"], meta["ic_state"])
        ic_theta = meta["ic_state"][:, [0, 2]]
        return feats, meta["params"], meta["e0"], ic_theta

    def SetupCollocCases(self, n_colloc=None, flip_bias=0.0):
        # flip finetune: colloc_cases 고정(stratified) + RAR 풀 초기화
        self._InitCollocPools()
        pool = self.train_pool
        n = n_colloc or self.trainCfg.n_colloc_cases or self.max_cases
        n = min(int(n), len(pool))

        if flip_bias > 0.0 and getattr(self, "is_flip", None) is not None:
            flip_mask = self.is_flip[pool]
            flip_ids = pool[flip_mask]
            nf_ids = pool[~flip_mask]
            n_flip = min(len(flip_ids), max(1, int(round(flip_bias * n))))
            n_nf = min(len(nf_ids), n - n_flip)
            n_flip = min(len(flip_ids), n - n_nf)
            fi = flip_ids[torch.randperm(len(flip_ids), device=self.device)[:n_flip]]
            ni = nf_ids[torch.randperm(len(nf_ids), device=self.device)[:n_nf]]
            chosen = torch.cat([fi, ni])
            self.colloc_cases = chosen[torch.randperm(len(chosen), device=self.device)]
        else:
            perm = torch.randperm(len(pool), device=self.device)[:n]
            self.colloc_cases = pool[perm]

        for t_lo, t_hi in self.segments:
            self.InitCollocPool(t_lo, t_hi)
        self._colloc_inited = True

    def InitCollocPool(self, t_lo, t_hi):
        self._InitCollocPools()
        key = (t_lo, t_hi)
        self._colloc_pools[key] = self._SampleCollocMeta(
            t_lo, t_hi, self._CollocCasePool()
        )

    def GetCollocMeta(self, t_lo, t_hi, case_pool=None):
        # Collocation meta (no feats) — feats built per chunk at train time // 학습 시 청크별 feats 생성
        if case_pool is not None:
            return self._SampleCollocMeta(t_lo, t_hi, case_pool)
        self._InitCollocPools()
        key = (t_lo, t_hi)
        if key not in self._colloc_pools:
            self.InitCollocPool(t_lo, t_hi)
        return self._colloc_pools[key]

    def GetColloc(self, t_lo, t_hi, case_pool=None):
        # Legacy feats tuple — RAR / diagnostics only // RAR·진단용 feats 튜플
        return self._MetaToColloc(self.GetCollocMeta(t_lo, t_hi, case_pool=case_pool))

    def RefineCollocPool(self, t_lo, t_hi):
        key = (t_lo, t_hi)
        pool = self._colloc_pools[key]
        span = t_hi - t_lo
        feats, params, e0, ic_theta = self._MetaToColloc(pool)
        residual = self.PhysicsResidualPerPoint(feats, params, e0, ic_theta)
        self._colloc_pools[key] = self.RARUpdate(pool, residual, span)
        self._colloc_pools[key]["step_birth"] = getattr(self, "_global_step", 0)
        p50 = float(torch.quantile(residual, 0.5))
        p90 = float(torch.quantile(residual, 0.9))
        return p50, p90, int(pool["tau"].shape[0])

    def BuildSegments(self, t_min, t_max):
        # Windows of duration march_dt; the network sees relative time τ∈[0,march_dt] // 비겹침 마칭 윈도우
        march_dt = self.dataCfg.march_dt
        n_win = max(1, int(round((t_max - t_min) / march_dt)))
        segs = [(t_min + k * march_dt, t_min + (k + 1) * march_dt) for k in range(n_win)]
        self.segments = segs
        return segs

    def SampleCollocation(self, t_lo, t_hi, n=None):
        return self._SampleCollocMeta(t_lo, t_hi, self.active_cases, n=n)

    @torch.no_grad()
    def MarchRollout(self, case_idx, n_windows):
        # True time-marching: window k's predicted end-state feeds window k+1's IC // 윈도우 연쇄 외삽
        # Returns times (K,), theta_pred (n,K,2)=[θ1,θ2], omega_pred (n,K,2)=[ω1,ω2]
        dev = self.device
        dt = self.dt
        march_dt = self.dataCfg.march_dt
        steps = int(round(march_dt / dt))
        n = case_idx.shape[0]
        tau = (torch.arange(1, steps + 1, device=dev).double() * dt).float().reshape(-1, 1)

        data, _ = self._ActiveSource()
        if data.device != dev:
            ci = case_idx.cpu()
        else:
            ci = case_idx
        state = data[ci, 0][:, [1, 2, 3, 4]].clone()
        if state.device != dev:
            state = state.to(dev)
        params = self._ParamsAt(case_idx)
        th_out = [state[:, [0, 2]].unsqueeze(1)]                  # include t=0 point: (n,1,2)
        om_out = [state[:, [1, 3]].unsqueeze(1)]

        params_flat = params.repeat_interleave(steps, dim=0)      # (n*steps,4)
        tau_flat = tau.repeat(n, 1)                               # (n*steps,1)
        for _ in range(n_windows):
            ic_flat = state.repeat_interleave(steps, dim=0)       # (n*steps,4)
            feats = self._BuildFeats(tau_flat, params_flat, ic_flat)
            preds = []
            for s in range(0, n * steps, 4096):
                preds.append(self(feats[s : s + 4096]))
            out = torch.cat(preds).reshape(n, steps, 4)           # [Δθ1,Δθ2,ω1,ω2]
            # absolute θ = window-start θ + Δθ // 윈도우 시작각에 상대각 누적 → 감김수 자연 복원
            th_abs = state[:, [0, 2]].unsqueeze(1) + out[:, :, :2]
            th_out.append(th_abs)
            om_out.append(out[:, :, 2:])
            # next IC = end-of-window absolute state, reordered to [θ1,ω1,θ2,ω2] // 다음 윈도우 IC
            last_om = out[:, -1, 2:]
            last_th = th_abs[:, -1, :]
            state = torch.stack([last_th[:, 0], last_om[:, 0], last_th[:, 1], last_om[:, 1]], dim=1)

        theta = torch.cat(th_out, dim=1)                          # (n, n_windows*steps+1, 2)
        omega = torch.cat(om_out, dim=1)
        times = torch.arange(0, n_windows * steps + 1, device=dev).double() * dt
        return times, theta, omega

    def _RollWindow(self, case_idx, params_raw, state, tau):
        # One flow-map window: predict [Δθ, ω] at relative times tau from IC `state` // 단일 윈도우 예측
        # state: (n,4) [θ1,ω1,θ2,ω2] ; params_raw: (n,4) ; tau: (P,1) ; returns (n,P,4) [Δθ1,Δθ2,ω1,ω2]
        n, p = case_idx.shape[0], tau.shape[0]
        params_flat = params_raw.repeat_interleave(p, dim=0)
        tau_flat = tau.repeat(n, 1)
        ic_flat = state.repeat_interleave(p, dim=0)
        feats = self._BuildFeats(tau_flat, params_flat, ic_flat)
        return self(feats).reshape(n, p, 4)

    def RARUpdate(self, pool, residual, span):
        # 잔차 하위 제거 + 상위 근방 τ jitter 재충전, N 유지 // 풀 메타 in/out
        c = self.collocCfg
        n = pool["tau"].shape[0]
        order = torch.argsort(residual, descending=True)
        n_top = max(1, int(c.rar_top_frac * n))
        n_bot = max(1, int(c.rar_bot_frac * n))
        n_bot = min(n_bot, n - 1)

        keep = order[: n - n_bot]
        tau_k = pool["tau"][keep]
        case_k = pool["case_idx"][keep]
        base_k = pool["ic_base"][keep]
        ic_k = pool["ic_state"][keep]
        params_k = pool["params"][keep]
        e0_k = pool["e0"][keep]

        top = order[:n_top]
        reps = (n_bot + n_top - 1) // n_top
        src = top.repeat(reps)[:n_bot]

        delta = c.rar_jitter_frac * self.dataCfg.march_dt
        jitter = (torch.rand(n_bot, 1, device=tau_k.device) - 0.5) * 2.0 * delta
        tau_new = (pool["tau"][src] + jitter).clamp(0.0, span)

        # 재충전 점은 현재 sigma로 ic_base 재섭동 + e0 재계산 // ic_sigma 램프 활성화
        base_new = pool["ic_base"][src]
        params_new = pool["params"][src]
        sigma = getattr(self, "_colloc_ic_sigma", 0.0)
        ic_new = base_new + sigma * torch.randn_like(base_new) if sigma > 0.0 else base_new
        e0_new = self.GetEnergy(ic_new.double(), params_new.double()).detach()

        return {
            "tau": torch.cat([tau_k, tau_new], dim=0),
            "case_idx": torch.cat([case_k, pool["case_idx"][src]], dim=0),
            "ic_base": torch.cat([base_k, base_new], dim=0),
            "ic_state": torch.cat([ic_k, ic_new], dim=0),
            "params": torch.cat([params_k, params_new], dim=0),
            "e0": torch.cat([e0_k, e0_new], dim=0),
            "step_birth": pool.get("step_birth", 0),
        }