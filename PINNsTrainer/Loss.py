import torch
import torch.func as tf

class Loss:
    # --- torch.compile 게이트 // trainCfg.use_compile ---
    # 컴파일 단위는 "feats를 받아 손실 스칼라를 내는 슬라이스"다. feats 생성(_BuildFeats,
    # param_embed 포함)은 밖에 두는데, 그래야 매 스텝 바뀌는 인덱싱·gather가 그래프에 안 들어온다.
    # param_embed의 gradient는 경계에서 feats의 grad로 받아 eager로 이어져 흐른다 — 손실 없음.
    #   mixed-dtype 그래프에서도 dtype은 그대로 보존된다. // [[fp64-not-bottleneck]]
    # dynamic=False: shape별 정적 그래프. 청크 크기가 몇 종류인지가 그래프 수 = 컴파일 대기시간이다
    #   (n_colloc이 colloc_chunk의 배수가 아니면 나머지 청크용으로 하나 더 생긴다).
    # 컴파일된 함수는 __dict__에 직접 담는다 — nn.Module 속성으로 들어가면 state_dict 오염 위험.
    def _Compiled(self, key, fn):
        if not getattr(self.trainCfg, "use_compile", False):
            return fn
        cache = self.__dict__.setdefault("_compiled_fns", {})
        if key not in cache:
            # allow_buffer_reuse=False: jvp 트레이스는 weight tangent를 _efficientzerotensor
            # 노드로 backward 그래프에 남기는데(adapter linear에서 발현), Inductor가 이
            # ZeroTensor 버퍼를 재사용 풀에 넣어 다른 mm의 out= 버퍼로 할당하면
            # "ZeroTensors are immutable"로 죽는다 (torch 2.12, 생성 코드에서
            # `buf90 = buf37  # reuse` 직접 확인). 입력 tangent를 zeros로 실체화한
            # StateDerivs 우회로는 못 막는 그래프 내부 접힘이라 재사용 자체를 끈다.
            # // ZeroTensor 버퍼 재사용 크래시 우회 (trap 27) — jvp 없는 data/ic 그래프에도
            # 걸리지만 비용은 alloc 재사용 손실뿐이고 캐싱 할로케이터가 흡수한다.
            torch._inductor.config.allow_buffer_reuse = False
            cache[key] = torch.compile(fn, dynamic=False)
        return cache[key]

    def StateDerivs(self, feats, n_q=1):
        # net outputs [Δθ, ω]; one forward-mode pass gives [dΔθ/dt, dω/dt]
        # n_q: xattn nets — case-major rows, n_q per case (token rows get zero τ-tangent inside forward)
        # dΔθ/dt = dθ/dt since θ_IC is constant → kin residual (dΔθ/dt = ω) unchanged
        # feats: (M, feat_dim) ; col 0 = τ, col 1: = case-constant (IC + param_embed)
        # FP32 jvp: network params are FP32 anyway — no precision gain from FP64 functional_call
        # // FP32 jvp로 FP32 코어 활용 (검증: domega/dt floor 1.8e-5 << 잔차 O(100)). AngularAccel/GetEnergy만 FP64 유지
        t = feats[:, 0:1].float()
        rest = feats[:, 1:].float()
        # FiLM (γ, β) outside the jvp: it depends on case columns only, so its τ-tangent
        # is identically 0 — but jvp would still push a materialized zero tangent through the cond
        # MLP (128 → 2·n_blocks·width: as many MACs as the whole trunk) and autograd would then
        # differentiate that dead tangent path too. Compute it once here and hand it in as a primal
        # with an explicit zero tangent — closure capture would give it a lazy ZeroTensor tangent
        # (trap 27). Same math: γ·SiLU(·) tangent = γ_t·SiLU + γ·SiLU'·pre_t with γ_t = 0.
        # // FiLM은 jvp 밖에서 1회, 0-tangent를 명시해 primal로 전달 (수학 동일, 죽은 경로 제거)
        # β = gb[:, :, 1] is a non-dense view; forward AD requires primal and tangent to share a
        # layout through view ops (unbind) — contiguous copy (N·n·width, memory-bound, ~0.1 ms).
        # // primal/tangent 레이아웃 불일치 → forward-AD 뷰 연산 assert. contiguous로 통일
        cond = tuple(c.contiguous() for c in self.CondOf(feats.float()))
        # xattn: window-token streams are query-independent → computed here once per case (outside
        # the jvp, still in the backward graph) and passed as zero-tangent primals. The forward-AD then runs on
        # the query rows only (token rows inside the jvp cost 7× on the kin term). // 토큰 상태는 jvp 밖에서
        n_c = len(cond)
        tok_hs = ()
        if getattr(self, "n_tok", 0) > 0:
            tok_hs = tuple(v.contiguous() for v in self.TokenStates(feats[::n_q].float(), tuple(c[::n_q] for c in cond)))

        def f(tc, rc, *rest_p):                           # (M, 1), (M, F-1), γ, β, [tok...] -> ((M,2), (M,2))
            x = torch.cat([tc, rc], dim=1)
            cd, th = rest_p[:n_c], rest_p[n_c:]
            out = self(x, cond=cd, n_q=n_q, tok_hs=th or None)   # (M, 4) [θ1, θ2, ω1, ω2]
            return out[:, :2], out[:, 2:]                 # theta, omega

        primals = (t, rest) + cond + tok_hs
        tangents = tuple(torch.ones_like(t) if i == 0 else torch.zeros_like(v) for i, v in enumerate(primals))

        # single jvp // d/dt [θ, ω] = [dθ/dt, dω/dt]
        # rest is a primal with an explicit zero tangent, not a closure capture: closure
        # capture gives it a lazy ZeroTensor tangent, and torch.compile's AOT backward
        # crashes writing into it ("ZeroTensors are immutable") on the adapter/dense-gate
        # graph. Materialized zeros are mathematically identical (case constants have
        # zero d/dτ). // rest를 클로저로 잡으면 ZeroTensor tangent가 생겨 compile 백워드가
        # 죽는다(finetune adapter 그래프에서만 발현). 명시적 0 tangent로 우회 — 수학적 동일.
        # weight tangent 경유의 두 번째 크래시 경로는 _Compiled의 allow_buffer_reuse=False가
        # 막는다 — 둘 다 필요하다 (trap 27).
        (theta, omega), (dtheta_dt, domega_dt) = tf.jvp(f, primals, tangents)
        return theta, omega, dtheta_dt, domega_dt

    @staticmethod
    def _TrimMean(per_row, q):
        # Mean over rows with the top-q fraction (by detached value) dropped from the gradient // 손실 상위 q 행 제외 평균
        #   kthvalue with a static k keeps the compiled graph shape-static.
        if q <= 0.0:
            return per_row.mean()
        d = per_row.detach().float()
        k = max(1, int(d.shape[0] * (1.0 - q)))
        keep = d <= torch.kthvalue(d, k).values
        return (per_row * keep).sum() / keep.sum()

    def _KinTerm(self, omega, dtheta_dt):
        # kin = 두 헤드를 잇는 커플러 (ω := dθ/dτ). // structural glue, not a physics law
        # 분모 = omega_rms² (데이터에서 뽑은 고정 스케일). 학습 중 변하지 않는다.
        #   구버전은 mean(dθ/dτ²).detach()를 썼는데 이게 자기무효화였다: dθ/dτ가 노이즈로 커지면
        #   분모도 같이 커져서 kin → 1.0에 포화하고 gradient가 (17.24/1.49)² ≈ 133× 감쇠.
        #   즉 "고칠 대상이 분모를 부풀려 자기 억제를 끈다".
        #   고정 분모면 노이즈가 커질수록 kin도 커져 억제가 유지된다. // [[kin-derivative-noise]]
        # init grad 폭주 방지는 kin_ramp_epochs(0→1 램프) + RebalanceGradScales([0.1,10] clamp)가 담당.
        # 되돌리려면 이 한 줄만 원복. 단 kin 값의 단위가 바뀌므로 과거 로그와 직접 비교 불가.
        # _kin_detach=True(Phase 2): dθ/dτ 기준 고정, ω_head가 추종 → rollout 핸드오프 ω 드리프트 억제.
        # _kin_detach=False(Phase 1): 대칭 — 두 헤드가 함께 커플링 학습.
        scale = self.omega_rms.detach() ** 2 + self.dataCfg.kin_eps
        q = self.trainCfg.trim_q_kin
        if q > 0.0:   # 행별 kin에서 상위 q 제외 // off면 아래 종전 경로 그대로
            ref = dtheta_dt.detach() if getattr(self, "_kin_detach", False) else dtheta_dt
            return self._TrimMean(((ref - omega) ** 2).mean(dim=1), q) / scale
        if getattr(self, "_kin_detach", False):
            return torch.mean((omega - dtheta_dt.detach()) ** 2) / scale
        return torch.mean((dtheta_dt - omega) ** 2) / scale

    def KinLoss(self, feats, n_q=1):
        return self._Compiled("kin", self._KinLossImpl)(feats, n_q)

    def _KinLossImpl(self, feats, n_q=1):
        # Phase 1 커플링 전용: kin만 (jvp, FP64 EOM/energy 없음) // cheap structural coupling
        _, omega, dtheta_dt, _ = self.StateDerivs(feats.float(), n_q)
        return self._KinTerm(omega, dtheta_dt)

    def DataLoss(self, feats, theta_true, omega_true, n_q=1):
        return self._Compiled("data", self._DataLossImpl)(feats, theta_true, omega_true, n_q)

    def _DataLossImpl(self, feats, theta_true, omega_true, n_q=1):
        out, stages = self(feats, Stages=True, n_q=n_q)   # (N, 4), [x̂_k]  (n_q: xattn 케이스 묶음)
        theta, omega = out[:, :2], out[:, 2:]
        q = self.trainCfg.trim_q_data
        if q > 0.0:   # 행별 (θ+ω) 손실 상위 q 제외, 단계 감독도 같은 방식 // trim_q_data
            def Row(x):
                return ((x[:, :2] - theta_true) ** 2).mean(dim=1) + ((x[:, 2:] - omega_true) ** 2).mean(dim=1)
            l_data = self._TrimMean(Row(out), q)
            for x_k in stages if self.stage_w > 0 else ():
                l_data = l_data + self.stage_w * self._TrimMean(Row(x_k), q)
            return l_data
        l_data = torch.mean((theta - theta_true) ** 2) + torch.mean(
            (omega - omega_true) ** 2
        )
        # 중간 단계 감독 (stage_at): x̂_k도 같은 데이터 손실, 가중 stage_w. 최종과 같은 스케일
        for x_k in stages if self.stage_w > 0 else ():
            l_data = l_data + self.stage_w * (torch.mean((x_k[:, :2] - theta_true) ** 2)
                                              + torch.mean((x_k[:, 2:] - omega_true) ** 2))
        return l_data

    def _PhysicsEnergySlice(self, feats, params, e0, ic_theta, n_q=1):
        return self._Compiled("phys", self._PhysicsEnergySliceImpl)(
            feats, params, e0, ic_theta, n_q
        )

    def _PhysicsEnergySliceImpl(self, feats, params, e0, ic_theta, n_q=1):
        # Single colloc slice // 콜로케이션 청크 1개분 물리+에너지 손실
        theta, omega, dtheta_dt, domega_dt = self.StateDerivs(feats.float(), n_q)
        l_kin = self._KinTerm(omega, dtheta_dt)

        # EOM·에너지도 FP32: 목표 상대잔차 1e-3~1e-5 대비 fp32 반올림×상쇄 증폭은 여유가 2자리 이상.
        #   변수명 *64는 FP64 시절 이름 그대로.
        with torch.autocast(device_type=self.device_type, enabled=False):
            p64 = params.float()
            e0_64 = e0.float()
            th64 = ic_theta.float() + theta.float()
            om64 = omega.float()
            dom64 = domega_dt.float()
            f_eom = self.AngularAccel(
                th64[:, 0], om64[:, 0], th64[:, 1], om64[:, 1], p64
            )
            denom = f_eom.abs() + self.dataCfg.phys_eps
            l_phys = self._TrimMean((((dom64 - f_eom) / denom) ** 2).mean(dim=1), self.trainCfg.trim_q_phys).float()
            state = torch.stack(
                [th64[:, 0], om64[:, 0], th64[:, 1], om64[:, 1]], dim=1
            )
            e = self.GetEnergy(state, p64)
            # max(|E0|,|E|) scale: E0≈0 zero-crossing 시 |E0|만 분모로 쓰면 폭발 // symmetric energy scale
            denom_e = torch.maximum(e0_64.abs(), e.abs()) + self.dataCfg.energy_eps
            l_energy = torch.mean(((e - e0_64) / denom_e) ** 2).float()

        return l_kin, l_phys, l_energy

    def _ICLossSlice(self, feats_ic, theta0_true, omega0_true):
        return self._Compiled("ic", self._ICLossSliceImpl)(
            feats_ic, theta0_true, omega0_true
        )

    def _ICLossSliceImpl(self, feats_ic, theta0_true, omega0_true):
        # τ=0 window-start match on a slice // IC 손실 청크
        out = self(feats_ic)
        theta, omega = out[:, :2], out[:, 2:]
        return torch.mean((theta - theta0_true) ** 2) + torch.mean(
            (omega - omega0_true) ** 2
        )

    def RollDepth(self):
        # depth 커리큘럼 추첨 — 학습 스텝과 grad balance가 같은 로직을 쓰게 한 곳으로 모음
        return int(torch.randint(1, max(1, self._roll_depth) + 1, (1,)).item())

    def RolloutLoss(self, case_idx, depth, n_points):
        # 스텝당 (k0, depth)를 roll_draws회 추첨해 케이스 예산을 나눠 담는다 // 분산 축소
        #   네트워크 evaluation 수는 그대로이므로 비용은 거의 동일. robust mean은 draw를 합친 뒤
        #   1회만 걸어 median 스케일이 draw별로 쪼개지지 않게 한다. // [[rollout-tail-cauchy-tradeoff]]
        n_draws = max(1, int(getattr(self.trainCfg, "roll_draws", 1)))
        if n_draws == 1 or case_idx.shape[0] < 2 * n_draws:
            return self._RobustRollMean(self._RolloutPerCase(case_idx, depth, n_points))
        return self._RobustRollMean(self._RolloutMergedDraws(case_idx, n_draws, n_points))

    def _RolloutMergedDraws(self, case_idx, n_draws, n_points):
        # All draws in one batch. Per-draw (k0, depth) become per-case vectors; the
        # pushforward runs max(depth) rounds over every case and freezes a case (torch.where) once
        # its own depth is reached. Rows are independent through the network, so each case sees
        # exactly the sequential-draw computation — only the batch it shares changes.
        # Sequential draws launched Σdepth (~40) tiny eager forwards per step = 22% of the step;
        # this is max(depth) (~15) compiled ones plus one grad window with per-case FiLM.
        # // draw 순차 → 단일 배치: 케이스별 depth/k0 벡터 + where 고정. 수학 동일, 런치 수 Σ→max
        dev = self.device
        md = self.dataCfg.march_dt
        dt = self.dt
        n_win = len(self.segments)
        depth_c, k0_c, max_depth = [], [], 0
        for j, chunk in enumerate(case_idx.chunk(n_draws)):
            if chunk.numel() == 0:
                continue
            # k0를 윈도우 축으로 stratify — uniform 추첨이면 draw를 늘려도 같은 구간에 몰린다
            d_j = max(1, min(self.RollDepth(), n_win - 1))
            k0_j = min((j * n_win) // n_draws, n_win - d_j - 1)
            depth_c.append(torch.full((chunk.numel(),), d_j, device=dev, dtype=torch.long))
            k0_c.append(torch.full((chunk.numel(),), k0_j, device=dev, dtype=torch.long))
            max_depth = max(max_depth, d_j)                        # python int — host sync 없음
        depth_c = torch.cat(depth_c)
        k0_c = torch.cat(k0_c)

        data, _ = self._ActiveSource()
        ci = case_idx.cpu() if data.device != self.device else case_idx
        i0_c = torch.round(k0_c.double() * md / dt).long()          # _RolloutPerCase의 int(round(k0*md/dt))
        state = data[ci, i0_c.to(ci.device)][:, [1, 2, 3, 4]]
        if state.device != self.device:
            state = state.to(self.device)
        params = self._ParamsAt(case_idx)

        tau_end = torch.full((1, 1), md, device=dev)
        alive = depth_c.unsqueeze(1)                               # (n,1) — round t 진행 조건 depth > t
        with torch.no_grad():
            for t in range(max_depth):
                out = self._RollWindow(case_idx, params, state, tau_end)   # (n,1,4) [Δθ,ω]
                last = out[:, -1, :]
                th_abs = state[:, [0, 2]] + last[:, :2]
                nxt = torch.stack([th_abs[:, 0], last[:, 2], th_abs[:, 1], last[:, 3]], dim=1)
                state = torch.where(alive > t, nxt, state)         # depth를 다 쓴 케이스는 고정
        state = state.detach()

        kg_c = k0_c + depth_c
        i_start_c = torch.round(kg_c.double() * md / dt).long()
        n_seg = int(round(md / dt))                                # 윈도우 격자 길이 — 전 케이스 동일
        off = torch.arange(1, n_seg + 1, device=dev)               # exclude τ=0, include end
        if off.numel() > n_points:
            sel = torch.linspace(0, off.numel() - 1, n_points, device=dev).round().long()
            off = off[sel]
        p = off.numel()
        grid = i_start_c.unsqueeze(1) + off.unsqueeze(0)           # (n,P)
        tau = (self.t_grid[grid] - self.t_grid[i_start_c].unsqueeze(1)).reshape(-1, 1).float()

        feats = self._BuildFeats(
            tau, params.repeat_interleave(p, dim=0), state.repeat_interleave(p, dim=0)
        )
        out = self._RollForward(feats, p)                          # (n,P,4) with grad, 케이스별 FiLM
        theta_pred = state[:, [0, 2]].unsqueeze(1) + out[:, :, :2]
        omega_pred = out[:, :, 2:]
        grid_idx = grid.cpu() if data.device != self.device else grid
        seg = data[ci.unsqueeze(1), grid_idx]                      # (n,P,5)
        if seg.device != self.device:
            seg = seg.to(self.device)
        theta_true = seg[:, :, [1, 3]]
        omega_true = seg[:, :, [2, 4]]
        return ((theta_pred - theta_true) ** 2).mean(dim=(1, 2)) + (
            (omega_pred - omega_true) ** 2
        ).mean(dim=(1, 2))

    def _RolloutPerCase(self, case_idx, depth, n_points, k0=None):
        # Pushforward (Brandstetter+ 2022): roll `depth` windows under no_grad to reach the
        # off-manifold IC the net actually produces, then one differentiable window matched to
        # true RK4 data — teaches the map to contract its own ω hand-off error back to truth.
        # // 예측 IC를 물려 만든 off-manifold 상태에서 다음 윈도우를 참 궤적으로 끌어오도록 학습
        # 반환은 per-case 벡터 — robust mean은 호출자가 draw를 합친 뒤 1회 건다.
        dev = self.device
        md = self.dataCfg.march_dt
        dt = self.dt
        n_win = len(self.segments)

        # grad window index kg = k0 + depth must stay inside the data range // 데이터 구간 안에 들어오게
        depth = max(1, min(depth, n_win - 1))
        if k0 is None:
            k0 = int(torch.randint(0, n_win - depth, (1,)).item())
        k0 = max(0, min(int(k0), n_win - depth - 1))

        # true IC at window k0 start // 참 시작 상태
        i0 = int(round(k0 * md / dt))
        data, _ = self._ActiveSource()
        if data.device != self.device:
            ci = case_idx.cpu()
        else:
            ci = case_idx
        state = data[ci, i0][:, [1, 2, 3, 4]]
        if state.device != self.device:
            state = state.to(self.device)
        params = self._ParamsAt(case_idx)

        # pushforward: net outputs Δθ; absolute θ_next = θ_start + Δθ_end // 감김수 누적 핸드오프
        tau_end = torch.full((1, 1), md, device=dev)
        with torch.no_grad():
            for _ in range(depth):
                out = self._RollWindow(case_idx, params, state, tau_end)   # (n,1,4) [Δθ,ω]
                last = out[:, -1, :]
                th_abs = state[:, [0, 2]] + last[:, :2]                    # 절대각 복원
                state = torch.stack(
                    [th_abs[:, 0], last[:, 2], th_abs[:, 1], last[:, 3]], dim=1
                )
        state = state.detach()                                     # fixed off-manifold input

        # differentiable window kg, matched to stored trajectory // grad 윈도우 = 참 데이터 매칭
        kg = k0 + depth
        i_start = int(round(kg * md / dt))
        i_end = int(round((kg + 1) * md / dt))
        grid = torch.arange(i_start + 1, i_end + 1, device=dev)    # exclude τ=0, include end
        if grid.numel() > n_points:
            sel = torch.linspace(0, grid.numel() - 1, n_points, device=dev).round().long()
            grid = grid[sel]
        tau = (self.t_grid[grid] - self.t_grid[i_start]).reshape(-1, 1).float()

        out = self._RollWindow(case_idx, params, state, tau)       # (n,P,4) [Δθ,ω] with grad
        theta_pred = state[:, [0, 2]].unsqueeze(1) + out[:, :, :2]  # 절대각 = 윈도우 시작각 + Δθ
        omega_pred = out[:, :, 2:]
        grid_idx = grid.cpu() if data.device != self.device else grid
        seg = data[ci][:, grid_idx]
        if seg.device != self.device:
            seg = seg.to(self.device)
        theta_true = seg[:, :, [1, 3]]
        omega_true = seg[:, :, [2, 4]]
        return ((theta_pred - theta_true) ** 2).mean(dim=(1, 2)) + (
            (omega_pred - omega_true) ** 2
        ).mean(dim=(1, 2))

    def _RobustRollMean(self, per_case):
        # Heavy-tail 억제: 발산 케이스 소수가 평균을 독식 → 그 스텝 gradient를 통째로 강탈.
        #   측정(88ep finetune): flip 에폭의 44%가 평균 roll>100, 중앙값은 ~7 — 3~5자리 tail.
        #   clip_grad_norm_은 전 항 합산 후 걸리므로 이때 data/phys/kin gradient가 사실상 소거됨.
        # Cauchy/Lorentzian: c·log1p(l/c) — 중앙값 근처는 준선형, l≫c는 log로 압축.
        #   d/dl = 1/(1+l/c) → 발산 케이스도 방향은 유지, 크기만 c/l로 감쇠 // 방향 보존 tail 감쇠
        # c는 배치 median의 배수(detach) → 하이퍼파라미터 없이 학습 진행에 따라 자동 축소.
        # c의 median EMA(roll_median_ema>0): 스케일 자체가 스텝마다 흔들리면 그것도 노이즈원이라
        #   같은 손실값이 스텝마다 다른 gradient 크기를 갖는다. EMA면 c가 학습 진행은 따라가되
        #   추첨 노이즈는 빠진다. detach 유지 — c는 상수 취급.
        k = self.trainCfg.roll_robust_k
        if k <= 0.0:
            return per_case.mean()                      # 0 → 기존 순수 평균 // opt-out
        # nanmedian + skip EMA update when med is non-finite: torch.median propagates NaN, and one diverged
        #   draw would write NaN into _roll_median → c=NaN → every later step skipped as "[NaN] roll" with
        #   weights frozen. where → no host sync.
        # // NaN draw는 median에서 제외, med 비유한이면 EMA는 직전 값 유지 — 1스텝 오염이 영구 교착이 됐던 버그
        med = per_case.detach().nanmedian()
        beta = getattr(self.trainCfg, "roll_median_ema", 0.0)
        if beta > 0.0:
            prev = self.__dict__.get("_roll_median")
            if prev is not None:
                med = torch.where(torch.isfinite(med), beta * prev + (1.0 - beta) * med, prev)
            self.__dict__["_roll_median"] = med.detach()
        c = k * med + 1e-12
        kind = getattr(self.trainCfg, "roll_robust_kind", "cauchy")
        if kind == "huber":
            # Huber(√l, √δ)를 손실값 l에 직접 쓴 형태: l ≤ δ는 그대로, 초과는 2√(δl) − δ.
            #   d/dl = √(δ/l) — Cauchy의 c/l보다 tail gradient가 √배 느리게 감쇠 →
            #   발산 케이스의 방향 신호를 더 보존한다. // [[rollout-tail-cauchy-tradeoff]] Huber 전환
            # clamp_min(c): where는 미선택 분기에도 gradient를 흘려 sqrt(0)의 inf가 NaN이 됨
            robust = torch.where(
                per_case <= c, per_case,
                2.0 * torch.sqrt(c * per_case.clamp_min(c)) - c,
            )
            return robust.mean()
        return (c * torch.log1p(per_case / c)).mean()