import torch
import torch.func as tf

class Loss:
    # --- torch.compile 게이트 // trainCfg.use_compile ---
    # 컴파일 단위는 "feats를 받아 손실 스칼라를 내는 슬라이스"다. feats 생성(_BuildFeats,
    # param_embed 포함)은 밖에 두는데, 그래야 매 스텝 바뀌는 인덱싱·gather가 그래프에 안 들어온다.
    # param_embed의 gradient는 경계에서 feats의 grad로 받아 eager로 이어져 흐른다 — 손실 없음.
    #   측정(2026-07-30): 물리 슬라이스 fwd+bwd 9.23 → 5.54 ms, 순수 forward 5.39 → 3.91 ms,
    #   peak VRAM 397 → 232 MB. mixed FP32/FP64 그래프에서 FP64는 dtype 그대로 보존된다
    #   (eager 대비 rel err 2e-16; FP32로 강등됐다면 2e-7이 나온다). // [[fp64-not-bottleneck]]
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

    def StateDerivs(self, feats):
        # net outputs [Δθ, ω]; one forward-mode pass gives [dΔθ/dt, dω/dt]
        # dΔθ/dt = dθ/dt since θ_IC is constant → kin residual (dΔθ/dt = ω) unchanged
        # feats: (M, feat_dim) ; col 0 = τ, col 1: = case-constant (IC + param_embed)
        # FP32 jvp: network params are FP32 anyway — no precision gain from FP64 functional_call
        # // FP32 jvp로 FP32 코어 활용 (검증: domega/dt floor 1.8e-5 << 잔차 O(100)). AngularAccel/GetEnergy만 FP64 유지
        t = feats[:, 0:1].float()
        rest = feats[:, 1:].float()
        ones = torch.ones_like(t)

        def f(tc, rc):                                    # (M, 1), (M, F-1) -> ((M,2), (M,2))
            x = torch.cat([tc, rc], dim=1)
            out = self(x)                                 # (M, 4) [θ1, θ2, ω1, ω2]
            return out[:, :2], out[:, 2:]                 # theta, omega

        # single jvp // d/dt [θ, ω] = [dθ/dt, dω/dt]
        # rest is a primal with an explicit zero tangent, not a closure capture: closure
        # capture gives it a lazy ZeroTensor tangent, and torch.compile's AOT backward
        # crashes writing into it ("ZeroTensors are immutable") on the adapter/dense-gate
        # graph. Materialized zeros are mathematically identical (case constants have
        # zero d/dτ). // rest를 클로저로 잡으면 ZeroTensor tangent가 생겨 compile 백워드가
        # 죽는다(finetune adapter 그래프에서만 발현). 명시적 0 tangent로 우회 — 수학적 동일.
        # weight tangent 경유의 두 번째 크래시 경로는 _Compiled의 allow_buffer_reuse=False가
        # 막는다 — 둘 다 필요하다 (trap 27).
        (theta, omega), (dtheta_dt, domega_dt) = tf.jvp(
            f, (t, rest), (ones, torch.zeros_like(rest))
        )
        return theta, omega, dtheta_dt, domega_dt

    def _KinTerm(self, omega, dtheta_dt):
        # kin = 두 헤드를 잇는 커플러 (ω := dθ/dτ). // structural glue, not a physics law
        # 분모 = omega_rms² (데이터에서 뽑은 고정 스케일). 학습 중 변하지 않는다.
        #   구버전은 mean(dθ/dτ²).detach()를 썼는데 이게 자기무효화였다: dθ/dτ가 노이즈로 커지면
        #   분모도 같이 커져서 kin → 1.0에 포화하고 gradient가 (17.24/1.49)² ≈ 133× 감쇠.
        #   즉 "고칠 대상이 분모를 부풀려 자기 억제를 끈다". 측정(2026-07-28,
        #   model/2026_07_24_10_33_11/best.pt): kin=0.978 고착, corr(ω, dθ/dτ)=0.12.
        #   고정 분모면 노이즈가 커질수록 kin도 커져 억제가 유지된다. // [[kin-derivative-noise]]
        # init grad 폭주 방지는 kin_ramp_epochs(0→1 램프) + RebalanceGradScales([0.1,10] clamp)가 담당.
        # 되돌리려면 이 한 줄만 원복. 단 kin 값의 단위가 바뀌므로 과거 로그와 직접 비교 불가.
        # _kin_detach=True(Phase 2): dθ/dτ 기준 고정, ω_head가 추종 → rollout 핸드오프 ω 드리프트 억제.
        # _kin_detach=False(Phase 1): 대칭 — 두 헤드가 함께 커플링 학습.
        scale = self.omega_rms.detach() ** 2 + self.dataCfg.kin_eps
        if getattr(self, "_kin_detach", False):
            return torch.mean((omega - dtheta_dt.detach()) ** 2) / scale
        return torch.mean((dtheta_dt - omega) ** 2) / scale

    def KinLoss(self, feats):
        return self._Compiled("kin", self._KinLossImpl)(feats)

    def _KinLossImpl(self, feats):
        # Phase 1 커플링 전용: kin만 (jvp, FP64 EOM/energy 없음) // cheap structural coupling
        _, omega, dtheta_dt, _ = self.StateDerivs(feats.float())
        return self._KinTerm(omega, dtheta_dt)

    def DataLoss(self, feats, theta_true, omega_true):
        return self._Compiled("data", self._DataLossImpl)(feats, theta_true, omega_true)

    def _DataLossImpl(self, feats, theta_true, omega_true):
        out = self(feats)                                 # (N, 4)
        theta, omega = out[:, :2], out[:, 2:]
        return torch.mean((theta - theta_true) ** 2) + torch.mean(
            (omega - omega_true) ** 2
        )

    def _PhysicsEnergySlice(self, feats, params, e0, ic_theta):
        return self._Compiled("phys", self._PhysicsEnergySliceImpl)(
            feats, params, e0, ic_theta
        )

    def _PhysicsEnergySliceImpl(self, feats, params, e0, ic_theta):
        # Single colloc slice // 콜로케이션 청크 1개분 물리+에너지 손실
        theta, omega, dtheta_dt, domega_dt = self.StateDerivs(feats.float())
        l_kin = self._KinTerm(omega, dtheta_dt)

        with torch.autocast(device_type=self.device_type, enabled=False):
            p64 = params.double()
            e0_64 = e0.double()
            th64 = ic_theta.double() + theta.double()
            om64 = omega.double()
            dom64 = domega_dt.double()
            f_eom = self.AngularAccel(
                th64[:, 0], om64[:, 0], th64[:, 1], om64[:, 1], p64
            )
            denom = f_eom.abs() + self.dataCfg.phys_eps
            l_phys = torch.mean(((dom64 - f_eom) / denom) ** 2).float()
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
        n_win = len(self.segments)
        outs = []
        for j, chunk in enumerate(case_idx.chunk(n_draws)):
            if chunk.numel() == 0:
                continue
            # k0를 윈도우 축으로 stratify — uniform 추첨이면 draw를 늘려도 같은 구간에 몰린다
            d_j = max(1, min(self.RollDepth(), n_win - 1))
            k0_j = min((j * n_win) // n_draws, n_win - d_j - 1)
            outs.append(self._RolloutPerCase(chunk, d_j, n_points, k0=k0_j))
        return self._RobustRollMean(torch.cat(outs))

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
        #   추첨 노이즈는 빠진다. detach 유지 — c는 상수 취급. // 2026-08-01
        k = self.trainCfg.roll_robust_k
        if k <= 0.0:
            return per_case.mean()                      # 0 → 기존 순수 평균 // opt-out
        med = per_case.detach().median()
        beta = getattr(self.trainCfg, "roll_median_ema", 0.0)
        if beta > 0.0:
            prev = self.__dict__.get("_roll_median")
            med = med if prev is None else beta * prev + (1.0 - beta) * med
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