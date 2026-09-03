import torch
import math


class LambdaBalance:
    
    def InitLambda(self):
        c = self.trainCfg
        self.grad_scale = {"data": 1.0, "kin": 1.0, "phys": 1.0, "energy": 1.0, "ic": 1.0, "roll": 1.0}
        self._roll_active = False          # roll_on 미러 — RebalanceGradScales가 roll 측정 여부 판단 // loop에서 갱신
        self._roll_depth = 1               # 현재 depth 커리큘럼 상한 — 밸런싱 측정도 같은 depth로 // loop에서 갱신
        self.roll_ramp = 1.0               # rollout 손실 0→1 램프 계수 (grad_scale 슬롯 재활용 폐기 → 명시 분리) // explicit ramp
        self.phys_ramp = 0.0               # Phase 2 phys/energy backward sigmoid ramp // loop에서 갱신
        self.kin_ramp = 1.0                # kin(헤드 커플러) 0→1 램프 — phys_ramp와 독립, Phase 1부터 상시 // loop에서 갱신
        self._kin_detach = False           # Phase 2에서 True → ω_head가 dθ/dτ 추종 (핸드오프 억제) // loop에서 갱신
        self._phys_balanced = False        # Phase 2 진입 시 첫 RebalanceGradScales로 True
        self._term_gnorm = {}              # 항별 clip 직전 grad 노름 EMA(0.9/0.1) // _FlushTerm 진단
        self._grad_acc = None              # 항별 clip 누적 버퍼 — 최초 1회 할당 후 zero_()로 재사용 // _GradAcc
        self._colloc_ic_sigma = 0.0        # 물리 콜로케이션 IC 섭동 폭 (Phase 3, 매 에폭 램프) // off-manifold
        # 고정 base 가중치 — 동적 조정은 grad_scale(RebalanceGradScales) 단독 담당.
        self._base_lambda = {
            "ic": c.lambda_ic, "data": c.lambda_data, "kin": c.lambda_kin,
            "phys": c.lambda_phys, "energy": c.lambda_energy, "roll": c.lambda_roll,
        }
        self.InitEMA(getattr(c, "ema_decay", 0.0))    # Polyak weight EMA — 평가/best.pt 안정화 (decay<=0 → 비활성)

    def IcTermOn(self):
        # hard_ic가 켜지면 Δθ(0)=0·ω(0)=ω_IC가 구조적 항등식이라 IC 손실이 정확히 0 —
        # 항을 계산할 이유가 없다(스텝 비용·grad_scale 슬롯 회수). // NetCfg.hard_ic
        return getattr(self.netCfg, "hard_ic", "off") == "off"

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
        # kin/phys/ic backward 청크 — data 미니배치와 분리 // decoupled from data batch to bound FP64 physics transient peak
        return self.dataCfg.colloc_chunk or self.dataCfg.batch_size

    def _PhysRamp(self):
        # kin/phys/energy backward scale (Phase 1 → 0) // training loop sets self.phys_ramp
        return self.phys_ramp

    # --- 항별 clip 후 누적 // per-term clip, then accumulate ---
    # BackwardAll은 항을 .grad에 순차 누적하고 loop_common이 끝에서 clip 1회를 건다(trap #8).
    # 그러면 한 항이 폭주할 때 clip 배율이 전 항에 똑같이 걸려 나머지가 지워진다.
    #   측정(2026-07-29, model/2026_07_28_03_39_08 ep3100~3140):
    #   유효 phys = scale 0.01 × ‖g‖ 1.0e2~2.9e3 = 1~29, 유효 data = 1.0 × 0.29 고정.
    #   grad_clip=1.0이므로 실제 적용되는 data gradient가 스텝마다 0.29~0.01로 29배 요동 —
    #   phys의 "평균 크기"가 아니라 "분산"이 data·roll·ic의 유효 LR을 무작위 감쇠시킨다.
    # 항별로 먼저 clip하면 폭주 항만 예산에 눌리고 나머지는 원래 크기를 유지한다.
    # roll은 이 모델의 목표(윈도우 핸드오프 오차 최소화)를 직접 담당하는 항이라
    # 예산 보장이 부수효과가 아니라 주목적이다. // roll must keep its share by design
    # 분리 단위는 {data, physics(kin+phys+energy combo), ic, roll} — physics는 청크당 combo 1회
    # backward라 셋을 쪼개면 jvp 비용이 3배가 된다. 폭주하는 건 그 combo 자체라 이 단위로 충분.
    def _TermClipBudget(self):
        # 예산 = grad_clip 고정 — term_grad_clip 옵션은 전 config가 0(=grad_clip 사용)으로
        # 수렴해 제거 (2026-08-19). 항별 clip 자체는 하한 제거의 전제라 끄는 경로도 없앴다.
        return self.trainCfg.grad_clip

    # --- rebalance 주기 자동화: 항별 노름비 drift 감지 ---
    # RebalanceGradScales의 log-space EMA(0.7/0.3)는 시간 상수가 "에폭"이 아니라 **호출 횟수**다:
    #   ln(0.1)/ln(0.7) ≈ 6.5회 → grad_balance_every=10이면 step 변화 추종에 65에폭.
    #   pretrain의 roll_depth_ramp_epochs=50보다 길어서 depth 램프 구간(ep 250~650) 내내
    #   roll grad_scale이 현재 depth에 수렴한 적이 없다.
    # 촘촘히 부르는 것 자체는 노이즈 손해가 없다 — EMA 정상상태 분산은 (1-b)/(1+b) ≈ 0.18배로
    #   호출 간격과 무관하고, 비용만 든다(1회 ≈ 에폭의 20~25%).
    # 그래서 "언제" 부를지를 손으로 쓴 에폭 스케줄 대신 공짜 진단값(_term_gnorm)으로 정한다.
    def GnormRatios(self):
        # 항별 post-scale 노름을 data 기준으로 정규화하고, **의도된** 램프는 나눠서 제거한다
        # (phys_ramp/roll_ramp는 예정된 스케줄이지 불균형이 아니다).
        # roll depth 증분은 나누지 않는다 — 그게 바로 감지 대상이다.
        # _term_gnorm 값은 0-dim CUDA 텐서라(trap 20) 여기서 에폭당 1회만 float()로 sync한다.
        tg = self._term_gnorm
        if not tg:
            return None
        vals = {k: float(v) for k, v in tg.items()}
        anchor = vals.get("data", 0.0)
        if not anchor > 0.0:
            return None
        ramps = {"phys": max(self.phys_ramp, 1e-6), "roll": max(self.roll_ramp, 1e-6)}
        out = {}
        for k, v in vals.items():
            if k == "data" or not v > 0.0:
                continue
            out[k] = math.log(v / (anchor * ramps.get(k, 1.0)))
        return out or None

    def GnormDrift(self, ref):
        # max_k |log r_k - log r_k^ref| — 공통 키만 비교 (roll 활성화 전후로 키 집합이 다름)
        cur = self.GnormRatios()
        if not cur or not ref:
            return None
        common = [k for k in cur if k in ref]
        if not common:
            return None
        return max(abs(cur[k] - ref[k]) for k in common)

    # 누적 버퍼는 런 전체에서 1회만 할당하고 스텝마다 zero_()로 재사용한다.
    # 매 스텝 zeros_like 리스트를 새로 만들면 파라미터 전체 크기(~3M)를 스텝마다 alloc/free 한다.
    # 짝으로 _FlushTerm의 zero_grad도 set_to_none=False — .grad 버퍼까지 autograd가 재사용하게 해
    # 스텝당 할당원을 둘 다 없앤다. // allocate once, reuse
    def _GradAcc(self):
        if getattr(self, "_grad_acc", None) is None:
            self._grad_acc = [torch.zeros_like(p) for p in self.parameters()]
        else:
            for a in self._grad_acc:
                a.zero_()
        return self._grad_acc

    def _FlushTerm(self, acc, budget, name=None):
        # 직전 항의 .grad(청크 누적 완료분)를 clip → acc로 이관 → .grad 비움
        # clip_grad_norm_의 반환값 = clip 이전 노름 (공짜) — 항별 포화 여부 진단용 EMA로 축적.
        # rebalance(10에폭)를 기다리지 않고 매 스텝 실제 유효 크기를 볼 수 있다. // saturation probe
        # GPU 텐서로 유지 — float()를 걸면 항마다 host sync가 생긴다 (읽는 쪽에서 1회 변환).
        pre = torch.nn.utils.clip_grad_norm_(self.parameters(), budget).detach()
        if name is not None:
            prev = self._term_gnorm.get(name)
            nxt = pre if prev is None else 0.9 * prev + 0.1 * pre
            # NaN 한 번이면 EMA가 영구 NaN이 되어 GnormRatios가 그 항을 통째로 버린다(= 밸런싱
            # 정지). 비유한 갱신은 직전 값을 유지해 흘려보낸다 — where로 처리해 host sync 없음.
            zero = torch.zeros_like(nxt)
            self._term_gnorm[name] = torch.where(
                torch.isfinite(nxt), nxt, prev if prev is not None else zero)
        for p, a in zip(self.parameters(), acc):
            if p.grad is not None:
                a.add_(p.grad)
        # set_to_none=False: .grad 텐서를 살려둔 채 0으로만 비운다 → 다음 항의 backward가
        #   같은 버퍼에 누적(재할당 없음). set_to_none=True면 항마다 .grad를 새로 할당한다.
        self.optimizer.zero_grad(set_to_none=False)

    def _WriteBackAcc(self, acc):
        if acc is None:
            return
        # copy_ 필수 — p.grad = a 로 대입하면 p.grad가 누적 버퍼를 **별칭**하게 된다.
        #   버퍼를 재사용하는 지금 그러면 다음 스텝에서 acc is p.grad 가 되어
        #   _FlushTerm의 a.add_(p.grad)가 첫 항을 2배로 세는 조용한 오류가 난다. // no aliasing
        for p, a in zip(self.parameters(), acc):
            if p.grad is None:
                p.grad = a.clone()   # NaN 스킵 경로가 set_to_none=True로 비운 직후 등 // rare
            else:
                p.grad.copy_(a)

    def _GradNormOfLoss(self, loss):
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=False)
        gnorm = self._GlobalGradNorm()
        self.optimizer.zero_grad(set_to_none=True)
        return gnorm

    # --- 손실 스칼라는 GPU에 누적, host sync는 스텝 끝 1회 // no per-chunk sync ---
    # float(loss.detach())는 그 자리에서 GPU→CPU 왕복을 강제해 파이프라인을 세운다. 청크마다
    # 4개 항(kin/phys/energy/combo)씩 걸려 있었으므로 스텝당 sync가 30회를 넘었다.
    #   측정(2026-07-30, 실제 스텝 eager+high, max_cases 10000/n_colloc 10240):
    #   122.60 → 117.88 ms (3.9%). gradient는 **bitwise 동일**(torch.equal 검증) — 계산은
    #   그대로고 읽는 시점만 뒤로 밀린다. 항별 avg 값만 GPU FP32 누적으로 바뀌어 relΔ ~1e-7.
    #   합성 스텝 모사에서는 21%가 나왔는데 그건 청크 수가 적어 sync 스톨이 안 가려진 조건이었다 —
    #   실제 스텝은 청크당 GPU 작업이 커서 CPU가 대부분 따라잡는다. 이득의 본체는 compile 쪽.
    # 남는 sync는 loop_common의 NaN 스킵 판정 1회뿐이다 (optimizer.step 전에 host 값이 필요).
    def _BackwardKinChunks(self, meta, metric_out, losses_out):
        # Phase 1 커플러 전용: kin만 backward (jvp, FP64 없음) // 헤드 연결을 데이터피팅과 함께 조기 확립
        n = meta["tau"].shape[0]
        chunk = self._LossChunk()
        lam = self.LambdaAt()
        kin_acc = 0.0
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            sl = slice(s, e)
            feats = self._BuildFeats(meta["tau"][sl], meta["params"][sl], meta["ic_state"][sl])
            lk = self.KinLoss(feats)
            frac = (e - s) / n
            weighted = lam["kin"] * self.grad_scale["kin"] * self.kin_ramp * lk * frac
            weighted.backward(retain_graph=False)
            kin_acc = kin_acc + lk.detach() * frac
            metric_out[0] = metric_out[0] + weighted.detach()
        losses_out["kin"] = kin_acc

    def _BackwardPhysicsChunks(self, meta, metric_out, losses_out):
        # Chunked colloc: rebuild feats per chunk // 청크마다 feats 새로 구성
        # kin은 kin_ramp(구조 커플러, 상시), phys/energy는 phys_ramp(Phase 2)로 분리 게이팅
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
            pr = self._PhysRamp()
            combo = (
                lam["kin"] * self.grad_scale["kin"] * self.kin_ramp * lk
                + (
                    lam["phys"] * self.grad_scale["phys"] * lp
                    + lam["energy"] * self.grad_scale["energy"] * le
                ) * pr
            ) * ((e - s) / n)
            combo.backward(retain_graph=False)
            frac = (e - s) / n
            kin_acc = kin_acc + lk.detach() * frac
            phys_acc = phys_acc + lp.detach() * frac
            en_acc = en_acc + le.detach() * frac
            metric_out[0] = metric_out[0] + combo.detach()
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
                ic_acc = ic_acc + lic.detach() * frac
                metric_out[0] = metric_out[0] + weighted.detach()
        losses_out["ic"] = ic_acc

    def BackwardDataIC(self, batch, ic_parts, colloc_meta=None):
        # Peak VRAM: data → (kin) → IC, one graph each // data·(kin)·IC 순차 backward
        # colloc_meta 주어지면 Phase 1에서도 kin 커플러 활성 (phys/energy 없음) // A1: 헤드 조기 연결
        # 반환값(metric, losses)은 0-dim GPU 텐서 — 호출자가 스텝 끝에서 1회만 float()로 읽는다.
        self.optimizer.zero_grad(set_to_none=True)
        lam = self.LambdaAt()
        losses = {}
        metric_box = [torch.zeros((), device=self.device)]
        budget = self._TermClipBudget()
        acc = self._GradAcc()

        l_data = self.DataLoss(*batch)
        w_data = lam["data"] * self.grad_scale["data"] * l_data
        w_data.backward(retain_graph=False)
        losses["data"] = l_data.detach()
        metric_box[0] = metric_box[0] + w_data.detach()
        self._FlushTerm(acc, budget, "data")

        if colloc_meta is not None and self.kin_ramp > 0.0:
            self._BackwardKinChunks(colloc_meta, metric_box, losses)
            self._FlushTerm(acc, budget, "kin")

        if ic_parts:
            self._BackwardICChunks(ic_parts, metric_box, losses)
            self._FlushTerm(acc, budget, "ic")
        else:
            losses["ic"] = self._ZeroLoss()   # hard_ic → IC 잔차가 항등적으로 0 // 항 자체를 건너뜀

        self._WriteBackAcc(acc)
        return metric_box[0], losses

    def _ZeroLoss(self):
        return torch.zeros((), device=self.device)

    def BackwardAll(self, batch, colloc_meta, ic_parts, roll_loss=None):
        # Peak VRAM: data → physics chunks → IC chunks → roll // 항목별 순차 backward
        # 반환값(metric, losses)은 0-dim GPU 텐서 — 호출자가 스텝 끝에서 1회만 float()로 읽는다.
        self.optimizer.zero_grad(set_to_none=True)
        lam = self.LambdaAt()
        losses = {}
        metric_box = [torch.zeros((), device=self.device)]
        budget = self._TermClipBudget()
        acc = self._GradAcc()

        l_data = self.DataLoss(*batch)
        w_data = lam["data"] * self.grad_scale["data"] * l_data
        w_data.backward(retain_graph=False)
        losses["data"] = l_data.detach()
        metric_box[0] = metric_box[0] + w_data.detach()
        self._FlushTerm(acc, budget, "data")

        self._BackwardPhysicsChunks(colloc_meta, metric_box, losses)
        self._FlushTerm(acc, budget, "phys")   # kin+phys+energy combo 단위 // combo is what blows up
        if ic_parts:
            self._BackwardICChunks(ic_parts, metric_box, losses)
            self._FlushTerm(acc, budget, "ic")
        else:
            losses["ic"] = self._ZeroLoss()   # hard_ic → IC 잔차가 항등적으로 0 // 항 자체를 건너뜀

        if roll_loss is not None:
            # roll도 grad_scale 경유 — 유일하게 밸런싱 밖에 있던 항. clip_grad_norm_은 전 항
            # 합산 뒤에 걸리므로 roll만 raw면 폭주 시 나머지 gradient가 소거됨 // 측정 근거는 _RobustRollMean
            w_roll = lam["roll"] * self.grad_scale["roll"] * self.roll_ramp * roll_loss
            w_roll.backward(retain_graph=False)
            losses["roll"] = roll_loss.detach()
            metric_box[0] = metric_box[0] + w_roll.detach()
            self._FlushTerm(acc, budget, "roll")

        self._WriteBackAcc(acc)
        return metric_box[0], losses

    def RebalanceGradScales(self, n_batches=3, eps=1e-12):
        # B: 항목별 독립 forward-backward로 grad 노름 측정 // retain_graph 없이 peak VRAM 절약
        self.train()
        sums = {"data": 0.0, "kin": 0.0, "phys": 0.0, "energy": 0.0}
        if self.IcTermOn():
            sums["ic"] = 0.0
        if self._roll_active:
            sums["roll"] = 0.0
        cnt = 0
        chunk = self._LossChunk()
        for _ in range(n_batches):
            for t_lo, t_hi in self.segments:
                frame = self.SegmentFrame(t_lo, t_hi)
                batch = self.SegmentBatch(frame)
                # 아래에서 쓰는 건 part0의 앞 chunk행뿐 — 전체를 만들면 그대로 버려진다 // 측정 비용 절감
                ic_parts = self.ICSamplesRaw(max_n=chunk) if self.IcTermOn() else None
                colloc_meta = self.CollocationForBalance(t_lo, t_hi)

                sums["data"] += self._GradNormOfLoss(self.DataLoss(*batch))

                e_col = min(chunk, colloc_meta["tau"].shape[0])
                sl = slice(0, e_col)
                ic_theta = colloc_meta["ic_state"][sl, [0, 2]]
                p_sl = colloc_meta["params"][sl]
                e0_sl = colloc_meta["e0"][sl]
                for key, idx in (("kin", 0), ("phys", 1), ("energy", 2)):
                    feats_i = self._BuildFeats(
                        colloc_meta["tau"][sl], colloc_meta["params"][sl], colloc_meta["ic_state"][sl]
                    )
                    parts = self._PhysicsEnergySlice(feats_i, p_sl, e0_sl, ic_theta)
                    sums[key] += self._GradNormOfLoss(parts[idx])

                if ic_parts:
                    part0 = ic_parts[0]
                    e_ic = min(chunk, part0["ic_state"].shape[0])
                    ic_feats = self._BuildFeats(
                        part0["tau"][:e_ic], part0["params"][:e_ic], part0["ic_state"][:e_ic]
                    )
                    sums["ic"] += self._GradNormOfLoss(
                        self._ICLossSlice(ic_feats, part0["theta0"][:e_ic], part0["omega0"][:e_ic])
                    )

                if self._roll_active:
                    # roll은 세그먼트 루프와 무관(자체적으로 윈도우 추첨) — 학습 스텝과 동일 조건으로 측정
                    n_roll = min(self.trainCfg.roll_balance_cases, len(self.active_cases))
                    sel = self.active_cases[
                        torch.randint(0, len(self.active_cases), (n_roll,), device=self.device)
                    ]
                    sums["roll"] += self._GradNormOfLoss(
                        self.RolloutLoss(sel, self.RollDepth(), self.trainCfg.roll_balance_points)
                    )

                cnt += 1
                del colloc_meta, batch, ic_parts, frame
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        g = {k: sums[k] / cnt for k in sums}        # 평균 grad 노름
        anchor = g["data"] + eps
        # clamp 하한 제거 (2026-07-29). 하한 0.1 → 0.01로 낮췄을 때 **0.01에도 그대로 붙박이**임을
        #   확인했다(같은 런 ep~1750~3141 내내 scale phys=1.00e-02). 앵커가 요구하는 값은 ~1e-3이라
        #   하한을 또 내리는 건 같은 실패의 반복 — 문제는 하한값이 아니라 clamp로 앵커를 쫓는 구조다.
        #   하한은 아무것도 보호하지 않는다: gradient가 큰 항이 작은 scale을 받는 게 밸런서의 목적이고,
        #   매 rebalance마다 fresh ‖g‖로 재계산하므로 "0으로 떨어져 못 돌아옴"도 없다.
        #   폭주 방지는 상한의 역할이며, 상한도 10 → 100으로 완화했다 (energy가 앵커 요구값 ~12를
        #   못 받고 9.94~9.99로 10에 붙박이였다 = energy 밸런싱도 꺼져 있었다).
        # 하한 제거는 항별 clip(_TermClipBudget)이 있어야 안전하다 — 단독으로는 쓰지 말 것.
        hi = getattr(self.trainCfg, "grad_scale_max", 100.0)
        for k in g:
            # 측정이 비유한이면 그 항은 이번 회차를 건너뛴다. max(lo, anchor/nan) = lo(=0)로 접혀
            # scale이 조용히 0으로 붕괴하던 경로다 — 밸런싱 실패를 "가중치 0"으로 오독하지 않는다.
            if not (math.isfinite(g[k]) and math.isfinite(anchor)):
                continue
            # EMA 교체: 계단 외란 제거 // smooth update instead of hard replace
            # log 공간 EMA: scale이 1e-3~1e2로 여러 자릿수에 걸치는데 선형 EMA는 상승/하강 응답이
            #   비대칭이다(0.01에서 1로 오르는 데 걸리는 rebalance 수 >> 반대 방향). // symmetric response
            new_scale = min(hi, anchor / (g[k] + eps))
            cur = max(self.grad_scale[k], eps)
            self.grad_scale[k] = math.exp(
                0.7 * math.log(cur) + 0.3 * math.log(max(new_scale, eps))
            )
        self._phys_balanced = True
        return g

    def LambdaAt(self):
        return dict(self._base_lambda)


class TimeMarching:
    # Non-overlapping march windows + fresh uniform-τ 콜로케이션 (매 step 재샘플, persistent pool 없음)
    def _CollocCasePool(self):
        # colloc case 집합; 미설정(pretrain) 시 active_cases 폴백 // finetune은 flip-biased colloc_cases 고정
        pool = getattr(self, "colloc_cases", None)
        if pool is None:
            return self.active_cases
        return pool

    def _SampleCollocMeta(self, t_lo, t_hi, case_pool, n=None):
        # Fresh 콜로케이션: iid uniform τ∈[0,span] + case_pool 균등 추출 // 매 호출 재샘플
        n = n or self.collocCfg.n_colloc
        span = t_hi - t_lo
        tau = torch.rand(n, 1, device=self.device) * span

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
        # off-manifold IC 섭동 (Phase 3, 에폭 램프) // ic_sigma 램프
        sigma = getattr(self, "_colloc_ic_sigma", 0.0)
        ic_state = ic_base + sigma * torch.randn_like(ic_base) if sigma > 0.0 else ic_base
        e0 = self.GetEnergy(ic_state.float(), params.float()).detach()   # 2026-08-30: FP32 (Loss.py 주석)
        return {"tau": tau, "ic_state": ic_state, "params": params, "e0": e0}

    def SetupCollocCases(self, n_colloc=None, flip_bias=0.0):
        # flip finetune: flip-biased colloc_cases 고정(stratified) // 명시적 is_flip balancing
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
        self._colloc_inited = True

    def GetCollocMeta(self, t_lo, t_hi, case_pool=None):
        # Fresh collocation meta (no feats) — feats built per chunk at train time // 매 호출 재샘플
        pool = case_pool if case_pool is not None else self._CollocCasePool()
        return self._SampleCollocMeta(t_lo, t_hi, pool)

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
        return self._RollForward(feats, p)

    def _RollForward(self, feats, n_points):
        # Window forward for case-major rows (n·P): FiLM once per case, expanded to P rows.
        # 같은 케이스의 P점은 cond가 동일한데 행마다 계산하면 cond MLP(트렁크와 같은 FLOP)가 P배 중복.
        # compile 단위 — feats 생성(gather)은 밖. // 케이스별 FiLM 1회 + compile (2026-08-31)
        return self._Compiled("roll", self._RollForwardImpl)(feats, n_points)

    def _RollForwardImpl(self, feats, n_points):
        n = feats.shape[0] // n_points
        cond = tuple(c.repeat_interleave(n_points, dim=0)
                     for c in self.CondOf(feats[::n_points]))      # 케이스 대표행 (case-major)
        return self(feats, cond=cond).reshape(n, n_points, 4)
