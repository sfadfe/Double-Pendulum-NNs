import tomllib
from dataclasses import dataclass


@dataclass
class NetCfg:
    fourier_l: int = 32       # number of frequencies L // 주파수 개수
    f_min: float = 0.2        # Hz, lowest oscillation // 최저 진동 주파수
    f_max: float = 20.0       # Hz // 데이터 대역폭 측정치 기반 상한
    # 측정(mixed corpus, 1s 윈도우, Hann): 데이터 Nyquist는 50Hz라 >50Hz 파워는 정확히 0.
    #   flip ω 케이스별 f(99% 파워) p99 = 15.8Hz, >20Hz 잔여 파워 7e-4. nonflip은 그 1/10.
    # 56Hz였을 때 문제: θ 진폭 17.24/(2π·56)=0.049면 dθ/dτ에 std 17의 노이즈를 만드는데,
    #   이는 Val θ RMS(~0.12) 아래라 data loss가 못 본다 → 무제약 성장. 20Hz면 0.137로
    #   data 오차 수준까지 올라와 data loss가 억제한다. // [[kin-derivative-noise]]

    ic_feat_dim: int = 6      # 4 trig(θ0) + 2 ω(0) // raw m/L 제외 (ParamEmbed로 대체)
    param_embed_dim: int = 32 # ParamEmbed 출력 차원 // option B3
    width: int = 384          # hidden layer width // 은닉층 너비 (M tier)
    n: int = 6                # FiLMBlock count // 잔차 블록 수

    # τ=0 경계조건 부과 방식 // Networks.forward 말미의 출력 ansatz
    #   "off" — 종전. 출력이 raw [Δθ, ω]이고 IC는 soft 손실로만 유도된다.
    #   "c0"  — Δθ = s·A_θ, ω = ω_IC + s·A_ω. Δθ(0)=0, ω(0)=ω_IC 가 항등식.
    #   "c1"  — 추가로 dΔθ/dτ(0) = ω_IC. Δθ = s·(march_dt·ω_IC + s·A_θ).
    #   "off"가 아니면 IC 손실이 정확히 0이므로 loop가 ic 항 자체를 끈다(비용 절감).
    #   구조 변경이라 체크포인트 비호환 — 값을 바꾸면 fresh pretrain이 필요하다.
    hard_ic: str = "c1"

    # 조건화: trunk 입력은 τ features뿐, 케이스(IC·param_embed)는 FiLMCond를
    #   타고 블록마다 (γ, β)로 잔차 가지를 곱셈 변조한다 (networks.FiLMBlock). 옵션 아님.
    film_hidden: int = 128    # 조건화 MLP 은닉폭. 출력은 2·width·n (블록당 γ, β)

    # State-correction stages: after block k (1-based, k in stage_at) an
    #   intermediate head reads x̂_k = HardIC(Σ_{j≤k} a_j), a zero-init Linear(4→width) adds it back to h,
    #   and later blocks correct that estimate instead of mapping τ from scratch. Final output is
    #   HardIC(Σ a_j + head_final). stage_w weights the intermediate data losses (deep supervision); 0 = none.
    #   () = off, unchanged network. Checkpoint-incompatible with () → fresh runs only.
    # // 상태 보정 단계: stage_at 블록 뒤에 중간 헤드 → x̂_k → h에 재주입(zero-init). 잔차형(a 누적), 중간 손실 stage_w
    #   stage_inj: "state" = x̂_k 재주입 | "tau" = 대조 팔, HardIC(a=0)=[s·md·ω_IC, ω_IC]만 재주입 (헤드·중간 손실은
    #   동일) → 두 팔의 차 = "상태 정보"의 몫, τ 재주입 자체의 몫과 분리.
    #   stage_inj "embed": predictor→corrector handoff. At stage k, h is replaced by
    #   FFN([x̂_k·stage_scale, τ feats]) (Linear→SiLU→Linear, width), final heads zero-init (output = x̂_k at init).
    #   Use with film_split_at = k (corrector gets its own FiLM) and stage_w = 1.0 (blocks ≤ k = full predictor).
    # // "embed": 예측기 h 대신 x̂_k·τ 임베딩으로 보정기 시작. film_split_at=k, stage_w=1.0과 함께 쓴다
    stage_at: tuple = ()
    stage_w: float = 0.1
    stage_inj: str = "state"
    # Correction-unit screen: per-block type list of length n.
    #   "film" = FiLMBlock (γ,β from FiLMCond) | "ffn" = same block with γ≡1, β≡0 (no conditioning, same MACs)
    #   | "attn" = AttnPoolBlock (h → attn_tokens sub-tokens, one learned query, zero-init output).
    #   FiLMCond is sized to the number of "film" blocks; () = all film (checkpoint-compatible, unchanged net).
    # // 블록 타입 열: film/ffn/attn. FiLMCond 출력은 film 블록 수만큼, () = 전부 film(종전과 동일)
    #   film_split_at > 0: blocks ≥ split get their own FiLMCond MLP (film_corr) so the prediction-region
    #   FiLM can be frozen exactly (requires_grad, no partial-row masking under AdamW decay). 0 = one shared MLP.
    # // film_split_at: 이 블록부터 별도 FiLMCond(film_corr). 동결 스크린용, 0 = 공유 1개(종전)
    block_types: tuple = ()
    film_split_at: int = 0
    attn_tokens: int = 8
    # "xattn" (window-token corrector): per case the predictor is also evaluated on a fixed τ grid
    #   τ_j = j·march_dt/P (j=1..P, xattn_tokens); those P rows go through the same handoff (stage_inj embed)
    #   and become tokens. An xattn block = multi-head cross-attention of every row (query and token rows alike)
    #   over the case's token rows (zero-init output), followed by the FiLM residual branch. Rows must be
    #   case-major with n_q rows per case (forward(n_q=)); the P token rows are appended inside forward.
    #   Corrector thus sees the predicted trajectory shape across τ, which no per-row input carries.
    # // 창 안 τ 토큰 보정기: 케이스별 예측기 x̂를 τ 격자 P점에서 뽑아 토큰으로, 각 행이 그 집합에 cross-attention
    xattn_tokens: int = 8
    xattn_heads: int = 4
    # Window-state corrector: per case the predictor is evaluated on the τ grid τ_j = j·march_dt/P
    #   (j=1..P, win_tokens; 25 = the dt 0.01 data grid of a 0.25 s window). The P states x̂_k·stage_scale are
    #   concatenated (4P) → FFN → z (width), added to the handoff embedding of every row of the case. Corrector
    #   blocks stay film. Token states are detached (no grad into the predictor through z). Needs stage_inj
    #   "embed", one stage, no xattn block; rows case-major with n_q per case (same plumbing as xattn). 0 = off.
    # // 창 상태 보정기: 예측기 x̂_k를 창 격자 P점에서 한 번에 뽑아 4P 벡터 → FFN → z, 케이스의 모든 행 handoff 임베딩에 더함
    win_tokens: int = 0
    # Activation arms (2026-10-14). block_acts: branch activation per block, length n — "silu" | "ztanh"
    #   (z·tanh(z), even: folds negatives to positive, σ'(0)=0, σ''(0)=2). film/ffn blocks only. () = all silu.
    #   stream_act_at: block boundaries (1-based, as stage_at) after which the whole stream goes through SiLU once,
    #   h ← SiLU(h) (no skip around it). () = off. Neither has parameters → state_dict unchanged.
    # // block_acts: 블록별 가지 활성함수 (ztanh = 짝함수). stream_act_at: 그 블록 뒤에서 스트림 전체에 SiLU 1회 (skip 없음)
    block_acts: tuple = ()
    stream_act_at: tuple = ()
    # Per-case time warp (2026-10-19, branch time-warp, networks.TimeWarp). warp_k > 0: the trunk reads σ(τ; c) instead
    #   of τ (Fourier features and t_norm; hard_ic keeps raw τ). σ = composition of warp_k speed dips with σ(0) = 0,
    #   σ(march_dt) = march_dt, (depth d, centre μ, sharpness β) per case from an MLP(cond) of width warp_hidden,
    #   β starts at warp_beta_init (window units). Output zero-init → d = 0 → σ = τ at init. Near the separatrix
    #   (θ2 ≈ π) the dwell time is exponentially sensitive to the energy, so cases share a shape but not a clock;
    #   the FiLM trunk after a fixed RFF cannot move the time axis. 0 = off, state_dict unchanged. Plain nets only.
    # // 케이스별 시간 워프: 트렁크 입력 τ → σ(τ; c). 감속 딥 warp_k개, 끝점 고정, zero-init → 시작 시 σ = τ. 0 = 끔
    warp_k: int = 0
    warp_hidden: int = 64
    warp_beta_init: float = 4.0
    # Class split (2026-10-19, branch FiLM-MoE, vault "Class-split FiLM"). class_split = k > 0: flip cases (gate
    #   log(e_rel/e_flip) > class_gate_center, Dataset.ClassGate = the states.py label rule, exact on the corpus) get their
    #   own proj_in, blocks 0..k-1, FiLMCond and heads; blocks k.. share fc1 and take (γ, β) from the row's class FiLMCond.
    #   The gate rides as the last feats column. 0 = off, state_dict unchanged. Plain film/ffn nets only.
    # // 클래스 분리: flip 케이스는 앞단(proj_in + 블록 0..k-1)·FiLMCond·헤드가 따로, 블록 k..는 fc1 공유. 0 = 끔
    class_split: int = 0
    class_gate_center: float = -0.255
    # K-way routes (2026-10-20, branch FiLM-MoE): class_routes routes of the class_split structure, each with its own
    #   front, FiLMCond, heads and (warp_k > 0) time warp. 2 = flip / NF by the energy gate above. 3 = no energy gate:
    #   route = argmax Router(window-start state, param_embed) — 0 NF, 1 flip window not passing over the top, 2 window
    #   passing over the top (speed: rotating < NF < flip not rotating → own warps). Window-constant, recomputed per
    #   window in every rollout; the router learns from TrainCfg.route_sup only. route_hidden = router MLP width.
    # // K경로: 2 = 에너지 게이트 flip/NF, 3 = 라우터(창 시작 상태) NF / flip 비회전 / flip 회전 창, 경로마다 워프
    class_routes: int = 2
    route_hidden: int = 64
    # Shared stem + scale routing (2026-10-20, branch FiLM-MoE). route_at = a > 0: proj_in and blocks 0..a-1 are shared
    #   by every route (stem FiLM slots from the shared FiLMCond), the route-own blocks are a..a+class_split-1, then the
    #   shared back; own FiLMCond slots, heads and warp as above. route_by "router" = as above; "scale" = no router, the
    #   window route is argmax_e of route e's own activation size Σ_j ‖SiLU(fc1 h_a)‖² at its first own block, h_a = the
    #   stem under route e's warp at route_pts τ points (AoE-style, Networks.ScaleScores; needs a ≥ 1 — block 0 reads τ
    #   only, scratch/aoe_depth_probe.log: windows separate from block 3 on). Window-constant, recomputed per window.
    # // 공유 몸통(블록 0..a-1) 뒤 경로 전용 블록. scale: 라우터 없이 expert 자기 활성 크기로 창 경로 선택
    # route_by "gate" (2026-10-21, branch FiLM-MoE, class_routes 3, route_at ≥ 1, warp_k 0): routes 0 nf / 1 flip window
    #   not passing over the top (pre f + post f) / 2 cross. Training rows take the true label of their window start
    #   (Dataset.TrueRoute: energy gate, then RK4 over the window); eval rows take the energy gate, then inside flip a
    #   binary gate MLP (route_hidden) on the stem output h_a averaged over route_pts τ points (Networks.GateLogits),
    #   trained by TrainCfg.route_sup CE on flip windows with its gradient into the stem scaled (gate_lam_frac).
    # // gate: 학습은 창 시작 상태의 참 라벨, 평가는 에너지 게이트 → flip 안에서 h_a 이진 게이트
    route_at: int = 0
    route_by: str = "router"
    route_pts: int = 4
    # route_beta_in (2026-10-22, branch FiLM-MoE): FiLM blocks of routes ≥ 1 (the flip experts) take β inside the
    #   activation, h ← h + γ·SiLU(W h + b + β), instead of h + γ·SiLU(W h + b) + β. Same parameters and init (β = 0 at
    #   init); route 0 (nf) and the shared stem unchanged. // flip expert FiLM 블록만 β를 SiLU 안으로. 파라미터·초기값 동일
    route_beta_in: bool = False

    @property
    def gx_dim(self) -> int:
        # Fourier(2L) + t_norm(1) + IC(6) + param_embed // concat trunk 입력 차원
        return 2 * self.fourier_l + 1 + self.ic_feat_dim + self.param_embed_dim

    @property
    def cond_dim(self) -> int:
        # IC(6) + param_embed // τ 무관 열만 = feats[:, 1:]
        #   τ에 의존하면 StateDerivs의 jvp dual 섭동이 γ·β를 오염시킨다
        return self.ic_feat_dim + self.param_embed_dim

    @property
    def trunk_in_dim(self) -> int:
        # trunk는 τ만 받는다 — 케이스는 γ·β로만 들어온다
        return 2 * self.fourier_l + 1


@dataclass
class TrainCfg:
    lambda_ic: float = 1.0
    lambda_data: float = 1.0
    lambda_kin: float = 1.0
    lambda_phys: float = 1.0
    lambda_energy: float = 0.5
    lambda_roll: float = 1.0      # rollout-aware (pushforward) 손실 base 가중치 // 윈도우 핸드오프 교정
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    # 항별 clip 예산은 grad_clip 고정, RebalanceGradScales clamp 하한은 없음(0) — 하드코딩.
    #   하한은 아무것도 보호하지 않음, 폭주 방지는 상한 역할 // [[grad-scale-floor-pinned]]
    grad_scale_max: float = 100.0 # RebalanceGradScales clamp 상한 // energy가 10에 붙박이라 완화
    replay_frac: float = 0.25
    n_colloc_cases: int = 0       # finetune colloc case 풀 크기 (0 → max_cases)
    phys_ramp_center: int = 30    # e2 = epoch - warmup_epochs 기준 중심
    phys_ramp_width: int = 10     # sigmoid 폭 // ramp steepness
    lr_drop_epoch: int = 0        # 고정 LR 1회 하향 에폭 (0=비활성) // OdeScheduler와 별개
    lr_drop_to: float = 0.0       # lr_drop_epoch부터 적용할 LR 상한 // min(cur, lr_drop_to)
    ema_decay: float = 0.999      # Polyak weight EMA decay (0=비활성) // best.pt/평가 안정화
    roll_robust_k: float = 4.0    # rollout Cauchy 스케일 = k × 배치 median (0=비활성, 순수 mean) // heavy-tail 억제
    roll_balance_cases: int = 128 # RebalanceGradScales의 roll grad 노름 측정용 케이스 수 // 측정 비용 절감
    roll_balance_points: int = 12 # 위 측정의 윈도우 내 시점 수 // 학습 스텝의 roll_points와 동일 스케일

    # 스텝당 (k0, depth) 추첨 횟수 (1 = 단일 추첨) // roll gradient 분산 축소
    #   roll gradient 분산은 케이스 수가 아니라 추첨 개수가 결정한다 — roll_cases를 늘려도 안 줄어든다.
    #   케이스 예산을 n_draws로 쪼개므로 네트워크 evaluation 수는 불변 = 비용 거의 동일.
    #   k0는 윈도우 축으로 stratify하고, robust mean은 합친 뒤 1회만 건다.
    roll_draws: int = 1
    # Cauchy 스케일 c의 median EMA 계수 (0 = 종전, 매 스텝 배치 median 그대로) // 스케일 자체가
    #   추첨마다 흔들리면 그것도 노이즈원이다. 0.9면 c가 학습 진행을 따라가되 스텝 노이즈는 뺀다.
    roll_median_ema: float = 0.0
    # robust mean 형태: "cauchy" = c·log1p(l/c) (tail grad ∝ c/l), "huber" = min(l, 2√(δl)−δ)
    #   (tail grad ∝ √(δ/l)) — Cauchy가 tail gradient를 끊어 flip 평균을 악화시킨 것의 완화
    #   후보. δ/c = roll_robust_k × median 공용. // [[rollout-tail-cauchy-tradeoff]]
    roll_robust_kind: str = "cauchy"
    # Target of the roll grad window: "label" = stored trajectory, "rk4" = RK4 over the window from the marched IC
    #   (ODE.Rk4Window, substep ≤ roll_rk4_dt). The stored label also holds the chaos-amplified IC error, which the
    #   detached pushforward cannot reduce; "rk4" leaves only the net's own one-window error — trap 31.
    #   // roll 타깃: "label" = 참 궤적, "rk4" = 마칭 IC에서 RK4로 적분한 한 창 (넷 자체 오차만 남김)
    roll_target: str = "label"
    roll_rk4_dt: float = 0.005
    # Drop the top-q fraction of rows (by per-row loss) from each term's gradient (0 = plain mean) // 손실 상위 q 행을
    #   grad에서 제외. 상위 소수 행이 손실·grad 노이즈를 지배해서. 분위는 청크(배치)별.
    trim_q_data: float = 0.0
    trim_q_phys: float = 0.0
    trim_q_kin: float = 0.0
    # Class-normalized data loss (class_split nets): L = L_NF + flip_w · L_flip · EMA(L_NF) / EMA(L_flip), EMAs detached
    #   with per-step decay class_norm_ema — each class gets the same relative weight, at the NF loss scale. False = row mean.
    # // 클래스 정규화 data 손실: flip 손실을 NF/flip EMA 비로 맞춰 더한다 (스케일은 NF 손실). False = 행 평균
    class_norm: bool = False
    flip_w: float = 1.0
    class_norm_ema: float = 0.99
    # Router supervision (NetCfg.class_routes ≥ 3): CE of the router on route_sup_cases random (case, window start)
    #   pairs of active_cases per step, labels Trainstep.RouteLabels from the corpus path. Input detached → only the
    #   router learns; weight route_sup, outside RebalanceGradScales. Logged as AvgLoss_Route / RouteBacc.
    #   route_by "scale": no CE, route_sup_cases windows per step only for the RouteBacc / RouteLoad diagnostics.
    # // 라우터 CE 감독 — 라우터만 학습, 밸런서 밖. scale이면 진단(RouteBacc·RouteLoad)용 추첨 수만
    route_sup: float = 1.0
    route_sup_cases: int = 4096
    # route_by "gate": the gate CE draws route_sup_cases flip windows; its gradient enters the stem scaled by λ (forward
    #   identity) with λ = min(1, gate_lam_frac · ‖data grad on the stem‖ / EMA ‖CE grad on the stem at λ = 1‖) —
    #   the stem's CE share stays ≤ gate_lam_frac of the data term. Stem = proj_in + blocks 0..route_at-1. Logged GateLam.
    # // gate CE의 몸통 gradient를 data 몸통 gradient의 gate_lam_frac 이하로 (λ, forward 항등)
    gate_lam_frac: float = 0.1

    # IC 항 스텝당 케이스 상한 (0 = 전체 active_cases) // IC는 세그먼트마다 반복되는 항이라
    #   비용이 max_cases × n_segments로 곱셈이다 (march_dt를 줄이면 조용히 커진다).
    #   서브샘플은 사실상 공짜: 1024에서도 전체 대비 gradient cos ≈ 0.998로 data 미니배치 노이즈보다
    #   조용하고, 손실이 평균 정규화라 유효 가중치도 불변. // [[ic-term-dominates-step]]
    ic_max_n: int = 0

    # Frozen-prediction-region screen: load proj_in / param_embed / blocks[:freeze_below] /
    #   stage_heads[0] / film (split A) / scaler buffers from freeze_from and freeze them, so every arm sees the
    #   same x̂_{freeze_below}; the rest (blocks ≥ freeze_below, later stage heads, final heads, stage_inj,
    #   film_corr) starts fresh. Requires net.film_split_at == freeze_below and stage_at[0] == freeze_below.
    # // 예측 구간 동결: freeze_from ckpt에서 블록 1..freeze_below·stage head·FiLM(A)·사영 로드 후 동결. ""=off
    freeze_from: str = ""
    freeze_below: int = 0

    # torch.compile: data/IC/kin/phys 손실 슬라이스를 Inductor로 컴파일 // 고정 shape 전제
    #   스텝 ~1.66× 빠름, 컴파일 대기 ~27s. 전제: 청크 shape이 몇 개로 고정. n_colloc을 colloc_chunk 배수로 두면 그래프 1개,
    #   나머지 청크가 생기면 shape별로 1개 추가(허용치 내). // [[fp64-not-bottleneck]]
    use_compile: bool = True   # 전 config에서 true로 확정 — 프로브만 코드에서 끈다(probe_common)

    # FP32 matmul 정밀도. "high"=TF32 허용, "highest"=순수 FP32. 학습은 "high" 고정(toml 옵션 없음):
    #   현재 n_colloc에서 TF32 반올림 노이즈는 콜로케이션 추첨 노이즈보다 작다. 이상치(ic 항)는 트랩 18.
    #   **추론은 별개** — eval/프로브는 TF32 off (트랩 25).
    matmul_precision: str = "high"

    # fp16 GEMMs on the residual branch only: "off" | "blk" (FiLMBlock.fc1) | "blk+film" (+ FiLMCond last layer).
    #   The skip stream, heads and losses stay fp32, and so does everything under no_grad (eval, marching, rollout
    #   pushforward). Not the rejected whole-net autocast. R9700 solo step 62.0 → 48.7 (blk) / 46.4 ms (blk+film);
    #   gradient cos against fp32 0.9998 (same batch; a different batch gives 0.95). Needs the backward loss
    #   scale of LambdaBalance._BwScale — trap 29. // 잔차 가지 GEMM만 fp16 (전체 autocast 아님), 기본 off
    half_gemm: str = "off"

    # Rows per case in the colloc and data batches of a plain FiLM net (no xattn / window tokens). 1 = iid rows.
    #   K > 1: n/K cases × K τ rows each, case-major. The FiLM cond is then computed once per case and broadcast
    #   in the block (no per-row (γ, β)): the cond GEMMs and the (γ, β) streams shrink by K. Same total rows, but
    #   K times fewer distinct cases (and IC perturbations) per step — a change of the training distribution.
    #   Must divide n_colloc, colloc_chunk and batch_size. // 케이스당 K행으로 묶어 FiLM을 케이스당 1회 계산
    case_group: int = 1

    # Supervision of the first warp dip centre μ_1 (NetCfg.warp_k > 0, Trainstep.WarpSupDraw / WarpSupLoss). Each step
    #   draws warp_sup_cases (case, window start) pairs from active_cases on the corpus grid and labels them over the
    #   window extended by march_dt/2 on both sides (μ's range). warp_sup_label "cross": s* = the over-the-top
    #   crossing (θ = π + 2πk, either arm) nearest the window centre, windows without one unlabeled (nonflip never
    #   labeled). "ke": s* = interior minimum of the true kinetic energy, kept where min/max KE < warp_sup_ratio.
    #   Loss = mean (μ_1 − s*)² over labeled windows, gradient into the warp MLP only (detached feats), own clip
    #   budget, outside RebalanceGradScales. Weight warp_sup until epoch warp_sup_hold, then linear to 0 at
    #   warp_sup_end (≤ hold → step). Without it the dip has no gradient to its centre while d = 0 and the trunk can
    #   absorb the misalignment first (the MoE chicken-and-egg). warp_sup = 0 → off; with warp_k > 0 and
    #   warp_sup_cases > 0 the same loss is still logged (no_grad) as AvgLoss_Warp = μ_1 vs label.
    # // 첫 딥 중심 μ_1 감독: 라벨 "cross" = 꼭대기 통과 시각, "ke" = 운동에너지 내부 극소. 워프 MLP에만 gradient, 밸런싱 밖
    warp_sup: float = 0.0
    warp_sup_hold: int = 0
    warp_sup_end: int = 0
    warp_sup_cases: int = 4096
    warp_sup_ratio: float = 0.25
    warp_sup_label: str = "cross"

@dataclass
class CollocCfg:
    n_colloc: int = 20000
    ## 비용은 n_colloc에 선형 — colloc_chunk 단위 fwd+bwd 횟수가 그대로 늘어난다. 병목은 jvp fwd+bwd.
    ## use_compile 시에는 colloc_chunk의 배수로 두는 게 좋다 (나머지 청크가 그래프를 하나 더 만든다).

    colloc_flip_bias: float = 0.0   # finetune: flip case 하한 비율 (0=uniform) // 명시적 is_flip balancing

    ic_sigma: float = 0.0             # Phase 3: 물리 콜로케이션 IC 섭동 target (0=비활성)
    ic_sigma_warmup: int = 200        # sigma 0→target 선형 램프 에폭 (Phase 2 진입 후)
    # xattn nets: colloc rows grouped per case — n_colloc/per_case cases × per_case iid τ each,
    #   so every chunk is case-major for forward(n_q=per_case). IC perturbation is drawn per case.
    #   Ignored (1) when the net has no xattn block. colloc_chunk and n_colloc must be multiples of it.
    # // xattn: 콜로케이션을 케이스당 per_case개 τ로 묶음 (토큰 행 공유), 비-xattn net에선 무시
    per_case: int = 8
    # Collocation schedule (Trainstep.CollocSchedAt). sched_period = 0 → off: "grid" anchors, the ic_sigma ramp.
    #   mode of an epoch = sched_modes[(epoch // sched_period) % len(sched_modes)]:
    #     "grid"    — IC anchor = true state at the segment's window start, τ ~ U[0, span] (the only mode before)
    #     "any"     — IC anchor = true state at a uniformly random saved time index up to the last window start
    #     "tau_end" — "grid" anchors; every second row draws τ with density ∝ τ (span·√u)
    #   sigma target = sched_sigmas[min(epoch // (sched_period·len(sched_modes)), last)], [] → ic_sigma. The first
    #   target ramps like ic_sigma (0 → target over ic_sigma_warmup epochs after phase-2 entry), each later one
    #   linearly from the previous target over ic_sigma_warmup epochs after its switch epoch.
    # // 콜로케이션 스케줄: sched_period 에폭마다 모드 순환, 모드 한 바퀴마다 sigma 목표를 다음 값으로 (0 = 끔)
    sched_period: int = 0
    sched_modes: tuple = ("grid",)
    sched_sigmas: tuple = ()


@dataclass
class DataCfg:
    g: float = 9.81

    t_data_max: float = 3.0
    march_dt: float = 1.0     # time-marching window duration (s) // 플로우맵 윈도우 길이 — 네트워크가 보는 상대시간 τ∈[0,march_dt]
    energy_eps: float = 0.01  # energy 상대잔차 분모 바닥값 // floor when |E0|≈|E|≈0 (symmetric denom)
    phys_eps: float = 1.0     # EOM 상대잔차 분모 바닥값 // floor for relative physics residual denom (≈ small frac of typical |dω/dt|)
    kin_eps: float = 1.0      # kin 분모(= omega_rms²)에 더하는 바닥값 // floor, ω_rms²에 비해 작음

    nonflip_path: str = "data/nonflip_RK4_0_3s.npy"
    data_path: str = ""           # finetune mixed dataset (empty → nonflip_path)
    scaler_name: str = "scaler.npy"
    scaler_extra_omega: str = ""  # pretrain scaler ω RMS에 포함할 추가 궤적(예: mixed) — flip OOD 방지
    batch_size: int = 8192
    # kin/phys/ic backward 청크 크기 // 0이면 batch_size 사용. data 배치와 분리해 physics
    #   transient peak 억제. use_compile 하 스위트스팟 4096 (그 위는 수확체감).
    colloc_chunk: int = 0
    # Data batch rows in phase 2 (0 = batch_size). Must be a multiple of case_group. The compiled data graph
    #   gets a second shape at the phase boundary (one recompile). // phase 2의 data 배치 크기 (0 = batch_size)
    batch_size_phase2: int = 0

    # Fine label source of the data term (Dataset.LoadLabels / LabelBatch). "" = off: SegmentFrame / SegmentBatch
    #   on the corpus grid. label_path: (N, T_f, 4) [θ1, ω1, θ2, ω2] of the same cases in the same order, saved
    #   every dt_f = t_data_end / (T_f − 1); kept on the training device. Plain nets only (no xattn / window tokens).
    #   label_dt: label stride inside a window in phase 1. Offsets m·dt_f with (M − m) % stride == 0,
    #     M = march_dt / dt_f — the window end is always a label, τ = 0 never.
    #   label_dt_phase2: stride in phase 2 (0 = label_dt), except the "dense" cases, which keep label_dt:
    #     the top label_dense_frac of all cases by label_score_path ((N,), higher = faster dynamics).
    #   slide_frac: share of the groups whose window starts at any fine step with the window inside the data span,
    #     instead of the segment's grid start.
    #   march_dt, label_dt and label_dt_phase2 must be integer multiples of dt_f.
    # // data 항의 촘촘한 라벨 소스 ("" = 끔). 창 안 라벨 간격은 phase 1 label_dt, phase 2 label_dt_phase2
    #   (점수 상위 label_dense_frac 케이스는 label_dt 유지), slide_frac 몫의 그룹은 창 시작을 격자 밖 임의 시점으로
    label_path: str = ""
    label_score_path: str = ""
    label_dt: float = 0.0
    label_dt_phase2: float = 0.0
    label_dense_frac: float = 0.0
    slide_frac: float = 0.0
    # Corpus meta with is_flip / val_idx / train_idx (state/build_joint.py). Set → loop_common.SetupCaseSplit takes the
    #   split from it instead of "first n_val cases", and is_flip enables the flip / nonflip split of val and rollout.
    #   [train] flip_repeat k > 1 lists the flip train cases k times in the train pool. "" = off.
    # // 코퍼스 meta로 val/train 분할 + is_flip (pretrain 경로에서 flip 혼합 학습). "" = 종전 (앞 n_val)
    split_meta: str = ""


def LoadConfig(path):
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    t   = raw["train"]
    n   = raw["net"]
    o   = raw["optimizer"]
    s   = raw.get("scheduler", {})   # val scheduler 제거 후 하위호환용 // min_lr 폴백 소스
    lam = raw["lambda"]
    c   = raw["colloc"]
    d   = raw["data"]
    os_ = raw.get("ode_scheduler", {})

    net_cfg = NetCfg(
        fourier_l=n["fourier_l"],
        f_min=n["f_min"],
        f_max=n["f_max"],
        width=n["width"],
        n=n["n_blocks"],
        param_embed_dim=n.get("param_embed_dim", 32),
        hard_ic=n.get("hard_ic", "c1"),
        film_hidden=n.get("film_hidden", 128),
        stage_at=tuple(n.get("stage_at", [])), stage_w=n.get("stage_w", 0.1),
        stage_inj=n.get("stage_inj", "state"),
        block_types=tuple(n.get("block_types", [])), film_split_at=n.get("film_split_at", 0),
        attn_tokens=n.get("attn_tokens", 8),
        xattn_tokens=n.get("xattn_tokens", 8), xattn_heads=n.get("xattn_heads", 4),
        win_tokens=n.get("win_tokens", 0),
        block_acts=tuple(n.get("block_acts", [])), stream_act_at=tuple(n.get("stream_act_at", [])),
        warp_k=n.get("warp_k", 0), warp_hidden=n.get("warp_hidden", 64),
        warp_beta_init=n.get("warp_beta_init", 4.0),
        class_split=n.get("class_split", 0), class_gate_center=n.get("class_gate_center", -0.255),
        class_routes=n.get("class_routes", 2), route_hidden=n.get("route_hidden", 64),
        route_at=n.get("route_at", 0), route_by=n.get("route_by", "router"), route_pts=n.get("route_pts", 4),
        route_beta_in=n.get("route_beta_in", False),
    )

    train_cfg = TrainCfg(
        lambda_ic=lam["ic"],
        lambda_data=lam["data"],
        lambda_kin=lam.get("kin", 1.0),
        lambda_phys=lam.get("phys", 1.0),
        lambda_energy=lam.get("energy", 0.5),
        lambda_roll=lam.get("roll", 1.0),
        lr=o["lr"],
        weight_decay=o["weight_decay"],
        grad_clip=o["grad_clip"],
        grad_scale_max=t.get("grad_scale_max", 100.0),
        replay_frac=t.get("replay_frac", 0.25),
        n_colloc_cases=t.get("n_colloc_cases", c.get("n_colloc_cases", 0)),
        phys_ramp_center=t.get("phys_ramp_center", 30),
        phys_ramp_width=t.get("phys_ramp_width", 10),
        lr_drop_epoch=t.get("lr_drop_epoch", 0),
        lr_drop_to=t.get("lr_drop_to", 0.0),
        ema_decay=t.get("ema_decay", 0.999),
        roll_robust_k=t.get("roll_robust_k", 4.0),
        roll_balance_cases=t.get("roll_balance_cases", 128),
        roll_draws=t.get("roll_draws", 1),
        roll_median_ema=t.get("roll_median_ema", 0.0),
        roll_robust_kind=t.get("roll_robust_kind", "cauchy"),
        roll_target=t.get("roll_target", "label"), roll_rk4_dt=t.get("roll_rk4_dt", 0.005),
        trim_q_data=t.get("trim_q_data", 0.0), trim_q_phys=t.get("trim_q_phys", 0.0),
        trim_q_kin=t.get("trim_q_kin", 0.0),
        class_norm=t.get("class_norm", False), flip_w=t.get("flip_w", 1.0),
        class_norm_ema=t.get("class_norm_ema", 0.99),
        route_sup=t.get("route_sup", 1.0), route_sup_cases=t.get("route_sup_cases", 4096),
        gate_lam_frac=t.get("gate_lam_frac", 0.1),
        roll_balance_points=t.get("roll_balance_points", 12),
        ic_max_n=t.get("ic_max_n", 0),
        freeze_from=t.get("freeze_from", ""), freeze_below=t.get("freeze_below", 0),
        use_compile=t.get("use_compile", True),
        matmul_precision=t.get("matmul_precision", "high"),
        half_gemm=t.get("half_gemm", "off"),
        case_group=t.get("case_group", 1),
        warp_sup=t.get("warp_sup", 0.0), warp_sup_hold=t.get("warp_sup_hold", 0),
        warp_sup_end=t.get("warp_sup_end", 0), warp_sup_cases=t.get("warp_sup_cases", 4096),
        warp_sup_ratio=t.get("warp_sup_ratio", 0.25), warp_sup_label=t.get("warp_sup_label", "cross"),
    )

    colloc_cfg = CollocCfg(
        n_colloc=c["n_colloc"],
        colloc_flip_bias=c.get("colloc_flip_bias", 0.0),
        ic_sigma=c.get("ic_sigma", 0.0),
        ic_sigma_warmup=c.get("ic_sigma_warmup", 200),
        per_case=c.get("per_case", 8),
        sched_period=c.get("sched_period", 0),
        sched_modes=tuple(c.get("sched_modes", ["grid"])),
        sched_sigmas=tuple(c.get("sched_sigmas", [])),
    )

    data_cfg = DataCfg(
        g=d["g"],
        t_data_max=d["t_data_max"],
        march_dt=d.get("march_dt", 1.0),   # 구 config 하위호환 // backward-compat for pre-marching configs
        energy_eps=d["energy_eps"],
        phys_eps=d.get("phys_eps", 1.0),   # 구 config 하위호환 // backward-compat for configs predating relative phys loss
        kin_eps=d.get("kin_eps", 1.0),     # 구 config 하위호환 // backward-compat for configs predating relative kin loss
        batch_size=d["batch_size"],
        colloc_chunk=d.get("colloc_chunk", 0),
        nonflip_path=d["nonflip_path"],
        data_path=d.get("data_path", d["nonflip_path"]),
        scaler_name=d["scaler_name"],
        scaler_extra_omega=d.get("scaler_extra_omega", ""),
        batch_size_phase2=d.get("batch_size_phase2", 0),
        label_path=d.get("label_path", ""), label_score_path=d.get("label_score_path", ""),
        label_dt=d.get("label_dt", 0.0), label_dt_phase2=d.get("label_dt_phase2", 0.0),
        label_dense_frac=d.get("label_dense_frac", 0.0), slide_frac=d.get("slide_frac", 0.0),
        split_meta=d.get("split_meta", ""),
    )

    # 단위 주의: rel_tol_*은 에폭당 개선율 하한, patience_*는 에폭 수.
    #   StallTracker가 측정 간 경과 에폭으로 복리 환산하므로 측정 주기와 무관하게 게이트 간 잣대가 같다.
    #   rel_tol_*는 "그 지표가 아직 살아있는가"의 하한이지 목표 개선율이 아니다 — 작게 잡을 것.
    # phys 트래커 없음 — patience는 decay 시도 간격(에폭). 구 config의 ema_beta/rel_tol 키는 무시된다.
    ode_s_params = {
        "patience":        os_.get("patience", 80),      # decay 시도 간격(에폭) // 구 phys patience와 실효 동일
        "factor":          os_.get("factor", 0.75),
        "min_lr":          os_.get("min_lr", s.get("min_lr", 1e-6)),
        "phase2_lr":       os_.get("phase2_lr", 0.0),    # Phase 2 진입 1회 LR 캡 (0 = 비활성)
        # veto 예산: phys 정체인 채 연속 veto된 에폭이 이 값을 넘으면 강제 decay (0 = 무제한)
        #   게이트가 작은 개선 한 번에 리셋되는 구조라 veto가 영구 무장될 수 있다 // 손해 상한
        "max_veto_epochs": os_.get("max_veto_epochs", 0),
        "patience_val":    os_.get("patience_val", 20),
        "patience_ext":    os_.get("patience_ext", 50),
        "patience_roll":   os_.get("patience_roll", 75),
        "rel_tol_val":     os_.get("rel_tol_val", 4e-4),
        "rel_tol_ext":     os_.get("rel_tol_ext", 4e-4),
        "rel_tol_roll":    os_.get("rel_tol_roll", 4e-4),
        "ema_beta_val":    os_.get("ema_beta_val", 0.9),
        "ema_beta_ext":    os_.get("ema_beta_ext", 0.9),
        "ema_beta_roll":   os_.get("ema_beta_roll", 0.8),  # 측정이 드물어 약하게
    }

    return net_cfg, train_cfg, colloc_cfg, data_cfg, t, ode_s_params
    