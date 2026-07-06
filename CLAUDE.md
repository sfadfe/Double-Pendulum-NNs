코드 스타일

- 기본적으로 PEP 8 스타일을 사용하되 내가 아래 적은대로 변형해야함.
- 변수는 소문자, 언더바 사용
- bool인자를 지정하는 변수는 단어의 첫글자를 대문자로 작성, 언더바 x
- 함수는 클래스 이름 지을때처럼 단어의 첫 글자를 대문자로 작성, 언더바 x
- 주석은 영어설명 // 한글설명 형식으로 작성.
- 기능별로 주석을 작성하여 코드의 흐름을 설명하되, 너무 남발하지 말기.

모델 아키택쳐 설계도 이외에 참고할점

- FP64/FP32 혼합정밀도는 절대 포기하지 않는다.

- 내가 어디 폴더에 log.csv 봐달라고 하면 학습 로드 확인하고 분석하고 문제점 파악 이런거 해달라는건데 파일 전체 읽지 말고 중요한 부분만 골라서 읽기.
- offer.txt에 내가 너한테 평가받을 내용 적어두고 너한테 offer.txt에 적어준걸 읽어서 평가하라고 할거야.

모델 아키택쳐는 double_pendulum_pinns를 참고하기 무조건.

---

## 현재 설계 요약 (2026-06)

### Time-marching flow map (핵심 구조)

네트워크는 전역 시간 회귀가 아니라 **플로우맵** `f(τ, state, params) → state`. `[0, t_data_max]`를 길이 `march_dt` 윈도우로 분할(예: march_dt=1.0 → [0,1],[1,2],[2,3]). 네트워크 입력의 시간은 윈도우 내 **상대시간** `τ ∈ [0, march_dt]`이고, IC(trig·ω)는 **윈도우 시작 상태**.

- **학습**: 각 윈도우의 IC = 데이터의 실제 윈도우 시작 상태(teacher forcing). 윈도우들이 독립·병렬 학습되어 학습 중 오차 누적 없음. `ICLoss`는 τ=0에서 전체 상태(θ,ω) 일치를 강제 → 윈도우 연결부 연속성.
- **추론(`MarchRollout`)**: t=0 IC에서 시작, 윈도우 끝 예측 상태를 다음 윈도우 IC로 전달 → t>t_data_max 외삽. 연결부 mismatch 누적이 물리 학습의 진짜 테스트.
- `ComputeRollout`(val) = [0, t_data_max] 마칭 (stitching 오차 포함). `Test/extrapolate.py` = 마칭을 외삽 구간까지 돌려 RK4 GT와 비교.

### 학습 구조

**Phase 1 (epoch 0 ~ warmup_epochs-1):** data + IC 손실만 학습. `ComputeDataICLosses` 사용. 물리 경로(jvp) 없음.

**Phase 2 (epoch warmup_epochs ~):** `MeasureL0Physics`로 kin/phys/energy l0 측정 후 `ComputeAllLosses`로 전체 손실 학습.

### 손실 정규화

`NormalizedTotal = Σ lambda_k * (loss_k / l0_k)`

- `l0_k`: 각 손실 항의 초기 스케일 기준값. data/ic는 Phase 1 warmup_steps 동안 누적 평균. kin/phys/energy는 Phase 2 시작 시점에 `MeasureL0Physics`로 측정.
- `phys_loss`: EOM 잔차를 per-sample로 상대화 — `((dω/dt - f_eom) / (|f_eom| + phys_eps))²` (Loss.py). 저가속/고가속 구간 동등 비중. (구 `accel_rms` 전역 정규화는 폐기)
- `energy_loss`: `|E - E0| / (|E0| + energy_eps)` 상대 오차.

### Lambda 균형: ReLoBRaLo (Bischof & Kraus 2021)

sigmoid 전이 제거. 에폭마다 손실 변화율 기반으로 lambda 동적 조정.

```
rho_k(t)     = loss_k(t) / loss_k(t-1)          # 변화율: 클수록 수렴 안 됨
hat_lambda_k = n * softmax({rho_k / tau})_k * base_lambda_k
lambda_k(t)  = (1 - alpha) * hat_lambda_k + alpha * lambda_k(t-1)
```

- 중립 상태(모든 손실 같은 속도 수렴): lambda = base_lambda 유지.
- phys 정체 + 나머지 수렴: lambda_phys 증가, 나머지 감소.
- 주요 파라미터: `relobralo_tau` (softmax 온도), `relobralo_alpha` (EMA 평활, TC = 1/(1-alpha) 에폭).

### 네트워크

Fourier embedding + ResidualBlock × n. 입력: (τ, ic_trig, ic_ω, m, L) — τ는 윈도우 상대시간, ic는 윈도우 시작 상태. 출력: [θ1, θ2, ω1, ω2]. τ 정규화 기준은 `march_dt`(Fourier 주파수는 물리 Hz라 윈도우 무관).

**FP32 jvp**: `StateDerivs`는 `self(x)` 직접 호출(FP32) → dθ/dt, dω/dt 계산. `AngularAccel`/`GetEnergy`만 FP64 유지(수식 상쇄 보호). grad는 `.double()` 캐스트를 타고 FP32 파라미터로 환원. `functional_call` + FP64 jvp 폐기(2026-06-23, 에폭당 22s→5s). 모듈 dtype 토글 없음.

### 스케줄러

`OdeScheduler`: phys 상대잔차 EMA를 모니터링. `activate_threshold` 이하로 내려올 때부터 patience 기반 LR decay. `rel_tol_decay`/`patience_decay`로 후반 ratchet-down.

### 파일 구조

- `PINNsTrainer/Loss.py`: 손실 계산 (DataLoss, PhysicsEnergyLoss, ICLoss)
- `PINNsTrainer/Trainstep.py`: LambdaBalance (ReLoBRaLo) + TimeMarching (윈도우 분할, 콜로케이션, RAR, `MarchRollout`)
- `PINNsTrainer/Trainer.py`: PINNTrainer 메인 클래스 (mixin 조합)
- `Train/train.py`: 풀 학습 + Resume
- `Train/train_small.py`: 파라미터 테스트용 (train.py와 동일 루프, Resume 없음)
- `Sweep/sweep.py`: OAT 그리드 스윕 (train_small 멀티프로세스)
- `Test/extrapolate.py`: 마칭 외삽 평가 (보간 [0,t_data_max] vs 외삽 RK4 GT)

- Test/ : 니가 디버깅하는 코드 모아논데
- state/ RK4 data, initiak states 생성하는 코드
