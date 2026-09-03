# Double-Pendulum Flow-Map PINN

[English](README.md) | 한국어

이중진자의 상태를 한 창(`march_dt` = 0.25 s)씩 앞으로 보내는 **flow map** 신경망.
궤적 전체를 한 번에 맞추는 대신, 창 끝 상태를 다음 창의 초기조건으로 넘겨 긴 궤적을 만든다.
학습은 데이터 항 + 물리 잔차(운동학·운동방정식·에너지) + 롤아웃(pushforward) 항으로 이루어진다.

![RK4 vs model](scratch/rk4_vs_model.gif)

원본 mp4: [`scratch/rk4_vs_model.mp4`](scratch/rk4_vs_model.mp4). 왼쪽: RK4 참값(회색)과 모델(색) 진자 겹침. 오른쪽: θ 궤적과 θ RMSE.
RK4는 데이터 생성과 같은 dt = 5e-6, 모델은 창당 25개 τ 지점 출력.

## 숫자로 보기

검증 케이스 0, 0–5 s 자유 마칭(20창) 기준.

| 항목 | 값 |
|---|---|
| 속도비 (RK4 dt 5e-6 / 모델) | 40× |
| θ RMSE 평균 / 최대 / 5 s | 1.3e-3 / 4.9e-3 / 2.9e-3 rad |
| θ RMSE 1e-3 초과 시각 | 0.60 s |
| 파라미터 | 1.61 M |
| RolloutBestVal (100 케이스 마칭 MSE) | 1.4e-5 |

## 문제 설정

입력 `(τ, IC, m1, m2, L1, L2)`, 출력 `[Δθ1, Δθ2, ω1, ω2]`, `τ ∈ [0, march_dt]`.

- θ는 sin/cos로만 들어가고 출력은 **Δθ = θ(τ) − θ_IC**. 절대각을 출력하면 감김수(2π)만 다른 두 창이
  입력은 같고 라벨은 다른 모순이 생긴다. Δθ는 τ=0에서 0이라 이 모순이 없다.
- 마칭: `θ_next = θ_start + Δθ(march_dt)`, `ω_next = ω(march_dt)`. 5 s = 20창.
- 창 하나의 오차가 다음 창의 IC 오차가 되고, 카오스(λ ≈ 0.66/s)가 이를 증폭한다. 한계 절 참고.

## 모델

```
τ ──Fourier(32, 0.2–20 Hz)──▶ trunk: 12 × FiLMBlock(256) ──▶ head_θ(2), head_ω(2)
                                      ▲ (γ_i, β_i)
IC(6) + ParamEmbed(m,L → 32) ──FiLMCond──┘
```

- trunk 입력은 τ뿐. 케이스 정보는 FiLM `(γ, β)`로 블록마다 주입: `h ← h + γ·SiLU(W h + b) + β`.
  skip은 무변조라 블록 야코비안이 `I + O(γ)`로 유계.
- `(γ, β)`는 τ와 무관하므로 케이스당 한 번 계산(`CondOf`)하고 롤아웃·jvp에서 재사용.
- hard IC (`c1`): 출력을 τ=0에서 Δθ=0, ω=ω_IC가 되도록 구조적으로 강제.

## 손실

| 항 | 내용 |
|---|---|
| `data` | Δθ, ω의 teacher-forced MSE |
| `kin` | `dΔθ/dt = ω` (jvp) |
| `phys` | `dω/dt = f(θ_IC+Δθ, ω)` 상대잔차 |
| `energy` | `E(τ) = E(IC)` 상대잔차 |
| `roll` | pushforward: no_grad로 여러 창을 굴려 off-manifold IC에 도달한 뒤 1창을 참 궤적에 맞춤 (Huber) |

항별 gradient 크기 균형(`grad_scale`), 커리큘럼 램프, 항별 gradient clip이 곱해진다.
콜로케이션은 매 스텝 iid uniform 재추첨.

## 한계

- 0–5 s 전 구간 1e-3 rad는 **달성하지 못했다**. 원인은 표현력이 아니라 카오스 증폭이다.
  창 하나의 오차가 20창을 지나며 ~80× 커지므로, 5 s 1e-3에는 창당 오차를 지금의 1/11 이하로
  줄여야 한다. march_dt를 줄여도 핸드오프 횟수가 늘어 상쇄된다.
- 1e-3 지평은 케이스에 따라 0.6–0.8 s. 이 범위 안에서는 창 경계(seam)에 스파이크가 없다.
- flip(회전) 궤적은 pretrain 대상이 아니다. `Train/finetune.py`가 flip 혼합 데이터로 파인튜닝하지만
  flip 오차는 nonflip보다 두 자릿수 크다.

## 재현

의존성: Python 3.12, PyTorch 2.12 (CUDA), NumPy, Numba, tqdm.

```bash
# 1. IC 추첨 → RK4 라벨 생성 (nonflip, dt 5e-6)
python state/states.py nonflip --n_total 70000 --out data/nonflip_70k_states.npy --seed 0
python state/dfss.py   nonflip --states data/nonflip_70k_states.npy --out data/nonflip_RK4_0_5s_70k.npy
# 여러 corpus를 합칠 때
python state/concat_corpora.py a.npy b.npy --out data/nonflip_RK4_0_5s_70k.npy

# 2. pretrain (config.toml 참고; 결과는 model/<timestamp>_<name>/)
python Train/train.py --config config.toml
python Train/train.py --resume model/<run> --ckpt latest        # 이어서

# 3. flip finetune
python Train/finetune.py --pretrain model/<run> --ckpt val --config <finetune.toml>
```

체크포인트: `best.pt`(teacher-forced val), `best_ode.pt`(물리 잔차), `best_extrap.pt`(외삽), `latest.pt`.
로그 `log.csv`는 최신 행이 위에 온다.

## 레이아웃

| 경로 | 역할 |
|---|---|
| `PINNsTrainer/` | 네트워크(`networks.py`), 손실(`Loss.py`), 학습 스텝·트레이너, LR 스케줄러, config 파서 |
| `state/` | 이중진자 EOM·RK4(`Double_pendulum.py`), IC 추첨(`states.py`), 라벨 생성(`dfss.py`) |
| `Train/` | 학습 루프(`train.py`, `finetune.py`, `loop_common.py`) |
| `config.toml` | 현재 기준 런 설정 |
