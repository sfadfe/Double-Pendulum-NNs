# 체크포인트 성능 종합 보고 — 티처포싱 / 0-5s 보간 / 외삽, 2026-08-02
#
# 세 체제를 같은 체크포인트·같은 held-out 케이스로 한 번에 재고, 전부 **RMSE**로 낸다.
#   로그의 Val/Rollout/Extrap 계열은 전부 MSE라 1e-3 목표와 직접 비교하면 100배 틀린다(trap 24).
#
#   [tf]     티처포싱 — 창 시작 IC를 참값으로 주는 단일 창 적합. 마칭 누적이 빠진 하한.
#   [interp] 0-5s 자유 마칭 — goal 1의 실제 판정 대상. 평균이 아니라 **전 구간 worst**를 본다.
#   [extrap] 학습 지평(t_data_max) 밖으로 계속 마칭 — 어디까지, 어느 오차로 가는가.
#            참값은 마지막 데이터 상태에서 RK4로 새로 적분한다(loop_common.BuildExtrapGT).
#
# 학습하지 않는다 — 순수 추론. 평가는 TF32 off(trap 25), 가중치는 EMA(best*.pt, trap 10).
#
#   python3 Train/eval_report.py --run model/2026_08_01_23_50_03
#   python3 Train/eval_report.py --run model/2026_08_01_23_50_03 --ckpt latest --t-ext 12

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from Train.loop_common import BuildExtrapGT, EvalPrecision, SelectRolloutCases
from Train.probe_common import BuildTrainer, FrameErrors

TARGET = 1.0e-3


def Crossings(t, e, levels=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1)):
    # 오차가 각 수준을 처음 넘는 시각 // "어디까지 어느 정확도로"를 한 줄로 답하는 표
    out = []
    for lv in levels:
        i = np.nonzero(e > lv)[0]
        out.append((lv, float(t[i[0]]) if len(i) else None))
    return out


def PrintCrossings(t, e, label):
    print(f"\n  {label} — RMSE가 각 수준을 처음 넘는 시각")
    for lv, tc in Crossings(t, e):
        s = f"t = {tc:.3f}s" if tc is not None else f"넘지 않음 (끝까지 < {lv:.0e})"
        print(f"    {lv:.0e}  {s}")


def TeacherForcing(trainer, batch_size):
    # 창 시작 IC가 참값인 단일 창 오차 // 마칭 누적이 없는 하한. τ 프로파일이 본론이다.
    trainer.eval()
    trainer.active_cases = trainer.val_cases
    th_all, om_all = [], []
    with EvalPrecision(), torch.no_grad():
        for t_lo, t_hi in trainer.segments:
            th_e, om_e, _, tau = FrameErrors(trainer, t_lo, t_hi, batch_size)
            th_all.append(th_e)
            om_all.append(om_e)
    th = torch.cat(th_all, 0)                        # (n_case*n_seg, tw, 2)
    om = torch.cat(om_all, 0)

    print(f"\n{'=' * 74}\n[tf] 티처포싱 (창 시작 IC = 참값)  케이스 {len(trainer.val_cases)} × 창 {len(trainer.segments)}\n{'=' * 74}")
    print(f"  전체 RMSE   θ {th.pow(2).mean().sqrt():.4e}   ω {om.pow(2).mean().sqrt():.4e}   (목표 {TARGET:.0e})")
    print(f"\n  창 안 τ-프로파일 (τ=0은 hard_ic로 항등적 0)")
    print(f"    {'τ':>6s} {'θ RMSE':>12s} {'ω RMSE':>12s} {'θ p99':>12s} {'ω p99':>12s}")
    tp = th.pow(2).mean(dim=2).sqrt()
    op = om.pow(2).mean(dim=2).sqrt()
    for i in range(0, tau.shape[0], max(1, tau.shape[0] // 10)):
        print(f"    {tau[i]:6.3f} {tp[:, i].pow(2).mean().sqrt():12.4e} {op[:, i].pow(2).mean().sqrt():12.4e}"
              f" {torch.quantile(tp[:, i], 0.99):12.4e} {torch.quantile(op[:, i], 0.99):12.4e}")
    i = tau.shape[0] - 1
    print(f"    {tau[i]:6.3f} {tp[:, i].pow(2).mean().sqrt():12.4e} {op[:, i].pow(2).mean().sqrt():12.4e}"
          f" {torch.quantile(tp[:, i], 0.99):12.4e} {torch.quantile(op[:, i], 0.99):12.4e}")
    print(f"\n  창 끝/중간 비  θ {tp[:, -1].pow(2).mean().sqrt() / tp[:, i // 2].pow(2).mean().sqrt():.2f}배"
          f"   ω {op[:, -1].pow(2).mean().sqrt() / op[:, i // 2].pow(2).mean().sqrt():.2f}배")


def Marching(trainer, case_idx, t_end, label, gt=None):
    # 자유 마칭 // 보간(t≤t_data_max)과 외삽(t>t_data_max)이 같은 코드 경로다
    dev = trainer.device
    n_win = int(round(t_end / trainer.dataCfg.march_dt))
    with EvalPrecision(), torch.no_grad():
        times, th_p, om_p = trainer.MarchRollout(case_idx, n_win)
    T = times.shape[0]
    if gt is None:
        gi = torch.arange(T, device=dev).clamp(max=trainer.n_step - 1)
        th_t = trainer.data[case_idx][:, gi][:, :, [1, 3]]
        om_t = trainer.data[case_idx][:, gi][:, :, [2, 4]]
    else:
        th_t, om_t = gt["th_true"][:, :T], gt["om_true"][:, :T]

    t = times.cpu().numpy()
    th_rmse = (th_p - th_t).pow(2).mean(dim=(0, 2)).sqrt().float().cpu().numpy()
    om_rmse = (om_p - om_t).pow(2).mean(dim=(0, 2)).sqrt().float().cpu().numpy()
    th_p99 = torch.quantile((th_p - th_t).pow(2).mean(dim=2).sqrt(), 0.99, dim=0).float().cpu().numpy()
    return t, th_rmse, om_rmse, th_p99


def Interp(trainer, case_idx):
    t_max = trainer.dataCfg.t_data_max
    t, th, om, th99 = Marching(trainer, case_idx, t_max, "interp")
    print(f"\n{'=' * 74}\n[interp] 0-{t_max:g}s 자유 마칭 (보간)  케이스 {len(case_idx)}  — goal 1의 판정 대상\n{'=' * 74}")
    print(f"  전 구간 평균 RMSE   θ {np.sqrt((th ** 2).mean()):.4e}   ω {np.sqrt((om ** 2).mean()):.4e}")
    i = int(th.argmax())
    print(f"  전 구간 **최악**    θ {th[i]:.4e} @ t={t[i]:.2f}s   (goal 1은 이 값이 {TARGET:.0e} 이하여야 성립)")
    print(f"  목표 대비          {th[i] / TARGET:.1f}배 초과")

    md = trainer.dataCfg.march_dt
    spw = max(1, int(round(md / trainer.dt)))
    print(f"\n  창별 (march_dt={md:g}s) θ RMSE — 창 시작 직후 / 창 끝, 그리고 seam 계단")
    print(f"    {'창':>3s} {'t 범위':>13s} {'창끝 θ':>11s} {'다음창 첫점':>12s} {'seam Δ':>11s} {'창끝 θ p99':>12s}")
    for w in range(len(t) // spw):
        i0, i1 = w * spw, min((w + 1) * spw, len(t) - 1)
        nxt = f"{th[i1 + 1]:12.4e}" if i1 + 1 < len(t) else f"{'-':>12s}"
        seam = f"{th[i1 + 1] - th[i1]:11.3e}" if i1 + 1 < len(t) else f"{'-':>11s}"
        print(f"    {w:3d} {t[i0]:5.2f}-{t[i1]:5.2f} {th[i1]:11.4e} {nxt} {seam} {th99[i1]:12.4e}")
    PrintCrossings(t, th, "θ (보간)")
    return t[-1]


def Extrap(trainer, case_idx, t_ext, rk4_dt):
    t_max = trainer.dataCfg.t_data_max
    print(f"\n{'=' * 74}\n[extrap] {t_max:g}s → {t_ext:g}s 외삽  케이스 {len(case_idx)}  (참값 = RK4 dt={rk4_dt:g} 재적분)\n{'=' * 74}")
    gt = BuildExtrapGT(trainer, case_idx, t_max, t_ext, rk4_dt)
    t, th, om, th99 = Marching(trainer, case_idx, t_ext, "extrap", gt)
    print(f"    {'t':>6s} {'θ RMSE':>12s} {'ω RMSE':>12s} {'θ p99':>12s}")
    step = max(1, len(t) // 24)
    for i in list(range(0, len(t), step)) + [len(t) - 1]:
        mark = " " if t[i] <= t_max else "*"
        print(f"   {mark}{t[i]:6.2f} {th[i]:12.4e} {om[i]:12.4e} {th99[i]:12.4e}")
    print("    (* = 학습 지평 밖)")
    PrintCrossings(t, th, "θ (외삽 포함 전 구간)")


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="model/2026_08_01_23_50_03")
    ap.add_argument("--ckpt", default="val", choices=["val", "latest", "ode", "extrap"])
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--t-ext", type=float, default=0.0, help="0이면 config의 extrap_t_ext")
    ap.add_argument("--batch-size", type=int, default=32768)
    ap.add_argument("--skip", default="", help="쉼표 구분: tf,interp,extrap")
    a = ap.parse_args()

    trainer, run_dir, _, _, _, data_cfg, t_params, dev = BuildTrainer(a.run, a.ckpt)
    skip = set(s.strip() for s in a.skip.split(",") if s.strip())
    t_ext = a.t_ext or t_params.get("extrap_t_ext", 6.0)   # 외삽 키는 [train]이 아니라 t_params에 산다
    print(f"[eval] run={a.run}  ckpt={a.ckpt}  device={dev}  march_dt={data_cfg.march_dt:g}  "
          f"t_data_max={data_cfg.t_data_max:g}  held-out 케이스 {len(trainer.val_cases)}개")
    print(f"       모든 수치는 **RMSE** (로그의 Val/Rollout/Extrap은 MSE — trap 24)")

    if "tf" not in skip:
        TeacherForcing(trainer, a.batch_size)
    case_idx = SelectRolloutCases(trainer, a.cases)
    if "interp" not in skip:
        Interp(trainer, case_idx)
    if "extrap" not in skip:
        Extrap(trainer, case_idx[: min(len(case_idx), 50)], t_ext, t_params.get("extrap_rk4_dt", 1e-4))


if __name__ == "__main__":
    Main()
