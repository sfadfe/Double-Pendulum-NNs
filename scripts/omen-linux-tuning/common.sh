#!/usr/bin/env bash
set -euo pipefail

REFACTORING_ROOT="${REFACTORING_ROOT:-/home/sfadfe/refactoring}"
VENV_PY="${VENV_PY:-$REFACTORING_ROOT/.venv/bin/python}"
TRAIN_PY="${TRAIN_PY:-$REFACTORING_ROOT/Train/train.py}"
LOG_DIR="${LOG_DIR:-$REFACTORING_ROOT/scripts/omen-linux-tuning/logs}"
REPO_BOOST="${REPO_BOOST:-/tmp/5080-Unlock_Linux}"
REPO_FAN="${REPO_FAN:-/tmp/omen-fan-control}"

mkdir -p "$LOG_DIR"

log() { printf "[%s] %s\n" "$(date "+%F %T")" "$*"; }
die() { log "ERROR: $*"; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

need_sudo() {
  if ! sudo -n true 2>/dev/null; then
    die "sudo NOPASSWD required, or run in a terminal: sudo -v then re-run"
  fi
}

gpu_snapshot() {
  nvidia-smi --query-gpu=name,utilization.gpu,power.draw,power.limit,power.default_limit,power.max_limit,temperature.gpu \
    --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed"
}

fan_snapshot() {
  sensors hp-isa-0000 2>/dev/null | grep -E "^fan[12]:" || true
  if [[ -r /sys/devices/platform/hp-wmi/hwmon/hwmon6/pwm1_enable ]]; then
    echo "pwm1_enable=$(cat /sys/devices/platform/hp-wmi/hwmon/hwmon6/pwm1_enable)"
  fi
}

omen_performance() {
  if command -v omen >/dev/null; then
    omen mode performance || true
    omen fan max || true
  fi
}

summarize_csv() {
  local csv="$1"
  [[ -f "$csv" ]] || { echo "no csv: $csv"; return 1; }
  awk -F, 'NR>1 {
      if ($3+0 > maxp) maxp = $3+0
      if ($2+0 > maxu) maxu = $2+0
      if ($6+0 > f1) f1 = $6+0
      if ($7+0 > f2) f2 = $7+0
      lim = $4
    }
    END {
      printf "samples=%d max_gpu_power_w=%.1f max_gpu_util=%.0f max_fan1=%.0f max_fan2=%.0f last_limit=%s\n",
        NR-1, maxp+0, maxu+0, f1+0, f2+0, lim
    }' "$csv"
}
