#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/common.sh"
log "kernel $(uname -r) board $(cat /sys/class/dmi/id/board_name)"
log "GPU $(nvidia-smi --query-gpu=power.default_limit,power.max_limit --format=csv,noheader)"
sensors hp-isa-0000 2>/dev/null | grep fan || true
[[ -x "$PY" ]] && "$PY" -c "import torch; print('cuda', torch.cuda.get_device_name(0))"
lsmod | grep -q omen_wmi_boost && cat /sys/kernel/omen_wmi_boost/gpu_state 2>/dev/null || log "omen_wmi_boost: not loaded"
[[ -r /sys/devices/platform/hp-wmi/hwmon/hwmon6/pwm1 ]] && log "pwm1: OK" || log "pwm1: 없음 (setup-fan.sh)"
