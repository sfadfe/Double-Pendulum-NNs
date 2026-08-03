#!/usr/bin/env bash
set -euo pipefail
[[ $EUID -eq 0 ]] || exec sudo "$0" "$@"
omen fan auto 2>/dev/null || true
H=/sys/devices/platform/hp-wmi/hwmon/hwmon6
[[ -w $H/pwm1_enable ]] && echo 2 > "$H/pwm1_enable" || true
systemctl disable --now nvidia-powerd 2>/dev/null || true
modprobe -r omen_wmi_boost 2>/dev/null || true
systemctl start hpm-fan.service 2>/dev/null || true
echo "reverted (DKMS hp-wmi는 repo uninstall 수동)"
