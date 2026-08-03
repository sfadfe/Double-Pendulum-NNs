#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/common.sh"
[[ $EUID -eq 0 ]] || exec sudo "$0" "$@"

REPO=/tmp/omen-fan-control
H=/sys/devices/platform/hp-wmi/hwmon/hwmon6
RPM="${TARGET_RPM:-5000}"

systemctl stop hpm-fan.service 2>/dev/null || true
[[ -d "$REPO/.git" ]] || git clone --depth 1 https://github.com/arfelious/omen-fan-control.git "$REPO"
[[ -x "$REPO/install.sh" ]] && "$REPO/install.sh" || die "repo install.sh 확인"

modprobe -r hp_wmi 2>/dev/null || true
modprobe hp_wmi
[[ -r "$H/pwm1" ]] || die "pwm1 still missing on 8D42"

pwm=$(( RPM * 255 / 6200 ))
echo 1 > "$H/pwm1_enable"
echo "$pwm" > "$H/pwm1"
log "manual pwm=$pwm (~${RPM} RPM) — sensors hp-isa-0000 | grep fan"
