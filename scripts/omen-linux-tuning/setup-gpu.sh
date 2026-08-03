#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/common.sh"
[[ $EUID -eq 0 ]] || exec sudo "$0" "$@"

REPO=/tmp/5080-Unlock_Linux
[[ -d "$REPO/.git" ]] || git clone --depth 1 https://github.com/IzzieBoopers/5080-Unlock_Linux.git "$REPO"

# 8D42는 validated 목록 밖 → --force-unsupported 필수 (plain install.sh는 여기서 die)
"$REPO/install.sh" --force-unsupported

modprobe -r omen_wmi_boost 2>/dev/null || true
modprobe omen_wmi_boost || die "modprobe failed: journalctl -k | grep omen_wmi | tail"
cat /sys/kernel/omen_wmi_boost/gpu_state

UNIT=/etc/systemd/system/nvidia-powerd.service
if [[ ! -f "$UNIT" ]]; then
  cat > "$UNIT" <<'EOF'
[Unit]
Description=NVIDIA Power Daemon
After=nvidia-persistenced.service
[Service]
ExecStart=/usr/bin/nvidia-powerd
Restart=on-failure
[Install]
WantedBy=multi-user.target
EOF
fi
systemctl daemon-reload
systemctl enable --now nvidia-powerd
log "done — ./test-train.sh after"
