#!/usr/bin/env bash
# usage: ./test-train.sh before|after
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/common.sh"
tag="${1:?before|after}"
sec="${TRAIN_SECONDS:-150}"
csv="$LOG/train-${tag}-$(date +%H%M%S).csv"

[[ -x "$PY" ]] || die "no venv"
command -v omen >/dev/null && { omen mode performance; omen fan max; }

monitor_bg "$csv" 2 & MP=$!
sleep 2
timeout "$sec" "$PY" "$TRAIN" || true
kill $MP 2>/dev/null; wait $MP 2>/dev/null || true
log "=== $tag ==="
summarize "$csv"
log "$csv"
