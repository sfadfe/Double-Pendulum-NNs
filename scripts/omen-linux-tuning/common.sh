#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/home/sfadfe/refactoring}"
PY="$ROOT/.venv/bin/python"
TRAIN="$ROOT/Train/train.py"
LOG="$ROOT/scripts/omen-linux-tuning/logs"
mkdir -p "$LOG"
log(){ printf '[%s] %s\n' "$(date +%T)" "$*"; }
die(){ log "ERR: $*"; exit 1; }

summarize(){
  awk -F, 'NR>1{if($3+0>x)x=$3+0;if($6+0>a)a=$6+0;if($7+0>b)b=$7+0} END{
    printf "max_gpu_W=%.0f max_fan1=%.0f max_fan2=%.0f\n",x+0,a+0,b+0}' "$1"
}

monitor_bg(){
  local out="$1" sec="${2:-2}"
  echo "ts,util,power_w,limit_w,temp,fan1,fan2" > "$out"
  while true; do
    read -r u p l t <<< "$(nvidia-smi --query-gpu=utilization.gpu,power.draw,power.limit,temperature.gpu \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ' W%')"
    f1=$(sensors hp-isa-0000 2>/dev/null | awk '/fan1:/ {print $2}')
    f2=$(sensors hp-isa-0000 2>/dev/null | awk '/fan2:/ {print $2}')
    echo "$(date +%s),$u,$p,$l,$t,${f1:-0},${f2:-0}" >> "$out"
    sleep "$sec"
  done
}
