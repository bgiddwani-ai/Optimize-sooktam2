#!/usr/bin/env bash
# Start the BF16 AOTInductor Sooktam2 server inside the existing CUDA12 builder.
set -euo pipefail

readonly ROOT=/home/ubuntu/optimize-sooktam2
readonly BUILDER=sooktam2-build-cuda12
readonly SESSION=sooktam2_aoti_bf16
readonly LOG="$ROOT/logs/aoti_bf16_server.log"

docker inspect -f '{{.State.Running}}' "$BUILDER" | grep -qx true
mkdir -p "$ROOT/logs"
if screen -list | grep -q "[.]${SESSION}[[:space:]]"; then
  screen -S "$SESSION" -X quit
fi
screen -dmS "$SESSION" bash -lc \
  "exec docker exec $BUILDER env PYTHONPATH=/workspace/src/sooktam2/src python3 /workspace/aoti_bf16_server.py --port 8010 >$LOG 2>&1"
echo "started $SESSION; log: $LOG"
