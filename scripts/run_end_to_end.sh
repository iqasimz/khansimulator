#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ALGO="${ALGO:-ppo}"
TIMESTEPS="${TIMESTEPS:-200000}"
LOGDIR="${LOGDIR:-runs/room_rl}"
EPISODES="${EPISODES:-5}"
OUTDOOR_NOISE="${OUTDOOR_NOISE:-0.5}"
PROCESS_NOISE="${PROCESS_NOISE:-0.1}"
EPISODE_HOURS="${EPISODE_HOURS:-72}"
VALIDATE_HOURS="${VALIDATE_HOURS:-6}"
VALIDATE_SETPOINT="${VALIDATE_SETPOINT:-24}"
VALIDATE_INITIAL="${VALIDATE_INITIAL:-30}"

CSV_PATH="${CSV_PATH:-/tmp/dynamics.csv}"
PLOT_PATH="${PLOT_PATH:-/tmp/dynamics.png}"

echo "==> Validating dynamics"
python "$ROOT_DIR/scripts/validate_dynamics.py" \
  --outdoor-profile \
  --hours "$VALIDATE_HOURS" \
  --setpoint "$VALIDATE_SETPOINT" \
  --initial "$VALIDATE_INITIAL" \
  > "$CSV_PATH"
python "$ROOT_DIR/scripts/plot_dynamics.py" "$CSV_PATH" --out "$PLOT_PATH"
echo "Saved plot to $PLOT_PATH"

echo "==> Training $ALGO"
python "$ROOT_DIR/scripts/train_rl.py" \
  --algo "$ALGO" \
  --timesteps "$TIMESTEPS" \
  --logdir "$LOGDIR" \
  --episode-hours "$EPISODE_HOURS" \
  --outdoor-noise "$OUTDOOR_NOISE" \
  --process-noise "$PROCESS_NOISE"

echo "==> Evaluating $ALGO"
python "$ROOT_DIR/scripts/eval_rl.py" \
  --algo "$ALGO" \
  --model "$LOGDIR/${ALGO}_room_model.zip" \
  --vecnorm "$LOGDIR/vecnormalize.pkl" \
  --episodes "$EPISODES" \
  --episode-hours "$EPISODE_HOURS" \
  --outdoor-noise "$OUTDOOR_NOISE" \
  --process-noise "$PROCESS_NOISE"
