#!/bin/zsh
# Wrapper to run the telemetry updater from the repo directory.

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
WORKBOOK="${SCRIPT_DIR}/quantum_platform_telemetry_model_free_v2.xlsx"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/update_telemetry.log"
STAMP_FILE="${LOG_DIR}/last_run_epoch.txt"
# minimum spacing between runs (in seconds): 6.5 days (~156 hours) to give buffer
MIN_SPACING=$((6 * 24 * 3600 + 12 * 3600))

mkdir -p "$LOG_DIR"

# Skip if we ran recently (handles missed launchd windows due to sleep)
now_epoch=$(date +%s)
if [ -f "$STAMP_FILE" ]; then
  last_epoch=$(cat "$STAMP_FILE" 2>/dev/null || echo 0)
  if [ "$((now_epoch - last_epoch))" -lt "$MIN_SPACING" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skip: last run ${now_epoch-last_epoch}s ago (< spacing)" >> "$LOG_FILE"
    exit 0
  fi
fi

cd "$SCRIPT_DIR" || exit 1
date "+[%Y-%m-%d %H:%M:%S] Starting update" >> "$LOG_FILE"
python3 "$SCRIPT_DIR/update_telemetry_free.py" "$WORKBOOK" >> "$LOG_FILE" 2>&1
echo "[exit_code=$?]" >> "$LOG_FILE"
# record successful attempt time regardless of exit_code (prevents tight loop)
echo "$now_epoch" > "$STAMP_FILE"
