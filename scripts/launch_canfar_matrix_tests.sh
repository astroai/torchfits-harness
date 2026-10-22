#!/usr/bin/env bash
# Launch torchfits test legs on CANFAR headless: one session per
# (CPython, device) pair, each running scripts/canfar_matrix_tests_incontainer.sh
# from a fresh clone of the fork.
#
# Usage:
#   bash scripts/launch_canfar_matrix_tests.sh                 # all legs
#   bash scripts/launch_canfar_matrix_tests.sh --pythons 3.13 --devices cpu,cuda
#   bash scripts/launch_canfar_matrix_tests.sh --collect-only <run dir>
#
# --collect-only reads the sessions recorded in a previous run directory and
# prints their verdicts without launching anything, so a run whose collection
# was interrupted is never thrown away.
#
# Env:
#   TORCHFITS_MT_REPO    clone source           [sfabbro/torchfits]
#   TORCHFITS_MT_REF     branch or sha          [main]
#   TORCHFITS_MT_TORCH   torch version          [2.14.1]
#   TORCHFITS_MT_CU      CUDA flavor for gpu    [cu130]
#   TORCHFITS_MT_SUITE   torchfits|harness|both [torchfits]
#   TORCHFITS_MT_EXAMPLES 1 to run examples     [0]
#   TORCHFITS_MT_REAL_DATA 1 to fetch the ~2.5 GB CFHT sample per leg [0]
#   TORCHFITS_MT_CPU / _MEMORY / _MAX_WAIT_SECS / _IMAGE
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONS="3.11,3.12,3.13,3.14,3.15"
DEVICES="cpu,cuda"
COLLECT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pythons) PYTHONS="$2"; shift 2 ;;
    --devices) DEVICES="$2"; shift 2 ;;
    --collect-only) COLLECT_DIR="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO="${TORCHFITS_MT_REPO:-sfabbro/torchfits}"
REF="${TORCHFITS_MT_REF:-main}"
TORCH_VERSION="${TORCHFITS_MT_TORCH:-2.14.1}"
CU="${TORCHFITS_MT_CU:-cu130}"
SUITE="${TORCHFITS_MT_SUITE:-torchfits}"
EXAMPLES="${TORCHFITS_MT_EXAMPLES:-0}"
REAL_DATA="${TORCHFITS_MT_REAL_DATA:-0}"
IMAGE="${TORCHFITS_MT_IMAGE:-images.canfar.net/astroai/base:latest}"
CPU="${TORCHFITS_MT_CPU:-8}"
MEMORY="${TORCHFITS_MT_MEMORY:-32}"
MAX_WAIT="${TORCHFITS_MT_MAX_WAIT_SECS:-7200}"
POLL="${TORCHFITS_MT_POLL_SECS:-30}"
RUN_ID="${TORCHFITS_MT_RUN_ID:-matrix_tests_$(date -u +%Y%m%d_%H%M%S)}"
if [[ -n "$COLLECT_DIR" ]]; then
  OUT="$(cd "$COLLECT_DIR" && pwd)"
  RUN_ID="$(basename "$OUT")"
else
  OUT="${ROOT_DIR}/benchmarks_results/canfar_${RUN_ID}"
  mkdir -p "$OUT"
fi

command -v canfar >/dev/null || { echo "canfar CLI not found" >&2; exit 1; }

session_status() {
  canfar info "$1" 2>/dev/null | awk '$1 == "Status" { print $2; exit }'
}

launch_leg() {
  local py="$1" device="$2"
  local tag="py${py//./}-${device}"
  local dir="${OUT}/${tag}"
  mkdir -p "$dir"
  local name="tf-${tag}-${RUN_ID}"
  name="$(printf '%s' "$name" | tr '_.' '--' | tr -cd '[:alnum:]-')"
  # macOS ships bash 3.2, where expanding an empty array under `set -u` is an
  # error, so the flag is a plain string that may be empty.
  local gpu_args=""
  [[ "$device" == "cuda" ]] && gpu_args="--gpu 1"

  echo "== launching py=${py} device=${device} (${name})"
  # The harness and torchfits live in different repositories; clone both, then
  # run the leg from the harness checkout (its script expects `../torchfits`).
  local remote
  remote="git clone --depth 1 https://github.com/sfabbro/torchfits-harness.git /tmp/torchfits-harness; git clone --depth 1 --branch ${REF} https://github.com/${REPO}.git /tmp/torchfits; bash /tmp/torchfits-harness/scripts/canfar_matrix_tests_incontainer.sh"
  local tabs
  tabs="$(printf '%s' "$remote" | tr ' ' '\t')"

  if ! canfar create headless "$IMAGE" \
      --name "$name" \
      --cpu "$CPU" \
      --memory "$MEMORY" \
      ${gpu_args} \
      --env "TORCHFITS_MT_PY=${py}" \
      --env "TORCHFITS_MT_DEVICE=${device}" \
      --env "TORCHFITS_MT_CU=${CU}" \
      --env "TORCHFITS_MT_TORCH=${TORCH_VERSION}" \
      --env "TORCHFITS_MT_SUITE=${SUITE}" \
      --env "TORCHFITS_MT_EXAMPLES=${EXAMPLES}" \
      --env "TORCHFITS_MT_REAL_DATA=${REAL_DATA}" \
      -- bash -c "${tabs}" > "${dir}/create.log" 2>&1; then
    echo "[FAIL] ${tag}: session create failed (see ${dir}/create.log)"
    return 1
  fi

  local session_id
  session_id="$(python - "${dir}/create.log" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"ID:\s*([^)]+)\)", text)
print(m.group(1).strip() if m else "")
PY
)"
  if [[ -z "$session_id" ]]; then
    echo "[FAIL] ${tag}: could not parse session id (see ${dir}/create.log)"
    return 1
  fi
  echo "$session_id" > "${dir}/session_id.txt"
  echo "   session ${session_id}"
  return 0
}

echo "=== CANFAR torchfits test matrix (run ${RUN_ID}) ==="
echo "repo=${REPO} ref=${REF} torch=${TORCH_VERSION} cu=${CU} suite=${SUITE} examples=${EXAMPLES} real_data=${REAL_DATA}"
echo "pythons=${PYTHONS} devices=${DEVICES} out=${OUT}"

FAILED=0
if [[ -z "$COLLECT_DIR" ]]; then
  IFS="," read -r -a PY_LIST <<< "$PYTHONS"
  IFS="," read -r -a DEV_LIST <<< "$DEVICES"
  for py in "${PY_LIST[@]}"; do
    for device in "${DEV_LIST[@]}"; do
      launch_leg "$py" "$device" || FAILED=$((FAILED + 1))
      sleep 2
    done
  done
fi

echo "=== launched; collecting (max wait ${MAX_WAIT}s) ==="
START="$(date +%s)"
for dir in "$OUT"/*/; do
  [[ -f "${dir}/session_id.txt" ]] || continue
  session_id="$(cat "${dir}/session_id.txt")"
  tag="$(basename "$dir")"
  while true; do
    # awk, not sed: the CLI pads its labels with spaces and BSD sed has no \s,
    # so a sed-based parse never matched and every run waited out MAX_WAIT even
    # though its legs had finished.
    status="$(session_status "$session_id")"
    case "$status" in Succeeded | Completed | Failed | Error | Terminating) break ;; esac
    if (( $(date +%s) - START > MAX_WAIT )); then
      echo "[FAIL] ${tag}: still ${status:-unknown} after ${MAX_WAIT}s"
      break
    fi
    sleep "$POLL"
  done
  canfar logs "$session_id" > "${dir}/canfar_logs.txt" 2>&1 || true
  if grep -q '^\[ OK \]' "${dir}/canfar_logs.txt"; then
    grep '^\[ OK \]' "${dir}/canfar_logs.txt"
  else
    echo "[FAIL] ${tag}: session ${status:-unknown}, no [ OK ] marker"
    grep -E '^\[FAIL\]' "${dir}/canfar_logs.txt" | head -5 || true
    FAILED=$((FAILED + 1))
  fi
done

echo "=== summary: ${FAILED} failing legs, results in ${OUT} ==="
[[ "$FAILED" -eq 0 ]]
