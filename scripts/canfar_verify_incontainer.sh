#!/usr/bin/env bash
# Run the whole torchfits-harness suite inside an astroai/base CANFAR session.
#
# Companion to canfar_gpu_bench_incontainer.sh: that one produces benchmark
# scorecards, this one proves the suite itself is green against a real wheel.
# The subject is an installed torchfits wheel - this repository never builds
# torchfits and never vendors CFITSIO.
#
#   canfar create headless images.canfar.net/astroai/base:latest --name \
#     torchfits-harness-verify --cpu 8 --memory 32 \
#     -- bash -c "<tab-encoded: git clone ...; cd ...; bash scripts/canfar_verify_incontainer.sh>"
#
# Env:
#   TORCHFITS_HARNESS_RUN_ID   run id (default: verify_<UTC timestamp>)
#   TORCHFITS_WHEEL            pip spec or path for the subject
#   TORCHFITS_VERSION          exact release to install (wins over the default
#                              floor; use this from `canfar --env`, which
#                              drops values containing `>` or `==`)

#   TORCHFITS_HARNESS_PYTEST_ARGS  extra pytest args (default: -q -m "not performance")
set -euo pipefail

: "${TORCHFITS_HARNESS_RUN_ID:=verify_$(date -u +%Y%m%d_%H%M%S)}"
SCRATCH="${TMP_SCRATCH_DIR:-/scratch}"
RUN_DIR="${SCRATCH}/torchfits-harness/${TORCHFITS_HARNESS_RUN_ID}"
mkdir -p "$RUN_DIR"

# notebook/base images ship PIXI_HOME=/usr/local/share/pixi, which the job user
# cannot write, so only an explicit TORCHFITS_PIXI_HOME may override scratch.
export PIXI_HOME="${TORCHFITS_PIXI_HOME:-${SCRATCH}/torchfits-pixi-home}"
export PIXI_CACHE_DIR="${TORCHFITS_PIXI_CACHE:-${SCRATCH}/torchfits-pixi-cache}"
mkdir -p "$PIXI_HOME" "$PIXI_CACHE_DIR"

# Keep user-site off the shared /arc home; stale ~/.local packages break
# setuptools uninstalls across concurrent sessions.
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${SCRATCH}/pip-cache"
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The image ships some pixi, but pixi.toml pins one. Fetch the pinned release
# into scratch instead of self-updating: the image's pixi lives in a read-only
# prefix, and `requires-pixi` otherwise refuses to run at all.
use_pinned_pixi() {
  local want have arch dir
  want="$(sed -n 's/^requires-pixi *= *"==\([^"]*\)".*/\1/p' pixi.toml | head -1)"
  if [[ -z "${want}" ]]; then
    return 0
  fi
  have="$(pixi --version 2>/dev/null | awk '{print $2}')"
  if [[ "${have}" == "${want}" ]]; then
    return 0
  fi
  case "$(uname -m)" in
    x86_64) arch=x86_64 ;;
    aarch64 | arm64) arch=aarch64 ;;
    *) echo "FAIL: unsupported arch $(uname -m) for the pixi download" >&2; return 1 ;;
  esac
  dir="${SCRATCH}/pixi-${want}"
  mkdir -p "$dir"
  echo "pixi ${have:-missing} != pinned ${want}; fetching v${want} into ${dir}"
  curl -fsSL --retry 3 -o "${dir}/pixi" \
    "https://github.com/prefix-dev/pixi/releases/download/v${want}/pixi-${arch}-unknown-linux-musl"
  chmod +x "${dir}/pixi"
  export PATH="${dir}:${PATH}"
  pixi --version
}

# stderr -> stdout only (process substitution on stderr loses the exit code).
exec > >(tee -a "${RUN_DIR}/stdout.log") 2>&1

NPROC="$(nproc 2>/dev/null || echo 4)"
echo "=== torchfits-harness headless verify ==="
echo "run_id=${TORCHFITS_HARNESS_RUN_ID} run_dir=${RUN_DIR}"
echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown) nproc=${NPROC}"
echo "PIXI_HOME=${PIXI_HOME} PIXI_CACHE_DIR=${PIXI_CACHE_DIR}"
python3 --version
command -v pixi >/dev/null || { echo "FAIL: pixi not on PATH in this image"; exit 1; }
use_pinned_pixi

pixi install

# The subject: a released wheel, never this checkout. TORCHFITS_WHEEL takes any
# pip spec or path; TORCHFITS_VERSION pins an exact release by number, because
# Skaha drops env values that carry `>` or `==`.
if [[ -n "${TORCHFITS_WHEEL:-}" ]]; then
  WHEEL_SPEC="${TORCHFITS_WHEEL}"
elif [[ -n "${TORCHFITS_VERSION:-}" ]]; then
  WHEEL_SPEC="torchfits==${TORCHFITS_VERSION}"
else
  WHEEL_SPEC="torchfits>=1.2.1"
fi
echo "subject spec: ${WHEEL_SPEC}"
pixi run python -m pip install --no-cache-dir --no-deps "${WHEEL_SPEC}"
pixi run dev
pixi run python -c "import torchfits_harness; print('subject:', torchfits_harness.subject())"

rc_gate=0
rc_suite=0

echo "=== gate: pixi run test ==="
pixi run test || rc_gate=$?

echo "=== suite: pytest tests/ ==="
# shellcheck disable=SC2086
pixi run python -m pytest tests/ ${TORCHFITS_HARNESS_PYTEST_ARGS:--q -m "not performance"} \
  --junitxml="${RUN_DIR}/junit.xml" || rc_suite=$?

echo "=== bounded bench: pixi run bench-cpp ==="
rc_bench=0
pixi run bench-cpp || rc_bench=$?

echo "=== summary ==="
echo "gate_rc=${rc_gate} suite_rc=${rc_suite} bench_rc=${rc_bench} run_dir=${RUN_DIR}"
if [[ "${rc_gate}" -eq 0 && "${rc_suite}" -eq 0 && "${rc_bench}" -eq 0 ]]; then
  echo "RESULT: PASS"
else
  echo "RESULT: FAIL"
fi
# One failing stage must not hide the other's result.
if [[ "${rc_gate}" -ne 0 || "${rc_suite}" -ne 0 || "${rc_bench}" -ne 0 ]]; then
  exit 1
fi
