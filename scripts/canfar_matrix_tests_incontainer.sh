#!/usr/bin/env bash
# One leg of the torchfits test matrix inside astroai/base on CANFAR, headless.
#
# This repo's launchers used to drive a grid of per-Python pixi environments.
# That grid no longer exists, so a leg provisions its own interpreter with uv
# instead and builds torchfits from the checkout it was launched in. Nothing
# here compiles the harness, and nothing writes under $HOME.
#
# Env:
#   TORCHFITS_MT_PY        CPython to test (3.11..3.15)            [required]
#   TORCHFITS_MT_DEVICE    cpu | cuda                              [cpu]
#   TORCHFITS_MT_CU        CUDA flavor when device=cuda (cu126|cu130) [cu130]
#   TORCHFITS_MT_TORCH     torch version to install                [2.14.1]
#   TORCHFITS_MT_SUITE     torchfits | harness | both              [torchfits]
#   TORCHFITS_MT_EXAMPLES  1 to run the bounded example gallery    [0]
#
# Prints one "[ OK ]" line per leg, or "[FAIL] ..." plus the tail of the pytest
# log; the launcher greps for them. Everything durable lands under $SCRATCH.
set -euo pipefail

: "${TORCHFITS_MT_PY:?set TORCHFITS_MT_PY}"
DEVICE="${TORCHFITS_MT_DEVICE:-cpu}"
CU="${TORCHFITS_MT_CU:-cu130}"
TORCH_VERSION="${TORCHFITS_MT_TORCH:-2.14.1}"
SUITE="${TORCHFITS_MT_SUITE:-torchfits}"
EXAMPLES="${TORCHFITS_MT_EXAMPLES:-0}"

SCRATCH="${TMP_SCRATCH_DIR:-/scratch}"
if [[ "${DEVICE}" == "cuda" ]]; then
  TAG="py${TORCHFITS_MT_PY//./}-${CU}"
else
  TAG="py${TORCHFITS_MT_PY//./}-cpu"
fi
WORK="${SCRATCH}/torchfits-matrix-tests/${TAG}"
RUN_LOG="${WORK}/leg.log"
mkdir -p "${WORK}" "${SCRATCH}/pip-cache" "${SCRATCH}/tmp" "${SCRATCH}/uv/bin"
export PIP_CACHE_DIR="${SCRATCH}/pip-cache"
export TMPDIR="${SCRATCH}/tmp"
export PYTHONNOUSERSITE=1
export UV_INSTALL_DIR="${SCRATCH}/uv/bin"
export UV_CACHE_DIR="${SCRATCH}/uv/cache"
export UV_PYTHON_INSTALL_DIR="${SCRATCH}/uv/python"

# No process substitution or tee: the platform captures stdout itself and a
# tee child dies with the session, which has swallowed the last lines of a run
# before. Per-command logs are written under ${WORK} instead.
echo "=== torchfits CANFAR matrix leg: python ${TORCHFITS_MT_PY} device ${DEVICE}${CU:+ (${CU})} torch ${TORCH_VERSION} ==="
echo "host=$(hostname) nproc=$(nproc 2>/dev/null || echo '?') work=${WORK}"
command -v nvidia-smi >/dev/null && { nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHFITS_DIR="$(cd "${REPO_ROOT}/../torchfits" 2>/dev/null && pwd || true)"
if [[ -z "${TORCHFITS_DIR}" || ! -f "${TORCHFITS_DIR}/pyproject.toml" ]]; then
  echo "[FAIL] ${TAG}: expected a torchfits checkout next to this repo (got '${TORCHFITS_DIR}')"
  exit 1
fi
echo "torchfits=$(git -C "${TORCHFITS_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# uv: a single static binary, so a leg needs no conda solve for its interpreter.
if ! command -v uv >/dev/null; then
  curl -fsSL https://astral.sh/uv/install.sh | sh
  export PATH="${UV_INSTALL_DIR}:${PATH}"
fi
uv --version

VENV="${WORK}/venv"
rm -rf "${VENV}"
uv venv --python "${TORCHFITS_MT_PY}" "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -V

if [[ "${DEVICE}" == "cuda" ]]; then
  TORCH_SPEC="torch==${TORCH_VERSION}"
  INDEX="https://download.pytorch.org/whl/${CU}"
else
  TORCH_SPEC="torch==${TORCH_VERSION}"
  INDEX="https://download.pytorch.org/whl/cpu"
fi
echo "--- installing ${TORCH_SPEC} from ${INDEX}"
pip install -q --upgrade pip
pip install -q "${TORCH_SPEC}" --index-url "${INDEX}"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "cuda", torch.version.cuda)
PY
if [[ "${DEVICE}" == "cuda" ]] && ! python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "[FAIL] ${TAG}: torch.cuda.is_available() is false on a GPU leg"
  exit 1
fi

echo "--- building torchfits from the checkout (C++20, vendored CFITSIO)"
pip install -q scikit-build-core nanobind cmake ninja numpy pyarrow pytest
pip install -q --no-build-isolation -e "${TORCHFITS_DIR}" --no-deps
python -c "import torchfits, torch; print('torchfits', torchfits.__version__, 'torch', torch.__version__)"

STATUS=0
if [[ "${SUITE}" == "torchfits" || "${SUITE}" == "both" ]]; then
  echo "--- pytest: torchfits tests/"
  if ! python -m pytest "${TORCHFITS_DIR}/tests" -q -m "not performance" \
      --ignore="${TORCHFITS_DIR}/tests/test_examples_runner.py" \
      -p no:cacheprovider > "${WORK}/pytest-torchfits.log" 2>&1; then
    STATUS=1
    tail -40 "${WORK}/pytest-torchfits.log"
    echo "[FAIL] ${TAG}: torchfits suite"
  else
    tail -3 "${WORK}/pytest-torchfits.log"
    echo "[ OK ] ${TAG}: torchfits suite"
  fi
fi

if [[ "${EXAMPLES}" == "1" ]]; then
  echo "--- example gallery (bounded)"
  if ! TORCHFITS_EXAMPLE_FAST=1 python "${TORCHFITS_DIR}/examples/test_examples.py" \
      > "${WORK}/examples.log" 2>&1; then
    STATUS=1
    tail -30 "${WORK}/examples.log"
    echo "[FAIL] ${TAG}: examples"
  else
    tail -3 "${WORK}/examples.log"
    echo "[ OK ] ${TAG}: examples"
  fi
fi

if [[ "${SUITE}" == "harness" || "${SUITE}" == "both" ]]; then
  echo "--- pytest: harness tests/"
  pip install -q fitsio astropy polars psutil
  if ! python -m pytest "${REPO_ROOT}/tests" -q -m "not performance" \
      -p no:cacheprovider > "${WORK}/pytest-harness.log" 2>&1; then
    STATUS=1
    tail -40 "${WORK}/pytest-harness.log"
    echo "[FAIL] ${TAG}: harness suite"
  else
    tail -3 "${WORK}/pytest-harness.log"
    echo "[ OK ] ${TAG}: harness suite"
  fi
fi

echo "leg=${TAG} status=${STATUS} log=${RUN_LOG}"
exit "${STATUS}"
