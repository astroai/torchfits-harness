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
#   TORCHFITS_MT_REAL_DATA 1 to fetch the ~2.5 GB CFHT sample      [0]
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
REAL_DATA="${TORCHFITS_MT_REAL_DATA:-0}"

SCRATCH="${TMP_SCRATCH_DIR:-/scratch}"
if [[ "${DEVICE}" == "cuda" ]]; then
  TAG="py${TORCHFITS_MT_PY//./}-${CU}"
else
  TAG="py${TORCHFITS_MT_PY//./}-cpu"
fi
WORK="${SCRATCH}/torchfits-matrix-tests/${TAG}"
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

TORCH_SPEC="torch==${TORCH_VERSION}"
if [[ "${DEVICE}" == "cuda" ]]; then
  INDEX="https://download.pytorch.org/whl/${CU}"
else
  INDEX="https://download.pytorch.org/whl/cpu"
fi
echo "--- installing ${TORCH_SPEC} from ${INDEX}"
# uv owns the venv and has no `pip` shim in it; the image's global pip
# would try to write /usr/local/bin/pip and fail.
uv pip install -q "${TORCH_SPEC}" --index-url "${INDEX}"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "cuda", torch.version.cuda)
PY
if [[ "${DEVICE}" == "cuda" ]] && ! python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "[FAIL] ${TAG}: torch.cuda.is_available() is false on a GPU leg"
  exit 1
fi

# The image carries the zlib runtime but no headers, and the build treats ZLIB
# as required for compressed FITS. Reuse any usable zlib prefix; otherwise build
# one under ${WORK} and point CMake at it, instead of skipping the feature.
find_zlib_prefix() {
  local prefix lib
  for prefix in "${CONDA_PREFIX:-}" /usr/local /usr; do
    [[ -n "${prefix}" && -f "${prefix}/include/zlib.h" ]] || continue
    for lib in "${prefix}/lib/libz.so" "${prefix}/lib/libz.a" \
               "${prefix}/lib64/libz.so" "${prefix}/lib64/libz.a"; do
      [[ -e "${lib}" ]] && { printf '%s' "${prefix}"; return 0; }
    done
  done
  return 1
}
printf '%s' "--- zlib: "
if ZLIB_PREFIX="$(find_zlib_prefix)"; then
  echo "image prefix ${ZLIB_PREFIX}"
else
  ZLIB_PREFIX="${WORK}/deps/zlib"
  ZLIB_VERSION="1.3.1"
  ZLIB_SRC="${WORK}/deps/src/zlib-${ZLIB_VERSION}"
  echo "building ${ZLIB_VERSION} into ${ZLIB_PREFIX}"
  mkdir -p "${WORK}/deps/src"
  curl -fsSL "https://github.com/madler/zlib/releases/download/v${ZLIB_VERSION}/zlib-${ZLIB_VERSION}.tar.gz" \
    | tar -xz -C "${WORK}/deps/src"
  ( cd "${ZLIB_SRC}" && ./configure --prefix="${ZLIB_PREFIX}" >/dev/null \
      && make -j"$(nproc)" >/dev/null && make install >/dev/null )
fi

# scikit-build-core forwards this to CMake, so FindZLIB resolves the prefix above.
export CMAKE_ARGS="${CMAKE_ARGS:-} -DZLIB_ROOT=${ZLIB_PREFIX}"

# Same story for bzip2: without headers the vendored CFITSIO builds with the
# BZIP2_1 codec compiled out and six tests skip, so the leg would silently cover
# less than the wheel does. CMake reads ENV{PREFIX} for the vendored path.
find_bzip2_prefix() {
  local prefix
  for prefix in "${CONDA_PREFIX:-}" /usr/local /usr; do
    [[ -n "${prefix}" && -f "${prefix}/include/bzlib.h" ]] || continue
    [[ -e "${prefix}/lib/libbz2.a" || -e "${prefix}/lib64/libbz2.a" ]] \
      && { printf '%s' "${prefix}"; return 0; }
  done
  return 1
}
printf '%s' "--- bzip2: "
if BZIP2_PREFIX="$(find_bzip2_prefix)"; then
  echo "image prefix ${BZIP2_PREFIX}"
else
  BZIP2_PREFIX="${WORK}/deps/bzip2"
  echo "building 1.0.8 into ${BZIP2_PREFIX}"
  mkdir -p "${WORK}/deps/src"
  curl -fsSL "https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz" \
    | tar -xz -C "${WORK}/deps/src"
  ( cd "${WORK}/deps/src/bzip2-1.0.8" && make -j"$(nproc)" >/dev/null \
      && make install PREFIX="${BZIP2_PREFIX}" >/dev/null )
fi

echo "--- building torchfits from the checkout (C++20, vendored CFITSIO)"
uv pip install -q scikit-build-core nanobind cmake ninja numpy pyarrow pytest
PREFIX="${BZIP2_PREFIX}" uv pip install -q --no-build-isolation -e "${TORCHFITS_DIR}" --no-deps
python -c "import torchfits, torch; print('torchfits', torchfits.__version__, 'torch', torch.__version__)"

STATUS=0
# Real CFHT frames are the only input whose expected values were not written by
# us, and the tests that use them skip without the samples. Keep the data on
# $SCRATCH (the clone lives in /tmp) and point the checkout at it, so the
# resolution inside the tests stays untouched.
if [[ "${REAL_DATA}" == "1" ]]; then
  DATA_ROOT="${SCRATCH}/torchfits-matrix-data"
  mkdir -p "${DATA_ROOT}"
  ln -sfn "${DATA_ROOT}" "${TORCHFITS_DIR}/benchmarks_data"
  echo "--- fetching the CFHT samples into ${DATA_ROOT}"
  # Both samples: the tests that need one half skip when it is missing, and a
  # leg that never proves the mosaics is not the run the wheel is judged on.
  for fetch in fetch_cfht_megacam_sample fetch_cfht_megapipe_sample; do
    if ! bash "${TORCHFITS_DIR}/scripts/${fetch}.sh" \
        > "${WORK}/${fetch}.log" 2>&1; then
      STATUS=1
      tail -10 "${WORK}/${fetch}.log"
      echo "[FAIL] ${TAG}: ${fetch}"
    fi
  done
  echo "$(find "${DATA_ROOT}" -name '*.fits*' | wc -l) sample files present"
fi

if [[ "${SUITE}" == "torchfits" || "${SUITE}" == "both" ]]; then
  echo "--- pytest: torchfits tests/"
  # Run from the harness checkout on purpose: a test that only passes from the
  # repository root is a bug, and this is the cheapest place to find it (-rs
  # records why anything was skipped).
  if ! python -m pytest "${TORCHFITS_DIR}/tests" -q -rs -m "not performance" \
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
  uv pip install -q fitsio astropy polars psutil
  if ! python -m pytest "${REPO_ROOT}/tests" -q -rs -m "not performance" \
      -p no:cacheprovider > "${WORK}/pytest-harness.log" 2>&1; then
    STATUS=1
    tail -40 "${WORK}/pytest-harness.log"
    echo "[FAIL] ${TAG}: harness suite"
  else
    tail -3 "${WORK}/pytest-harness.log"
    echo "[ OK ] ${TAG}: harness suite"
  fi
fi

echo "leg=${TAG} status=${STATUS} logs=${WORK}"
exit "${STATUS}"
