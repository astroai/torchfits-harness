#!/usr/bin/env bash
# Build a torchfits wheel from a sibling checkout and install it as the subject.
#
#   pixi run subject-local                 # ../torchfits
#   TORCHFITS_SOURCE=/path/to/torchfits pixi run subject-local
#
# The wheel is built with the checkout's own pixi environment, which carries
# scikit-build-core, nanobind and the lane torch, so nothing is added to this
# environment. Build products go to a temporary directory, not the checkout.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="${TORCHFITS_SOURCE:-${REPO_ROOT}/../torchfits}"
if [[ ! -f "${TF_DIR}/pyproject.toml" ]]; then
  echo "subject-local: no torchfits checkout at ${TF_DIR}; set TORCHFITS_SOURCE" >&2
  exit 1
fi
TF_DIR="$(cd "$TF_DIR" && pwd)"

HARNESS_PY="$(command -v python)"
want="$("$HARNESS_PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/torchfits-subject.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

# Run the build outside this environment's activation, or the checkout's pixi
# would inherit this env's CONDA_PREFIX and PATH.
build_env=(env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PIXI_PROJECT_ROOT
  -u PIXI_PROJECT_MANIFEST -u PIXI_ENVIRONMENT_NAME -u PIXI_IN_SHELL
  -u PIXI_EXE -u PIXI_PROJECT_NAME -u PIXI_PROJECT_VERSION)
have="$(cd "$TF_DIR" && "${build_env[@]}" pixi run python -c \
  'import sys; print("%d.%d" % sys.version_info[:2])' | tail -1)"
if [[ "$have" != "$want" ]]; then
  echo "subject-local: ${TF_DIR} builds for python ${have}, this env runs ${want}" >&2
  exit 1
fi

echo "--- building torchfits wheel from ${TF_DIR} ($(git -C "$TF_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown))"
(cd "$TF_DIR" && "${build_env[@]}" pixi run python -m pip wheel . \
  --no-build-isolation --no-deps -w "$TMP/dist" \
  --config-settings=build-dir="$TMP/build")

wheel=("$TMP"/dist/torchfits-*.whl)
"$HARNESS_PY" -m pip install --no-deps --force-reinstall "${wheel[0]}"
"$HARNESS_PY" -c 'import torchfits_harness; print("subject:", torchfits_harness.subject())'
