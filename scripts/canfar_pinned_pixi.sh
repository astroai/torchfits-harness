#!/usr/bin/env bash
# Put a pixi that satisfies an exact `requires-pixi = "==X.Y.Z"` pin on PATH.
#
# pixi.toml carries no such pin today, so use_pinned_pixi returns at once and
# the image's own pixi is used. If an exact pin is ever added: astroai/base
# ships whatever pixi was current when the image was built, so `pixi install`
# would abort with "this project requires pixi '==X', but you have pixi Y"; the
# image's pixi lives in a read-only prefix, so the pinned release is downloaded
# into $SCRATCH and prepended to PATH instead of calling `pixi self-update`.
#
# Sourced by canfar_verify_incontainer.sh and canfar_gpu_bench_incontainer.sh
# from the repository root; the caller sets SCRATCH.

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
  if ! curl -fsSL --retry 3 -o "${dir}/pixi" \
      "https://github.com/prefix-dev/pixi/releases/download/v${want}/pixi-${arch}-unknown-linux-musl"; then
    echo "FAIL: could not download pixi ${want} from GitHub releases" >&2
    return 1
  fi
  chmod +x "${dir}/pixi" || return 1
  export PATH="${dir}:${PATH}"
  have="$(pixi --version 2>/dev/null | awk '{print $2}')"
  if [[ "${have}" != "${want}" ]]; then
    echo "FAIL: pinned pixi ${want} is not on PATH (got ${have:-missing})" >&2
    return 1
  fi
  echo "pixi ${want} ready at ${dir}"
}
