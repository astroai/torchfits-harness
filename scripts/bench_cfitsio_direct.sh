#!/usr/bin/env bash
# Build vendored-CFITSIO pure-C microbench and compare to torchfits/fitsio peers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# The pure-C microbench compiles vendored CFITSIO from a torchfits checkout.
# This repo does not vendor or build CFITSIO.
if [[ ! -f extern/cfitsio/CMakeLists.txt ]]; then
  echo "bench_cfitsio_direct needs extern/cfitsio from a torchfits checkout; this harness does not vendor CFITSIO." >&2
  exit 1
fi

pixi run python benchmarks/run_cfitsio_direct_bench.py "$@"
