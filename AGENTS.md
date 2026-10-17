# torchfits-harness

This repository owns benchmarks, upstream parity, migration notes, scorecards,
and CANFAR evaluation for torchfits.

Torchfits owns the library, wheels, and tests that do not import astropy or
fitsio. Astropy and fitsio are comparators here, not torchfits dependencies.

## Subject

Install an explicit torchfits wheel, then this package:

```bash
pixi run subject /path/to/torchfits-*.whl
pixi run dev
```

`pixi run dev` is `pip install -e . --no-deps`. Do not path-depend on a
torchfits checkout, do not put a torchfits checkout on `PYTHONPATH`, and do
not vendor CFITSIO. Record the subject with `python -c "import torchfits_harness; print(torchfits_harness.subject())"`.

Pixi sets `PYTHONNOUSERSITE=1`. Do not install into `~/.local`.

## Verify

| When | Command |
|---|---|
| During edits | `pixi run test` |
| Bounded comparison | `pixi run bench-cpp` |
| Larger suites | `pixi run bench-fits`, `pixi run bench-all` |

Do not auto-run full benchmark suites or launch CANFAR jobs.

Tests that read `src/torchfits/cpp_src`, `extern/cfitsio`, or `tests/cpp`
belong in torchfits. The pure-C CFITSIO microbench
(`pixi run bench-cfitsio-direct`) needs that vendored tree and exits here
without it.

CANFAR launchers under `scripts/launch_canfar_*.sh` and
`scripts/canfar_*_incontainer.sh` still contain steps that rebuild a
torchfits extension (`bench-install`, `bench-gpu-install`). Those steps are
the torchfits repo's job. Do not point them at this checkout.
