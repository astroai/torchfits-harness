# torchfits-harness

Benchmarks, upstream parity, migration notes, and CANFAR evaluation for
[torchfits](https://github.com/astroai/torchfits).

Production code and the library test suite stay in torchfits. This repo
compares an installed torchfits build with astropy and fitsio. It does not
compile torchfits or vendor CFITSIO.

## Run

Install a torchfits wheel into the environment, then this package:

```bash
pip install torchfits
pip install -e . --no-deps
pixi run bench-cpp
pixi run test
```

`pixi run test` runs the fitsio and astropy upstream smokes. Full benchmark
suites are opt-in (`python benchmarks/bench_all.py`).
