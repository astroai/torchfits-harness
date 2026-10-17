# torchfits-harness

Benchmarks, upstream parity, migration notes, and CANFAR evaluation for
[torchfits](https://github.com/astroai/torchfits).

The library, wheels, and tests that do not import astropy or fitsio stay in
torchfits. This repo compares an installed torchfits wheel with astropy and
fitsio. It does not compile torchfits or vendor CFITSIO.

## Run

```bash
pixi install
pixi run subject /path/to/torchfits-*.whl
pixi run dev
pixi run test          # fitsio and astropy upstream smokes
pixi run bench-cpp     # bounded C++ backend comparison
```

`pixi run bench-fits`, `pixi run bench-fitstable`, and `pixi run bench-all`
run the larger suites in this same environment. Full suites are opt-in.

## Docs

| | |
|---|---|
| [Benchmarks](docs/benchmarks.md) | Methodology and published results |
| [Parity](docs/parity.md) | Behavior compared with astropy and fitsio |
| [From astropy](docs/migration_astropy.md) | Migration notes |
| [From fitsio](docs/migration_fitsio.md) | Migration notes |
