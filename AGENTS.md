# torchfits-harness

This repository owns benchmarks, exhaustive upstream parity, migration notes,
scorecards, and CANFAR evaluation for torchfits.

- Torchfits owns the library, wheels, and tests that do not import astropy or fitsio.
- Install an explicit torchfits wheel as the subject. Do not path-depend on a
  torchfits checkout and do not vendor CFITSIO here.
- Astropy and fitsio are comparators.
- Do not auto-run full benchmark suites.

Pixi: `pixi run test` for the upstream smokes. `pixi run bench-cpp` for the
bounded C++ backend smoke.
