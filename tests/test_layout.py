"""The evaluation tree that moved out of torchfits is present."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_benchmarks_and_parity_docs_live_here() -> None:
    for rel in (
        "benchmarks/bench_cpp_backend.py",
        "benchmarks/bench_all.py",
        "docs/benchmarks.md",
        "docs/parity.md",
        "docs/migration_astropy.md",
        "docs/migration_fitsio.md",
        "scripts/patch_bench_docs.py",
        ".github/workflows/bench-report.yml",
    ):
        assert (ROOT / rel).is_file(), rel
