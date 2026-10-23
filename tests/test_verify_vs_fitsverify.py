"""Cross-check ``torchfits verify`` against NASA HEASARC's ``fitsverify``.

``torchfits verify`` is checksum-scoped by design: it reads DATASUM/CHECKSUM
through the native reader and never shells out.  ``fitsverify`` also checks
FITS structure, so the two are not interchangeable; this test asserts that
they **agree on pass/fail** for fixtures that differ only in checksum integrity
or truncation, and documents the scope difference.

``fitsverify`` is not a Python package.  Install it from conda-forge
(``conda install -c conda-forge fitsverify``) or from HEASoft; the test skips
when the binary is absent, so CI without FTOOLS still runs.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import torchfits

FITSVERIFY = shutil.which("fitsverify")
pytestmark = pytest.mark.skipif(
    FITSVERIFY is None,
    reason="fitsverify (HEASoft/FTOOLS) is not installed",
)


def _torchfits_verify(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "torchfits.cli", "verify", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )


def _fitsverify(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [FITSVERIFY, "-q", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )


def _write_clean(path: Path, *, checksum: bool) -> None:
    torchfits.write(
        str(path),
        torch.arange(16, dtype=torch.float32).reshape(4, 4),
        checksum=checksum,
        overwrite=True,
    )


def _write_bad_checksum(path: Path) -> None:
    _write_clean(path, checksum=True)
    payload = bytearray(path.read_bytes())
    payload[2880] ^= 0xFF  # first data byte; header CHECKSUM/DATASUM stay
    path.write_bytes(payload)


def _write_truncated(path: Path) -> None:
    _write_clean(path, checksum=True)
    path.write_bytes(path.read_bytes()[:3000])


@pytest.mark.parametrize(
    ("name", "writer", "expected_pass"),
    [
        ("clean", lambda path: _write_clean(path, checksum=True), True),
        ("no_checksum", lambda path: _write_clean(path, checksum=False), True),
        ("checksum_mismatch", _write_bad_checksum, False),
        ("truncated", _write_truncated, False),
    ],
)
def test_verify_verdict_matches_fitsverify(tmp_path, name, writer, expected_pass):
    path = tmp_path / f"{name}.fits"
    writer(path)

    ours = _torchfits_verify(path)
    theirs = _fitsverify(path)

    assert (ours.returncode == 0) is expected_pass, (
        f"torchfits verify exit {ours.returncode} for {name}\n"
        f"stdout:\n{ours.stdout}\nstderr:\n{ours.stderr}"
    )
    assert (theirs.returncode == 0) is expected_pass, (
        f"fitsverify exit {theirs.returncode} for {name}\n"
        f"stdout:\n{theirs.stdout}\nstderr:\n{theirs.stderr}"
    )


def test_verify_is_self_contained(tmp_path):
    """The library must not invoke fitsverify; only this harness does."""
    path = tmp_path / "clean.fits"
    _write_clean(path, checksum=True)
    result = _torchfits_verify(path)
    assert result.returncode == 0
    assert "fitsverify" not in result.stdout
    assert "fitsverify" not in result.stderr
