"""Example scripts must check what they claim and stay out of the docs tree."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_cfitsio_ascii_failure_is_not_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import examples.example_cfitsio_cookbook as cook

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("ascii broke")

    monkeypatch.setattr(cook.tf_table, "read", boom)
    with pytest.raises(RuntimeError, match="ascii broke"):
        cook.test_cfitsio_ascii_table(str(tmp_path))
    assert "ASCII: OK" not in capsys.readouterr().out


def test_cfitsio_checksum_example_rejects_a_failed_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import examples.example_cfitsio_cookbook as cook

    monkeypatch.setattr(
        cook.torchfits,
        "verify_checksums",
        lambda _path, hdu=0: {
            "datastatus": -1,
            "hdustatus": -1,
            "ok": False,
            "present": True,
            "status": "fail",
        },
    )
    with pytest.raises(AssertionError):
        cook.test_cfitsio_checksum(str(tmp_path))


def test_cfitsio_multi_hdu_copy_uses_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    import examples.example_cfitsio_cookbook as cook

    calls: list[list[str]] = []
    real_run = subprocess.run

    def spy(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append([str(part) for part in cmd])
        return real_run(cmd, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(subprocess, "run", spy)
    cook.test_cfitsio_multi_hdu_copy(str(tmp_path))
    assert any(
        "torchfits.cli" in part and "copy" in parts for parts in calls for part in parts
    )
