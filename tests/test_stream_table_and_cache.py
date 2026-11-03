"""stream_table header laziness, cache config isolation, and PNG export."""

from __future__ import annotations

import struct
from unittest import mock

import pytest
import torch


def test_read_cpp_table_chunk_defers_header_when_read_fails():
    """A numeric row-slice read that fails before any header consumer runs must
    not touch read_header (header is lazy + unsigned dtypes deferred)."""
    from torchfits._table import _read_scan as scan_mod

    fake_header = mock.Mock(name="read_header")

    def boom(*_a, **_k):
        raise RuntimeError("no reader")

    with mock.patch.object(scan_mod, "_acquire_cpp_reader", side_effect=boom):
        with mock.patch("torchfits.read_header", fake_header, create=True):
            result = scan_mod._read_cpp_table_chunk(
                "/does/not/exist.fits",
                1,
                None,  # columns
                (1, 5),  # row_slice -> num_rows != -1, short-circuits full path
                None,  # rows
                None,  # where
                False,  # mmap
                False,  # decode_bytes
                "utf-8",
                True,  # strip
                False,  # include_fits_metadata
                False,  # apply_fits_nulls
            )
    assert result is None
    fake_header.assert_not_called()


def test_write_rgb_image_produces_valid_png(tmp_path):
    from torchfits.transforms.rgb import write_rgb_image

    rgb = torch.rand(3, 4, 3)  # H=3, W=4, 3 channels
    out = tmp_path / "preview.png"
    write_rgb_image(str(out), rgb)

    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    # IHDR width/height live at bytes 16..24 (big-endian).
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (4, 3)


def test_sigmaclip_clips_outlier():
    from torchfits.transforms.clip import SigmaClip

    x = torch.tensor([[1.0] * 10 + [100.0]])
    clipped = SigmaClip(n_sigma=3.0, max_iter=5, dim=(-1,))(x)
    # The 100.0 outlier is replaced by the mean of the retained ones (~1.0).
    assert torch.allclose(clipped[0, :10], torch.ones(10))
    assert clipped[0, 10].item() < 5.0
    # Deterministic across calls (no leftover buffer state).
    again = SigmaClip(n_sigma=3.0, max_iter=5, dim=(-1,))(x)
    assert torch.equal(clipped, again)


def test_stream_table_skips_header_when_total_rows_given(tmp_path):
    from torchfits._io_engine.table_streaming import stream_table

    f = tmp_path / "empty.fits"
    f.write_bytes(b"\x00")  # only needs to exist

    header_fn = mock.Mock(side_effect=AssertionError("header should not be read"))
    # total_rows=0 returns immediately, after the existence + capability checks.
    chunks = list(stream_table(header_fn, str(f), total_rows=0))
    assert chunks == []
    header_fn.assert_not_called()


def test_stream_table_missing_path_raises():
    from torchfits._io_engine.table_streaming import stream_table

    with pytest.raises(FileNotFoundError):
        list(stream_table(lambda *_: None, "/no/such/torchfits_stream.fits"))


def test_stream_table_accepts_cfitsio_extended_local_path(tmp_path):
    from torchfits._io_engine.table_streaming import stream_table

    f = tmp_path / "t.fits"
    f.write_bytes(b"\x00")
    header_fn = mock.Mock(side_effect=AssertionError("header should not be read"))
    # Extended section must not make exists() fail before CFITSIO opens.
    chunks = list(stream_table(header_fn, f"{f}[1]", total_rows=0))
    assert chunks == []
    header_fn.assert_not_called()


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="torchfits 1.3.1+: a non-numeric tnull reaches the native schema check "
    "and raises RuntimeError without naming TNULL (reported upstream)",
)
def test_insert_column_rejects_bad_tnull(tmp_path):
    """A malformed ``tnull`` is caller input: ValueError naming TNULL."""
    import numpy as np
    import torch

    import torchfits

    path = str(tmp_path / "t.fits")
    torchfits.write(path, {"A": torch.arange(3, dtype=torch.int16)}, overwrite=True)
    with pytest.raises(ValueError, match="(?i)tnull"):
        torchfits.table.insert_column(
            path, "QUAL", np.zeros(3, dtype=np.int16), format="I", tnull="not-an-int"
        )


@pytest.mark.parametrize("reader", ["read", "read_torch"])
def test_table_read_of_a_missing_file_names_the_path(reader):
    import torchfits.table as tf_table

    with pytest.raises((OSError, RuntimeError), match="/does/not/exist.fits"):
        getattr(tf_table, reader)("/does/not/exist.fits", hdu=1)


def _write_scaled_table(tmp_path):
    """J column with TSCAL/TZERO -> physical values cannot come from raw mmap."""
    import numpy as np
    from astropy.io import fits

    path = str(tmp_path / "scaled.fits")
    raw = np.array([10, 20, 30, 40], dtype=np.int32)
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="PHYS", format="J", array=raw)]
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=True)
    with fits.open(path, mode="update") as hdul:
        hdul[1].header["TSCAL1"] = 0.5
        hdul[1].header["TZERO1"] = 100.0
    return path, raw.astype(np.float64) * 0.5 + 100.0


def test_stream_table_scaled_probe_failure_routes_to_buffered(tmp_path, monkeypatch):
    """A schema-probe failure must fall back to the buffered reader (r4c-07).

    Falling through to the raw mmap path would either raise mid-stream or hand
    back unscaled raw integers for a TSCAL/TZERO column.
    """
    import numpy as np

    import torchfits
    from torchfits._io_engine.table_streaming import stream_table

    path, expected = _write_scaled_table(tmp_path)

    def bad_schema(*_a, **_k):
        raise ValueError("hostile header")

    with monkeypatch.context() as m:
        m.setattr(torchfits, "read_header", bad_schema)
        chunks = list(
            stream_table(
                lambda *_: None,
                path,
                hdu=1,
                columns=["PHYS"],
                mmap=True,
                total_rows=4,
            )
        )
    got = np.concatenate([np.asarray(c["PHYS"]).reshape(-1) for c in chunks])
    np.testing.assert_allclose(got, expected)


def test_stream_table_probe_io_error_propagates(tmp_path, monkeypatch):
    """IO errors during the route probes must re-raise, not fall through (r4c-07)."""
    import numpy as np
    import pytest
    from astropy.io import fits

    import torchfits
    from torchfits._io_engine.table_streaming import stream_table

    path = str(tmp_path / "plain.fits")
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="ID", format="J", array=np.arange(4, dtype=np.int32))]
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=True)

    def io_bomb(*_a, **_k):
        raise OSError("storage gone")

    with monkeypatch.context() as m:
        m.setattr(torchfits, "read_header", io_bomb)
        with pytest.raises(OSError, match="storage gone"):
            list(stream_table(lambda *_: None, path, hdu=1, mmap=True, total_rows=4))
