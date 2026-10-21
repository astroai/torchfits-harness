"""Differential parity over astropy's own FITS regression corpus.

torchfits is otherwise tested on frames it wrote itself, or on clean
observatory products. astropy ships 30-odd hand-built files chosen for exactly
the cases a writer never produces: a zero-width table column, a ``TDIM`` that
does not match the data size, a heap with a gap, scaled unsigned images, ASCII
tables, random groups, checksum cards, and a ``NAXIS1`` of 777 million with no
data behind it.

Each file is opened by both libraries and the observable answers compared: HDU
type and count, image shape and payload, table row/column layout, and the
header keyword sequence. The corpus decides what is testable -- a file astropy
itself refuses is skipped with astropy's own error, and the two random-group
files are skipped on torchfits' documented ``GROUPS=T`` refusal -- so the list
cannot go stale against an astropy upgrade.

Blank padding cards are excluded from the keyword comparison: astropy keeps the
cards that pad a header to its 2880-byte block, torchfits reports only the cards
that carry a keyword.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torchfits  # noqa: E402
import torchfits.table  # noqa: E402
from astropy.io import fits  # noqa: E402

DATA = Path(fits.__file__).resolve().parent / "tests" / "data"
CORPUS = sorted(DATA.glob("*.fits"))

pytestmark = pytest.mark.skipif(
    not CORPUS, reason="astropy FITS regression corpus not installed"
)

#: astropy's class for each on-disk HDU, mapped to ``torchfits.read_hdu_type``.
HDUTYPE = {
    "PrimaryHDU": "IMAGE",
    "ImageHDU": "IMAGE",
    "CompImageHDU": "IMAGE",
    "GroupsHDU": "IMAGE",
    "BinTableHDU": "BINARY_TABLE",
    "TableHDU": "ASCII_TABLE",
}

#: Known product gaps this corpus exposes, per file and area ("open" or
#: "table"). Each is a strict xfail, so it turns red the moment the reader
#: starts handling the case -- the prompt to drop the marker.
KNOWN_GAPS = {
    ("ascii_i4-i20.fits", "table"): (
        "table.read refuses an ASCII integer column wider than I11: "
        "'Failed to read column col3: datatype conversion overflow'"
    ),
    ("zerowidth.fits", "table"): (
        "a legal zero-width column (TFORM 0E on ORBPARM) is refused: "
        "'TFORM repeat count must be positive for column ORBPARM (got 0)'"
    ),
    ("verify.fits", "open"): (
        "torchfits refuses a file astropy opens: 'Could not open FITS file'"
    ),
}


def _xfail_gap(path: Path, area: str) -> None:
    reason = KNOWN_GAPS.get((path.name, area))
    if reason is not None:
        pytest.xfail(reason)


def _rows(hdu: fits.hdu.base.ExtensionHDU) -> int:
    data = getattr(hdu, "data", None)
    return int(np.shape(data)[0]) if data is not None else 0


def _open(path: Path, *, raw: bool = False) -> fits.HDUList:
    try:
        return fits.open(path, disable_image_compression=raw)
    except Exception as exc:  # noqa: BLE001 - astropy decides what it can open
        pytest.skip(f"astropy cannot open it: {type(exc).__name__}: {exc}")


def _torchfits_or_skip(call, *args, **kwargs):
    try:
        return call(*args, **kwargs)
    except RuntimeError as exc:
        if "Random Groups" in str(exc):
            pytest.skip(str(exc))
        raise


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.name)
def test_hdu_types_and_image_shapes_match_astropy(path: Path) -> None:
    """Every HDU's kind and, for images, its shape must agree."""
    _xfail_gap(path, "open")
    with _open(path) as hdul:
        expected = [
            (HDUTYPE.get(type(hdu).__name__, type(hdu).__name__), np.shape(getattr(hdu, "data", None)))
            for hdu in hdul
        ]

    count = torchfits.read_num_hdus(str(path))
    assert count == len(expected), (
        f"{path.name}: {count} HDUs, astropy sees {len(expected)}: "
        f"{[name for name, _ in expected]}"
    )

    for index, (kind, shape) in enumerate(expected):
        got_kind = torchfits.read_hdu_type(str(path), index)
        assert got_kind == kind, (
            f"{path.name} HDU {index}: torchfits says {got_kind}, astropy {kind}"
        )
        if kind != "IMAGE":
            continue
        got_shape = _torchfits_or_skip(torchfits.read_shape, str(path), index)[1]
        assert tuple(got_shape) == tuple(shape), (
            f"{path.name} HDU {index}: shape {tuple(got_shape)} != astropy {tuple(shape)}"
        )


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.name)
def test_table_row_and_column_counts_match_astropy(path: Path) -> None:
    """Row count and column count for every table HDU, ASCII tables included."""
    with _open(path) as hdul:
        tables = [
            (index, len(hdu.columns), _rows(hdu))
            for index, hdu in enumerate(hdul)
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU))
        ]

    _xfail_gap(path, "table")
    for index, ncols, nrows in tables:
        table = torchfits.table.read(str(path), hdu=index)
        assert table.num_rows == nrows, (
            f"{path.name} HDU {index}: {table.num_rows} rows != astropy {nrows}"
        )
        assert len(table.column_names) == ncols, (
            f"{path.name} HDU {index}: {table.column_names} != astropy's {ncols} columns"
        )


@pytest.mark.parametrize(
    "path",
    [p for p in CORPUS if p.name not in {"zerowidth.fits"}],
    ids=lambda p: p.name,
)
def test_image_payload_matches_astropy(path: Path) -> None:
    """Every image array must match elementwise, scaled unsigned ones included."""
    _xfail_gap(path, "open")
    with _open(path) as hdul:
        images = [
            (index, np.asarray(hdu.data))
            for index, hdu in enumerate(hdul)
            if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU))
            and getattr(hdu, "data", None) is not None
            and np.asarray(hdu.data).size > 0
        ]

    for index, expected in images:
        got = np.asarray(
            _torchfits_or_skip(torchfits.read, str(path), index)
        ).reshape(expected.shape)
        if got.dtype.kind == "f" or expected.dtype.kind == "f":
            same = np.allclose(got.astype(np.float64), expected.astype(np.float64), equal_nan=True)
        else:
            same = np.array_equal(got.astype(np.int64), expected.astype(np.int64))
        assert same, (
            f"{path.name} HDU {index}: payload differs from astropy "
            f"(dtype {got.dtype} vs {expected.dtype}, "
            f"{int((got.astype(np.float64) != expected.astype(np.float64)).sum())} "
            f"of {expected.size} elements)"
        )


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.name)
def test_header_keyword_order_matches_astropy(path: Path) -> None:
    """The keyword sequence, in file order, with astropy's padding removed.

    astropy is opened with ``disable_image_compression=True`` on purpose:
    torchfits' ``Header`` is the on-disk one, so a tile-compressed HDU shows its
    ``ZIMAGE`` table cards here and its decompressed pixels through ``read``.
    astropy's default view rewrites that header into the logical image's, which
    is a different answer to a different question.
    """
    _xfail_gap(path, "open")
    with _open(path, raw=True) as hdul:
        expected = [
            [card.keyword.strip() for card in hdu.header.cards if card.keyword.strip()]
            for hdu in hdul
        ]

    with torchfits.open(str(path)) as hdul:
        for index, want in enumerate(expected):
            got = [card.key.strip() for card in hdul[index].header.cards if card.key.strip()]
            assert got == want, (
                f"{path.name} HDU {index}: card keywords differ from astropy\n"
                f"  first difference at "
                f"{next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))}\n"
                f"  astropy:   {want[:12]}...\n"
                f"  torchfits: {got[:12]}..."
            )
