"""The corpus benchmark: case selection, payload maths and a real timing pass.

``bench_corpus.py`` is a script users run, not a library, so the parts that can
silently go wrong are the ones tested here: reading the logical payload out of a
(possibly tile-compressed) header, picking an HDU that matches the declared
family, and emitting rows the shared result schema can rank. The timing pass
runs against a small synthetic file so the test does not need the corpus.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import bench_corpus  # noqa: E402
from benchmarks.bench_contract import RESULT_COLUMNS  # noqa: E402

astropy_fits = pytest.importorskip("astropy.io.fits")


@pytest.fixture()
def image_file(tmp_path: Path) -> Path:
    path = tmp_path / "image.fits"
    data = np.arange(40 * 50, dtype=np.float32).reshape(40, 50)
    astropy_fits.PrimaryHDU(data).writeto(path)
    return path


@pytest.fixture()
def table_file(tmp_path: Path) -> Path:
    path = tmp_path / "table.fits"
    rows = np.array([(1, 2.5), (3, 4.5)], dtype=[("id", "i4"), ("flux", "f4")])
    astropy_fits.HDUList(
        [astropy_fits.PrimaryHDU(), astropy_fits.BinTableHDU(rows, name="CAT")]
    ).writeto(path)
    return path


def _source(ident: str, family: str = "image") -> dict:
    return {"id": ident, "family": family, "families": [family], "kind": "https"}


def test_payload_bytes_reads_plain_header() -> None:
    header = astropy_fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 50
    header["NAXIS2"] = 40
    header["BITPIX"] = -32
    nbytes, shape, itemsize = bench_corpus._payload_bytes(header)
    assert shape == (50, 40)  # FITS axis order, innermost first
    assert itemsize == 4
    assert nbytes == 50 * 40 * 4


def test_payload_bytes_prefers_the_logical_shape() -> None:
    """A tile-compressed HDU stores a small tile in NAXISn, the image in ZNAXISn."""
    header = astropy_fits.Header()
    header["NAXIS"] = 0
    header["BITPIX"] = 8
    header["ZNAXIS"] = 2
    header["ZNAXIS1"] = 100
    header["ZNAXIS2"] = 200
    header["ZBITPIX"] = 16
    nbytes, shape, itemsize = bench_corpus._payload_bytes(header)
    assert shape == (100, 200)
    assert itemsize == 2
    assert nbytes == 100 * 200 * 2


def test_select_cases_reads_the_declared_family(
    monkeypatch: pytest.MonkeyPatch, image_file: Path, table_file: Path
) -> None:
    sources = [_source("img"), _source("tbl", "table")]
    paths = {"img": image_file, "tbl": table_file}
    monkeypatch.setattr(bench_corpus.corpus, "sources", lambda **_: sources)
    monkeypatch.setattr(
        bench_corpus.corpus, "source_files", lambda source: [paths[source["id"]]]
    )
    monkeypatch.setattr(
        bench_corpus.corpus, "families", lambda source: tuple(source["families"])
    )

    cases, skipped = bench_corpus.select_cases(tier="ci", limit=1)
    assert not skipped
    kinds = {(case.family, case.kind) for case in cases}
    assert kinds == {("image", "image"), ("table", "table")}
    by_id = {case.case_id: case for case in cases}
    assert by_id["corpus_img_image_full"].size_mb == pytest.approx(40 * 50 * 4 / 1e6)
    assert by_id["corpus_tbl_table_rows"].rows == 2
    assert by_id["corpus_tbl_table_rows"].size_mb == pytest.approx(2 * 8 / 1e6)


def test_select_cases_uses_a_window_above_the_payload_cap(
    monkeypatch: pytest.MonkeyPatch, image_file: Path
) -> None:
    monkeypatch.setattr(bench_corpus.corpus, "sources", lambda **_: [_source("img")])
    monkeypatch.setattr(
        bench_corpus.corpus, "source_files", lambda source: [image_file]
    )
    monkeypatch.setattr(bench_corpus.corpus, "families", lambda source: ("image",))
    cases, _ = bench_corpus.select_cases(tier="ci", max_mb=0.0001)
    window = next(case for case in cases if case.operation == "read_window")
    x0, y0, x1, y1 = window.window
    # The fixture is 40 rows × 50 columns: NAXIS1=50 is x, NAXIS2=40 is y, so
    # the 256-pixel window clamps to the whole image in each direction.
    assert (x1 - x0, y1 - y0) == (50, 40)
    assert (x0, y0) == (0, 0)
    assert not any(case.operation == "read_full" for case in cases)


def test_cube_above_the_cap_is_header_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "cube.fits"
    astropy_fits.PrimaryHDU(np.zeros((3, 4, 5), dtype=np.float32)).writeto(path)
    monkeypatch.setattr(bench_corpus.corpus, "sources", lambda **_: [_source("cube", "cube")])
    monkeypatch.setattr(bench_corpus.corpus, "source_files", lambda source: [path])
    monkeypatch.setattr(bench_corpus.corpus, "families", lambda source: ("cube",))
    cases, _ = bench_corpus.select_cases(tier="ci", max_mb=1e-6)
    assert [case.operation for case in cases] == ["read_header"]


def test_missing_corpus_file_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bench_corpus.corpus, "sources", lambda **_: [_source("gone")])
    monkeypatch.setattr(
        bench_corpus.corpus, "source_files", lambda source: [tmp_path / "gone.fits"]
    )
    monkeypatch.setattr(bench_corpus.corpus, "families", lambda source: ("image",))
    cases, skipped = bench_corpus.select_cases(tier="ci")
    assert cases == []
    assert skipped == [{"source": "gone", "reason": "not fetched"}]


def test_run_times_every_peer_and_writes_rows(
    monkeypatch: pytest.MonkeyPatch, image_file: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(bench_corpus.corpus, "sources", lambda **_: [_source("img")])
    monkeypatch.setattr(
        bench_corpus.corpus, "source_files", lambda source: [image_file]
    )
    monkeypatch.setattr(bench_corpus.corpus, "families", lambda source: ("image",))

    rows, skipped = bench_corpus.run(
        tier="ci", limit=1, runs=1, warmup=0, run_id="test", output_dir=tmp_path
    )
    assert not skipped
    methods = {row["method"] for row in rows if row["operation"] == "read_full"}
    assert {"torchfits", "astropy"} <= methods
    assert all(row["status"] == "OK" for row in rows)
    full = next(
        row
        for row in rows
        if row["method"] == "torchfits" and row["operation"] == "read_full"
    )
    assert float(full["throughput"]) > 0
    assert full["case_id"] in {row["case_id"] for row in rows}
    assert {row["case_id"] for row in rows if row["best_in_family"]}  # ranked

    out = tmp_path / "results.csv"
    bench_corpus.write_csv(out, rows, RESULT_COLUMNS)
    with out.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == RESULT_COLUMNS
        assert len(list(reader)) == len(rows)


def test_main_list_uses_the_registry(
    monkeypatch: pytest.MonkeyPatch, image_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(bench_corpus.corpus, "sources", lambda **_: [_source("img")])
    monkeypatch.setattr(
        bench_corpus.corpus, "source_files", lambda source: [image_file]
    )
    monkeypatch.setattr(bench_corpus.corpus, "families", lambda source: ("image",))
    assert bench_corpus.main(["--tier", "ci", "--limit", "1", "--list"]) == 0
    printed = capsys.readouterr().out
    assert "corpus_img_image_full" in printed


def test_main_without_corpus_files_is_a_clear_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bench_corpus.corpus, "sources", lambda **_: [_source("gone")])
    monkeypatch.setattr(
        bench_corpus.corpus, "source_files", lambda source: [tmp_path / "gone.fits"]
    )
    monkeypatch.setattr(bench_corpus.corpus, "families", lambda source: ("image",))
    assert bench_corpus.main(["--tier", "ci"]) == 1


@pytest.fixture()
def compressed_file(tmp_path: Path) -> Path:
    """A scaled, tile-compressed image after an empty primary (an archive MEF)."""
    path = tmp_path / "compressed.fits"
    data = (np.arange(64 * 80, dtype=np.int32).reshape(64, 80) % 3000).astype(np.int16)
    comp = astropy_fits.CompImageHDU(data, compression_type="RICE_1", name="SCI")
    comp.header["BSCALE"] = 2.0
    comp.header["BZERO"] = 100.0
    astropy_fits.HDUList([astropy_fits.PrimaryHDU(), comp]).writeto(path)
    return path


@pytest.mark.parametrize("family", ["image", "table"])
@pytest.mark.parametrize("max_mb", [128.0, 1e-4])
def test_tile_compressed_image_is_benchmarked_as_an_image(
    monkeypatch: pytest.MonkeyPatch, compressed_file: Path, family: str, max_mb: float
) -> None:
    """The raw view calls it a BinTableHDU; the case must still be an image."""
    monkeypatch.setattr(bench_corpus.corpus, "sources", lambda **_: [_source("cmp", family)])
    monkeypatch.setattr(bench_corpus.corpus, "source_files", lambda source: [compressed_file])
    monkeypatch.setattr(bench_corpus.corpus, "families", lambda source: (family,))
    cases, skipped = bench_corpus.select_cases(tier="ci", max_mb=max_mb)
    assert not skipped
    assert {case.kind for case in cases} == {"image"}
    assert all(case.compressed and case.scaled and case.hdu == 1 for case in cases)
    data_case = next(case for case in cases if case.operation != "read_header")
    assert data_case.operation == ("read_full" if max_mb > 1 else "read_window")
    if data_case.window is not None:
        x0, y0, x1, y1 = data_case.window
        assert (x1 - x0, y1 - y0) == (80, 64)

    mismatched = bench_corpus.check_agreement(data_case, bench_corpus._methods(data_case))
    assert mismatched == {}
    for method, fn in bench_corpus._methods(data_case).items():
        pixels = bench_corpus._as_array(fn())
        assert pixels.shape == (64, 80), method
        assert float(pixels[0, 1]) == pytest.approx(102.0), method  # 2 * 1 + 100


def test_astropy_rows_are_materialized(table_file: Path) -> None:
    case = bench_corpus.Case(
        case_id="t",
        label="t",
        operation="read_rows",
        family="table",
        path=table_file,
        hdu=1,
        kind="table",
        size_mb=0.0,
        n_points=2,
        rows=2,
    )
    result = bench_corpus._methods(case)["astropy"]()
    assert isinstance(result, dict)
    assert all(type(column) is np.ndarray for column in result.values())
    assert bench_corpus.check_agreement(case, bench_corpus._methods(case)) == {}


def test_a_reader_that_disagrees_is_flagged_not_ranked(
    monkeypatch: pytest.MonkeyPatch, image_file: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(bench_corpus.corpus, "sources", lambda **_: [_source("img")])
    monkeypatch.setattr(bench_corpus.corpus, "source_files", lambda source: [image_file])
    monkeypatch.setattr(bench_corpus.corpus, "families", lambda source: ("image",))
    real_methods = bench_corpus._methods

    def lying_methods(case):
        methods = real_methods(case)
        if case.operation == "read_full":
            methods["astropy"] = lambda: np.zeros((40, 50), dtype=np.float32)
        return methods

    monkeypatch.setattr(bench_corpus, "_methods", lying_methods)
    rows, _ = bench_corpus.run(
        tier="ci", limit=1, runs=1, warmup=0, run_id="test", output_dir=tmp_path
    )
    flagged = next(
        row for row in rows if row["method"] == "astropy" and row["operation"] == "read_full"
    )
    assert flagged["status"] == "MISMATCH"
    assert "mismatch" in flagged["skip_reason"]
    assert flagged["time_s"] is None
