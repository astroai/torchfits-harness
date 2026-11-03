import pytest
import torch
import time
import os

import torchfits


def test_opened_hdul_repr_reports_shape_and_dtype(tmp_path):
    """``repr`` reads the summary from parsed header values, which 1.3.0 keeps
    typed (NAXIS is an int; see ``HDUList.fromfile``). It must not depend on
    the raw FITS string form and must not choke on either."""
    path = tmp_path / "repr.fits"
    torchfits.write(str(path), torch.ones((2, 3), dtype=torch.float32))

    with torchfits.open(str(path)) as hdul:
        assert hdul[0].header["NAXIS"] == 2
        # The raw card is still a string, so the summary has to cope with a
        # string-typed numeric card as well (r4b: repr used to format it raw).
        hdul[0].header["NAXIS"] = "2"
        summary = repr(hdul)

    assert "(2, 3)" in summary
    assert "float32" in summary


def test_hdu_file_ops_insert_replace_delete(tmp_path):
    path = tmp_path / "ops.fits"
    base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    torchfits.write(str(path), base, overwrite=True)

    payload1 = torch.zeros((4, 4), dtype=torch.float32)
    torchfits.insert_hdu(str(path), payload1, index=1, header={"EXTNAME": "SCI"})

    with torchfits.open(str(path)) as hdul:
        assert len(hdul) == 2
        assert hdul["SCI"].header.get("EXTNAME") == "SCI"
    assert torch.equal(torchfits.read(str(path), hdu=1), payload1)

    payload2 = torch.ones((4, 4), dtype=torch.float32)
    torchfits.replace_hdu(str(path), "SCI", payload2)
    assert torch.equal(torchfits.read(str(path), hdu=1), payload2)

    torchfits.delete_hdu(str(path), "SCI")
    with torchfits.open(str(path)) as hdul:
        assert len(hdul) == 1
    with pytest.raises(Exception):
        _ = torchfits.read(str(path), hdu=1)


@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Flaky cache invalidation on CI"
)
def test_external_overwrite_invalidates_cached_handle(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    path = tmp_path / "external_overwrite.fits"

    original = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    updated = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 100.0

    torchfits.write(str(path), original, overwrite=True)
    first = torchfits.read(str(path), hdu=0)
    assert torch.equal(first, original)

    # SharedReadMeta re-stats at most once per
    # TORCHFITS_SHARED_META_VALIDATE_INTERVAL_MS (default 1000 ms; see
    # kSharedMetaValidateIntervalNs). Sleep past that throttle so the next
    # read is allowed to notice the external overwrite. The 0.01 s loop
    # below only retries until the bytes match; it is not an ordering sleep.
    time.sleep(1.1)
    fits.PrimaryHDU(updated.numpy()).writeto(path, overwrite=True)
    second = torchfits.read(str(path), hdu=0)
    deadline = time.monotonic() + 3.0
    while not torch.equal(second, updated) and time.monotonic() < deadline:
        time.sleep(0.01)
        second = torchfits.read(str(path), hdu=0)
    assert torch.equal(second, updated)


def test_writing_another_file_keeps_open_read_handle_valid(tmp_path):
    source = tmp_path / "source.fits"
    torchfits.write(
        str(source),
        torch.arange(12, dtype=torch.float32).reshape(3, 4),
        overwrite=True,
    )

    first, first_header = torchfits.read(str(source), return_header=True)
    for index in range(2):
        torchfits.write(
            str(tmp_path / f"mask_{index}.fits"),
            torch.zeros_like(first),
            overwrite=True,
        )

    reread, reread_header = torchfits.read(str(source), return_header=True)
    assert torch.equal(reread, first)
    assert reread_header["NAXIS1"] == first_header["NAXIS1"]


def test_tablehduref_file_mutators_roundtrip(tmp_path):
    path = tmp_path / "table_mutators.fits"
    torchfits.write(
        str(path),
        {
            "ID": torch.tensor([1, 2, 3], dtype=torch.int32),
            "VAL": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            "FLAG": torch.tensor([True, False, True], dtype=torch.bool),
        },
        overwrite=True,
    )

    with torchfits.open(str(path)) as hdul:
        table = hdul[1]
        table = table.append_rows_file({"ID": [4], "VAL": [0.4], "FLAG": [False]})
        table = table.insert_rows_file(
            {"ID": [99], "VAL": [9.9], "FLAG": [True]}, row=1
        )
        table = table.update_rows_file({"VAL": [8.8]}, row_slice=slice(0, 1))
        table = table.rename_columns_file({"VAL": "FLUX"})
        table = table.drop_columns_file(["FLAG"])

    with torchfits.open(str(path)) as hdul:
        table = hdul[1]
        assert table["ID"].tolist() == [1, 99, 2, 3, 4]
        assert table["FLUX"].shape[0] == 5
        assert abs(float(table["FLUX"][0]) - 8.8) < 1e-6
        assert "FLAG" not in table.columns


def test_tablehduref_mutation_refreshes_schema_and_rowcount(tmp_path):
    path = tmp_path / "table_refresh.fits"
    torchfits.write(
        str(path),
        {
            "ID": torch.tensor([1, 2], dtype=torch.int32),
            "VAL": torch.tensor([10.0, 20.0], dtype=torch.float32),
        },
        overwrite=True,
    )

    with torchfits.open(str(path)) as hdul:
        table = hdul[1]
        updated = table.append_rows_file({"ID": [3], "VAL": [30.0]})
        assert updated.num_rows == 3
        renamed = updated.rename_columns_file({"ID": "OBJID"})
        assert "OBJID" in renamed.columns
        assert "ID" not in renamed.columns


def test_tablehduref_column_mutators_roundtrip(tmp_path):
    path = tmp_path / "table_column_mutators.fits"
    torchfits.table.write(
        str(path),
        data={
            "ID": torch.tensor([1, 2], dtype=torch.int32),
            "QUAL": torch.tensor([10, 20], dtype=torch.int16),
        },
        schema={
            "ID": {"format": "J"},
            "QUAL": {"format": "I", "unit": "adu", "tnull": -999},
        },
        overwrite=True,
    )

    with torchfits.open(str(path)) as hdul:
        table = hdul[1]
        table = table.insert_column_file(
            "FLAGS",
            [7, 8],
            index=1,
            format="I",
            unit="flag",
            tnull=-1,
        )
        table = table.replace_column_file("QUAL", [101, 102])
        assert table.columns == ["ID", "FLAGS", "QUAL"]

    with torchfits.open(str(path)) as hdul:
        table = hdul[1]
        assert table["FLAGS"].tolist() == [7, 8]
        assert table["QUAL"].tolist() == [101, 102]
        assert table.header.get("TUNIT2") == "flag"
        assert int(table.header.get("TNULL2")) == -1
        assert table.header.get("TUNIT3") == "adu"
        assert int(table.header.get("TNULL3")) == -999


def test_replace_column_file_mirrors_tform(tmp_path) -> None:
    path = tmp_path / "t.fits"
    torchfits.write(str(path), {"X": [1.0, 2.0, 3.0]})
    ref = torchfits.open(str(path))[1]
    ref2 = ref.replace_column_file("X", [10, 20, 30], format="J")

    file_header = torchfits.read_header(str(path), hdu=1)
    assert str(file_header.get("TFORM1", "")).strip() == "J"
    assert str(ref2.header.get("TFORM1", "")).strip() == "J"
    schema_cols = ref2.schema["columns"]
    assert schema_cols[0]["tform"] == "J"
    values = ref2.read()["X"]
    assert values.tolist() == [10, 20, 30]
    assert values.dtype == torch.int32


def test_hdulist_write_preserves_table_extension_metadata(tmp_path):
    src = tmp_path / "table_src.fits"
    dst = tmp_path / "table_dst.fits"

    torchfits.table.write(
        str(src),
        data={
            "ID": torch.tensor([1, 2], dtype=torch.int32),
            "QUAL": torch.tensor([10, 20], dtype=torch.int16),
        },
        schema={
            "ID": {"format": "J"},
            "QUAL": {"format": "I", "unit": "adu", "tnull": -999},
        },
        overwrite=True,
    )

    with torchfits.open(str(src)) as hdul:
        hdul.write(str(dst), overwrite=True)

    with torchfits.open(str(dst)) as hdul:
        assert len(hdul) == 2
        table = hdul[1]
        assert table.columns == ["ID", "QUAL"]
        assert table["QUAL"].tolist() == [10, 20]
        assert str(table.header.get("TFORM2", "")).upper().startswith("I")
        assert table.header.get("TUNIT2") == "adu"
        assert int(table.header.get("TNULL2")) == -999


def test_insert_hdu_compressed_image_only(tmp_path):
    path = tmp_path / "insert_compressed.fits"
    base = torch.arange(16, dtype=torch.int16).reshape(4, 4)
    torchfits.write(str(path), base, overwrite=True, compress=True)

    inserted = torch.full((4, 4), 7, dtype=torch.int16)
    torchfits.insert_hdu(
        str(path),
        inserted,
        index=1,
        header={"EXTNAME": "SCI"},
        compress=True,
    )

    with torchfits.open(str(path)) as hdul:
        assert len(hdul) == 3
        assert hdul[1].header.get("EXTNAME") == "SCI"
    assert torch.equal(torchfits.read(str(path), hdu=1), inserted)
    assert torch.equal(torchfits.read(str(path), hdu=2), base)


def test_replace_hdu_compressed_image_only(tmp_path):
    path = tmp_path / "replace_compressed.fits"
    base = torch.arange(16, dtype=torch.int16).reshape(4, 4)
    torchfits.write(str(path), base, overwrite=True, compress=True)

    replacement = torch.full((4, 4), 3, dtype=torch.int16)
    torchfits.replace_hdu(
        str(path),
        1,
        replacement,
        header={"EXTNAME": "SCI_REPLACED"},
        compress="RICE_1",
    )

    with torchfits.open(str(path)) as hdul:
        assert len(hdul) == 2
        assert hdul[1].header.get("EXTNAME") == "SCI_REPLACED"
    assert torch.equal(torchfits.read(str(path), hdu=1), replacement)


def _image_and_table(path):
    image = torch.arange(16, dtype=torch.int16).reshape(4, 4)
    table = {"ID": torch.tensor([1, 2, 3], dtype=torch.int32)}
    torchfits.write(
        str(path),
        torchfits.HDUList([torchfits.TensorHDU(image), torchfits.TableHDU(table)]),
        overwrite=True,
    )
    return image


# HDU edits never renumber the untouched HDUs and compress= applies to the new
# HDU only (torchfits docs/api-core-io.md, "HDU Mutation").


def test_insert_hdu_compressed_mixed_hdus_roundtrip(tmp_path):
    path = tmp_path / "insert_compressed_mixed.fits"
    image = _image_and_table(path)

    inserted = torch.full((4, 4), 11, dtype=torch.int16)
    torchfits.insert_hdu(
        str(path),
        inserted,
        index=1,
        header={"EXTNAME": "SCI_INSERT"},
        compress=True,
    )

    with torchfits.open(str(path)) as hdul:
        assert len(hdul) == 3
        assert hdul[1].header.get("EXTNAME") == "SCI_INSERT"
    assert torchfits.read_header(str(path), 1).get("ZIMAGE") is True
    assert torch.equal(torchfits.read(str(path), hdu=0), image)
    assert torch.equal(torchfits.read(str(path), hdu=1), inserted)
    assert torchfits.read(str(path), hdu=2)["ID"].tolist() == [1, 2, 3]


def test_delete_hdu_compressed_image_only(tmp_path):
    path = tmp_path / "delete_compressed.fits"
    img0 = torch.arange(16, dtype=torch.int16).reshape(4, 4)
    img1 = torch.full((4, 4), 9, dtype=torch.int16)
    torchfits.write(
        str(path),
        torchfits.HDUList([torchfits.TensorHDU(img0), torchfits.TensorHDU(img1)]),
        overwrite=True,
        compress=True,
    )
    before = path.read_bytes()

    # A delete writes no HDU, so compress= is refused rather than ignored.
    with pytest.raises(ValueError, match="compress"):
        torchfits.delete_hdu(str(path), 1, compress=True)
    assert path.read_bytes() == before

    torchfits.delete_hdu(str(path), 1)
    with torchfits.open(str(path)) as hdul:
        assert len(hdul) == 2
    assert torch.equal(torchfits.read(str(path), hdu=1), img1)
    assert torchfits.read_header(str(path), 1).get("ZIMAGE") is True


def test_delete_hdu_compressed_mixed_hdus_roundtrip(tmp_path):
    path = tmp_path / "delete_compressed_mixed.fits"
    _image_and_table(path)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="compress"):
        torchfits.delete_hdu(str(path), 0, compress=True)
    # Without compress= it is still refused: the table cannot become primary.
    with pytest.raises(ValueError, match="cannot become the primary"):
        torchfits.delete_hdu(str(path), 0)
    assert path.read_bytes() == before

    torchfits.delete_hdu(str(path), 1)
    with torchfits.open(str(path)) as hdul:
        assert len(hdul) == 1


def test_replace_hdu_compressed_mixed_hdus_roundtrip(tmp_path):
    path = tmp_path / "replace_compressed_mixed.fits"
    image = _image_and_table(path)

    replacement = {"ID": torch.tensor([10, 20, 30], dtype=torch.int32)}
    torchfits.replace_hdu(
        str(path),
        1,
        replacement,
        header={"EXTNAME": "CAT_REPLACED"},
        compress=True,
    )

    with torchfits.open(str(path)) as hdul:
        assert len(hdul) == 2
        assert hdul[1].header.get("EXTNAME") == "CAT_REPLACED"
    assert torch.equal(torchfits.read(str(path), hdu=0), image)
    assert torchfits.read(str(path), hdu=1)["ID"].tolist() == [10, 20, 30]


@pytest.mark.parametrize(
    "edit",
    [
        lambda p: torchfits.insert_hdu(p, {"ID": torch.arange(3)}, index=0),
        lambda p: torchfits.replace_hdu(p, 0, {"ID": torch.arange(3)}),
        lambda p: torchfits.insert_hdu(p, torch.ones(3, 3), index=0, compress=True),
    ],
    ids=["insert-table-primary", "replace-primary-with-table", "compress-primary"],
)
def test_failed_hdu_mutation_preserves_original(tmp_path, edit):
    """An edit that cannot land raises ValueError and leaves the file as it was."""
    path = tmp_path / "mutation_preserved.fits"
    original = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    torchfits.write(str(path), original)
    before = path.read_bytes()

    with pytest.raises(ValueError):
        edit(str(path))

    assert path.read_bytes() == before
    assert torch.equal(torchfits.read(str(path)), original)
