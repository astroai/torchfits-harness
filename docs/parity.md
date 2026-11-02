# Feature Parity with Astropy & fitsio

A comprehensive comparison of astronomical FITS standard features supported by `torchfits` compared to `astropy.io.fits` and `fitsio`.

For environment, Python, and PyTorch ABI compatibility, see torchfits' [Environment Compatibility](https://github.com/astroai/torchfits/blob/main/docs/compatibility.md).

---

## Capabilities Overview

Status values:

- **Supported**: Fully implemented and verified against standard FITS test suites.
- **Partial**: Supported for common workflows with documented boundary behavior.
- **Unsupported**: A recognized FITS feature that torchfits does not currently expose.
- **Out of Scope**: High-level astronomy models (coordinates, cosmology, units) that belong in dedicated domain packages.

Every major format family has an explicit status below. The upstream parity
smokes and behaviour replays referenced by this matrix are inventoried in
`benchmarks/replays/upstream_sources.json` and checked by
`tests/test_upstream_parity_inventory.py`.

<!-- major-format-coverage:start -->
| Major FITS surface | Status | Contract |
|---|:---:|---|
| Primary HDU and 2D image arrays | **Supported** | Read/write with standard numeric BITPIX values. |
| 3D/4D image cubes and MEF | **Supported** | Tensor dimensions, extension selection, and metadata are preserved. |
| Binary tables (`BINTABLE`) | **Supported** | Projection, row windows, predicates, Arrow, and tensor reads. |
| ASCII tables (`TABLE`) | **Supported** | Fixed-width parsing through the table readers. |
| Variable-length arrays (`P`/`Q`) | **Partial** | Buffered reads and Arrow conversion are supported; memory-mapped updates are not. |
| Tile-compressed images | **Supported** | Rice, gzip, H-Compress, and PLIO read/write paths exposed by CFITSIO. |
| Header cards and checksums | **Supported** | Typed metadata, `CHECKSUM`, and `DATASUM` verification/rewrite. |
| Random-group HDUs | **Unsupported** | The core reader does not decode the random-group data model. |
| `A3DTABLE` HDUs | **Unsupported** | The reader exposes image, binary-table, and ASCII-table HDUs. |
| WCS/coordinate transformations | **Out of Scope** | Use `astropy.wcs`; torchfits preserves standard header cards only. |
| HEALPix, units, sphere geometry, and sky simulation | **Out of Scope** | These belong to dedicated astronomy/domain packages. |
<!-- major-format-coverage:end -->

---

## 1. Images, Datacubes & Multi-Extension Files (MEF)

| Feature | Status | Comparator | Implementation Details & Behavior |
|---|:---:|---|---|
| **2D Image Reading & Writing** | **Supported** | `astropy.io.fits`, `fitsio` | Direct C++ decoding to PyTorch tensors (`float32`, `float64`, `uint8`, `int8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`). |
| **3D/4D Datacubes** | **Supported** | `astropy.io.fits`, `fitsio` | Reads multi-dimensional cubes (e.g. IFU datacubes, radio velocity channels) into $[D, H, W]$ tensors. |
| **Multi-Extension (MEF)** | **Supported** | `astropy.io.fits.HDUList` | HDU iteration by integer index or extension name (`hdul["SCI"]`), lazy metadata scanning. |
| **Windowed Cutout Reads** | **Supported** | `astropy.io.fits`, `fitsio` | Fast pixel sub-region extraction via `read_subset` and zero-overhead `open_subset_reader`. |
| **Unsigned and signed-byte integers** | **Supported** | `BZERO` convention | `uint16`/`uint32`/`uint64` are stored signed with `BZERO` = `2**15`/`2**31`/`2**63`, and `int8` as unsigned bytes with `BZERO` = `-128`; both read and write give exact integer tensors, as in astropy. fitsio cannot read a `uint64` image (CFITSIO status 412), whoever wrote it. |
| **Physical Scaling (`BSCALE`/`BZERO`)** | **Supported** | FITS Standard | Any other `BSCALE`/`BZERO` scales to `float32` for `BITPIX` 8/16 and `float64` for `BITPIX` 32/64, the same dtypes as astropy. The sum is applied in double precision before the final cast; a large `BZERO` therefore matches astropy to float32 rounding (a few 1e-6 relative on a real MegaCam frame), not bit-for-bit. |

---

## 2. Tile-Compressed Images (`.fits.fz`)

| Compression Algorithm | Status | Read Support | Write Support |
|---|:---:|---|---|
| **Rice (`RICE_1`)** | **Supported** | Fast tile decompression via CFITSIO backend | Compressed tile writing (`compress="rice"`) |
| **Gzip (`GZIP_1`, `GZIP_2`)** | **Supported** | Byte-level and tile-compressed gzip streams | Compressed tile writing (`compress="gzip"`) |
| **H-Compress (`HCOMPRESS_1`)** | **Supported** | Multi-resolution astronomical wavelet decoding | Compressed tile writing (`compress="hcompress"`) |
| **PLIO (`PLIO_1`)** | **Supported** | IRAF-style pixel list mask decoding | Non-negative integers below `2**24`, and full-range `uint16`. CFITSIO stores `uint16` tiles without the `BZERO` offset; fitsio agrees, astropy reads those pixels 32768 off. |

---

## 3. Astronomical Catalogs & Tables (`torchfits.table`)

| Table Feature | Status | Comparator | Implementation Details & Behavior |
|---|:---:|---|---|
| **Binary Tables (`BINTABLE`)** | **Supported** | `astropy.io.fits`, `fitsio` | High-throughput Arrow decoding into tables and PyTorch tensors. |
| **ASCII Tables (`TABLE`)** | **Supported** | `astropy.io.fits.TableHDU` | Full column parsing and type inference for fixed-width ASCII tables. |
| **Column Projection** | **Supported** | `columns=["ra", "dec"]` | Reads only requested columns from disk, skipping unneeded byte offsets. |
| **Row Slicing (`start_row`, `num_rows`)** | **Supported** | `fitsio` row limits | Reads contiguous row ranges without scanning earlier or later records. |
| **Predicate Filtering (`where=`)** | **Supported** | `fitsio` WHERE clauses | SQL expression pushdown (`"mag < 21.0 AND flag == 0"`) executed during scanning. `TNULL` cells and float NaN are unknown: no comparison selects them and `IS NULL` does. |
| **In-Place Table Mutation** | **Supported** | `table.update_rows(...)` | Fast in-place column and cell updating on disk via memory-mapping. |

---

## 4. Advanced Table Data Types

| Column Data Type | FITS Format Code | Support Status | Notes |
|---|:---:|:---:|---|
| **Numeric Columns** | `B`, `I`, `J`, `K`, `E`, `D` | **Supported** | Full integer and floating-point support. `int8` is written as `B` with `TZERO = -128`, and `uint16`/`uint32`/`uint64` as `I`/`J`/`K` with `TZERO = 2**15`/`2**31`/`2**63`; all read back as exact integers (astropy reads the `int8` form as scaled `float64`). |
| **Fixed-Width Strings** | `nA` | **Supported** | Decoded as Arrow string arrays with automatic space padding on update. |
| **Boolean Bitmasks** | `X` | **Supported** | Bit-level MSB-first decoding into boolean tensors. |
| **Complex Numbers** | `C`, `M` | **Partial** | Tensor path (`table.read_torch`) decodes `torch.complex64` / `complex128`. Arrow `table.read` raises `NotImplementedError`. |
| **Variable-Length Arrays (VLA)** | `P`, `Q` | **Partial** | Supported via buffered CFITSIO reading; memory-mapped updates not supported. |
| **Scaled Columns (`TSCALn`/`TZEROn`)** | `TSCAL`, `TZERO` | **Supported** | A `TZERO` that is one of the integer conventions above (with `TSCAL = 1`) keeps an exact integer dtype; any other scaling reads as `float64`, with `TNULL` cells as NaN. |

---

## 5. Headers & Metadata

| Capability | Status | Notes |
|---|:---:|---|
| **Header Card Access** | **Supported** | Dict-like and attribute access to header keywords, comments, and values. |
| **Header Modification & Export** | **Supported** | Mutate cards and preserve metadata when writing new FITS files. |
| **Checksum & Datasum Verification** | **Supported** | Computes and validates standard FITS `CHECKSUM` and `DATASUM` cards. |

---

## 6. PyTorch ML & Accelerator Integration

| Feature | Status | Notes |
|---|:---:|---|
| **Direct GPU Tensor Placement** | **Supported** | Load FITS images directly onto `device="cuda"` or `device="mps"` without NumPy intermediate steps. |
| **PyTorch `Dataset` Integration** | **Supported** | Built-in `FitsImageDataset`, `FitsCutoutDataset`, `FitsCubeDataset`, and `FitsTableDataset`. |
| **Multi-Worker `DataLoader`** | **Supported** | `make_loader` is a `DataLoader` factory with FITS-aware batch collation; datasets fetch each remote file once into the disk cache. |
| **Header-Aware Transforms** | **Supported** | Astronomical image stretches (Arcsinh, ZScale, Lupton RGB) implemented as PyTorch transforms. |

---

## Scope & Design Philosophy

`torchfits` is designed specifically for high-throughput, low-latency FITS tensor and table I/O.

High-level astronomical analysis tools (such as world coordinate transformations with `astropy.wcs` or unit modeling with `astropy.units`) remain the domain of Astropy and companion libraries. `torchfits` integrates with these libraries by outputting standard PyTorch tensors and Arrow tables.
