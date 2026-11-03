"""
Test caching functionality.
"""

import os
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest
import torch

import torchfits


class TestCaching:
    """Test file caching functionality."""

    def create_test_fits(self, shape=(100, 100)):
        """Create a test FITS file."""
        data = np.random.normal(0, 1, shape).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            hdu = fits.PrimaryHDU(data)
            hdu.writeto(f.name, overwrite=True)
            return f.name, data

    def test_cache_performance_tracking(self):
        """Test cache hit/miss tracking.

        Deep-review unit 13, R2-057: this asserted only
        ``total_requests`` growing, which a counter that incremented on every
        request satisfies whether or not anything was ever cached -- deleting
        the body of ``store_cached_read`` left all 58 tests in this file green.
        The hit/miss *transition* is the contract, so both arms are pinned.

        ``return_header=True`` is what routes the read through the fallback
        path, the only one that populates ``file_cache``; a bare
        ``read(filepath)`` of an image never caches and would read as two
        misses here.
        """
        filepath, _ = self.create_test_fits()

        try:
            # Clear cache first
            torchfits.clear_file_cache()

            # First read should be a cache miss
            torchfits.read(filepath, return_header=True, cache_capacity=8)
            stats1 = torchfits.get_cache_performance()
            assert stats1["total_requests"] == 1, stats1
            assert stats1["hits"] == 0, stats1
            assert stats1["misses"] == 1, stats1
            assert stats1["cache_size"] == 1, "the read stored nothing to serve later"

            # Second read should be a cache hit
            torchfits.read(filepath, return_header=True, cache_capacity=8)
            stats2 = torchfits.get_cache_performance()
            assert stats2["total_requests"] == 2, stats2
            assert stats2["hits"] == 1, (
                f"second read was not served from cache: {stats2}"
            )
            assert stats2["misses"] == 1, "a hit must not also count as a miss"
            # A served-from-cache read must return the same bytes as a cold one.
            torchfits.clear_file_cache()
            cold_data, _cold_header = torchfits.read(
                filepath, return_header=True, cache_capacity=8
            )
            warm_data, _warm_header = torchfits.read(
                filepath, return_header=True, cache_capacity=8
            )
            assert torch.equal(warm_data, cold_data)

        finally:
            os.unlink(filepath)
            torchfits.clear_file_cache()

    def test_get_cache_stats(self):
        """Test get_cache_stats returns expected dictionary structure."""
        from torchfits.cache import get_cache_stats, clear_cache

        # Clear cache to start with a known state
        clear_cache()

        stats = get_cache_stats()

        # Verify it's a dictionary
        assert isinstance(stats, dict)

        # The I/O engine's counters; the policy "config" block went with the
        # cache-policy objects in 1.3.1.
        expected_keys = {"hits", "misses", "hit_rate", "total_requests", "cache_size"}
        assert expected_keys.issubset(stats.keys())
        assert "config" not in stats

        # Verify types of specific fields
        assert isinstance(stats["hits"], int)
        assert isinstance(stats["misses"], int)
        assert isinstance(stats["hit_rate"], float)

        # Basic hit_rate calculation check (should be 0.0 when hits=0, misses=0)
        assert stats["hit_rate"] == 0.0

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        filepath, _ = self.create_test_fits()

        try:
            # Read file to populate cache
            torchfits.read(filepath)
            torchfits.get_cache_performance()

            # Clear cache
            torchfits.clear_file_cache()

            # Verify cache is cleared. This test previously made no assertion
            # at all (deep-review unit 10, TE-003): the comment said "stats
            # should be reset" and nothing checked it, so a clear_file_cache
            # that did nothing would have passed.
            stats = torchfits.get_cache_performance()
            assert stats["total_requests"] == 0, stats
            assert stats["hits"] == 0, stats
            assert stats["misses"] == 0, stats

            # And a read after the clear is served cold, not from the cache.
            torchfits.read(filepath)
            assert torchfits.get_cache_performance()["misses"] >= 1

        finally:
            os.unlink(filepath)
            torchfits.clear_file_cache()

    def test_concurrent_python_lru_access_preserves_outputs(self):
        """Concurrent metadata/LRU access must not corrupt read results."""
        filepath, expected = self.create_test_fits((64, 48))

        def read_once(_: int) -> np.ndarray:
            torchfits.read_header(filepath, hdu=0)
            return np.asarray(torchfits.read(filepath, hdu=0, mmap="auto").numpy())

        try:
            torchfits.clear_file_cache()
            with ThreadPoolExecutor(max_workers=8) as pool:
                outputs = list(pool.map(read_once, range(32)))
            for output in outputs:
                np.testing.assert_array_equal(output, expected)
        finally:
            os.unlink(filepath)
            torchfits.clear_file_cache()

    def test_multiple_file_caching(self):
        """Test caching with multiple files."""
        files = []

        try:
            # Create multiple test files
            for i in range(5):
                filepath, _ = self.create_test_fits((50 + i * 10, 50 + i * 10))
                files.append(filepath)

            # Clear cache
            torchfits.clear_file_cache()

            # Read all files
            for filepath in files:
                torchfits.read(filepath, return_header=True, cache_capacity=16)

            # Read them again (should hit cache)
            for filepath in files:
                torchfits.read(filepath, return_header=True, cache_capacity=16)

            # Check cache performance
            stats = torchfits.get_cache_performance()
            assert stats["total_requests"] >= len(files) * 2
            # R2-057: the comment above says the second pass hits the cache;
            # before this line nothing checked it. See
            # test_cache_performance_tracking for the mutation evidence.
            assert stats["hits"] >= len(files), (
                f"no file was served from cache: {stats}"
            )
            assert stats["misses"] >= len(files), stats

        finally:
            for f in files:
                if os.path.exists(f):
                    os.unlink(f)
            torchfits.clear_file_cache()

    def test_cached_multibyte_read_matches_nocache_reference(self):
        """Regression: cached mmap raw path must match the no-cache path exactly."""
        cpp = pytest.importorskip("torchfits._cpp")
        if not hasattr(cpp, "read_full_cached") or not hasattr(
            cpp, "read_full_nocache"
        ):
            pytest.skip("cached/nocache read methods unavailable in this build")

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            data = np.arange(128 * 64, dtype=np.int32).reshape(128, 64) - 1000
            fits.PrimaryHDU(data).writeto(f.name, overwrite=True)
            path = f.name

        try:
            torchfits.clear_file_cache()
            cached = cpp.read_full_cached(path, 0, True).numpy()
            reference = cpp.read_full_nocache(path, 0, True).numpy()
            np.testing.assert_array_equal(cached, reference)
        finally:
            if os.path.exists(path):
                os.unlink(path)
            torchfits.clear_file_cache()

    def test_cold_nommap_heuristic_with_cache_enabled_int16(self):
        """Large int16 images should prefer non-mmap even when cache is enabled."""
        import torchfits.io

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            data = np.arange(1024 * 1024, dtype=np.int16).reshape(1024, 1024)
            fits.PrimaryHDU(data).writeto(f.name, overwrite=True)
            path = f.name

        try:
            torchfits.clear_file_cache()
            assert torchfits.io._should_use_cold_nommap(
                path, 0, cache_capacity=10, mmap=True
            )
        finally:
            if os.path.exists(path):
                os.unlink(path)
            torchfits.clear_file_cache()

    def test_cold_nommap_heuristic_float64_and_small_guard(self):
        """64-bit and sub-1MiB payloads should keep mmap enabled by default."""
        import torchfits.io

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f_large:
            from astropy.io import fits

            large = np.random.randn(1024, 1024).astype(np.float64)  # ~8 MiB payload
            fits.PrimaryHDU(large).writeto(f_large.name, overwrite=True)
            large_path = f_large.name
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f_small:
            from astropy.io import fits

            small = np.random.randn(128, 128).astype(np.float64)  # <1 MiB payload
            fits.PrimaryHDU(small).writeto(f_small.name, overwrite=True)
            small_path = f_small.name

        try:
            torchfits.clear_file_cache()
            assert not torchfits.io._should_use_cold_nommap(
                large_path, 0, cache_capacity=10, mmap=True
            )
            assert not torchfits.io._should_use_cold_nommap(
                small_path, 0, cache_capacity=10, mmap=True
            )
        finally:
            for path in (large_path, small_path):
                if os.path.exists(path):
                    os.unlink(path)
            torchfits.clear_file_cache()

    def test_read_mmap_auto_defaults_to_false_for_compressed(self, monkeypatch):
        """`mmap='auto'` should disable mmap for compressed HDUs."""
        import torchfits.io
        import torchfits._C as cpp

        if not hasattr(cpp, "read_full_cached"):
            pytest.skip("read_full_cached unavailable in this build")

        filepath, _ = self.create_test_fits()
        observed = []

        monkeypatch.setattr(
            torchfits.io,
            "_get_image_meta",
            lambda path, hdu: (-32, 2, (64, 64), 1.0, 0.0, True),
        )
        monkeypatch.setattr(
            torchfits.io,
            "_should_use_cold_nommap",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError(
                    "cold nommap heuristic should not run for compressed auto mode"
                )
            ),
        )
        monkeypatch.setattr(
            cpp,
            "read_full_cached",
            lambda path, hdu, use_mmap: (
                observed.append(bool(use_mmap)),
                torch.zeros((8, 8), dtype=torch.float32),
            )[1],
        )
        monkeypatch.setattr(
            cpp,
            "read_full",
            lambda path, hdu, use_mmap: (
                observed.append(bool(use_mmap)),
                torch.zeros((8, 8), dtype=torch.float32),
            )[1],
        )

        try:
            out = torchfits.read(
                filepath,
                hdu=0,
                mmap="auto",
                cache_capacity=10,
            )

            assert isinstance(out, torch.Tensor)
            assert observed == [False]
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
            torchfits.clear_file_cache()

    def test_read_mmap_true_is_explicit_override(self, monkeypatch):
        """`mmap=True` must stay enabled for compressed metadata reads."""
        import torchfits.io
        import torchfits._C as cpp

        if not hasattr(cpp, "read_full_cached"):
            pytest.skip("read_full_cached unavailable in this build")

        filepath, _ = self.create_test_fits()
        observed = []

        monkeypatch.setattr(
            torchfits.io,
            "_get_image_meta",
            lambda path, hdu: (-32, 2, (64, 64), 1.0, 0.0, True),
        )
        monkeypatch.setattr(
            torchfits.io,
            "_should_use_cold_nommap",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError(
                    "cold nommap heuristic should not run for explicit mmap=True"
                )
            ),
        )
        monkeypatch.setattr(
            cpp,
            "read_full_cached",
            lambda path, hdu, use_mmap: (
                observed.append(bool(use_mmap)),
                torch.zeros((8, 8), dtype=torch.float32),
            )[1],
        )
        monkeypatch.setattr(
            cpp,
            "read_full",
            lambda path, hdu, use_mmap: (
                observed.append(bool(use_mmap)),
                torch.zeros((8, 8), dtype=torch.float32),
            )[1],
        )

        try:
            out = torchfits.read(
                filepath,
                hdu=0,
                mmap=True,
                cache_capacity=10,
            )

            assert isinstance(out, torch.Tensor)
            assert observed == [True]
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
            torchfits.clear_file_cache()

    def test_read_mmap_auto_uses_direct_mmap(self, monkeypatch):
        """`mmap='auto'` on a plain uncompressed HDU should use direct mmap."""
        import torchfits.io
        import torchfits._C as cpp

        filepath, _ = self.create_test_fits()
        observed = []
        monkeypatch.setattr(
            cpp,
            "read_full",
            lambda path, hdu, use_mmap: (
                observed.append(bool(use_mmap)),
                torch.zeros((4, 4), dtype=torch.float32),
            )[1],
        )
        if hasattr(cpp, "read_full_cached"):
            monkeypatch.setattr(
                cpp,
                "read_full_cached",
                lambda path, hdu, use_mmap: (
                    observed.append(bool(use_mmap)),
                    torch.zeros((4, 4), dtype=torch.float32),
                )[1],
            )

        try:
            out = torchfits.read(
                filepath,
                hdu=0,
                mmap="auto",
                cache_capacity=10,
            )

            assert isinstance(out, torch.Tensor)
            assert observed == [True]
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
            torchfits.clear_file_cache()

    def test_read_rejects_invalid_mmap_mode(self):
        """Only bool or 'auto' mmap mode should be accepted."""
        with pytest.raises(ValueError, match="mmap must be bool or 'auto'"):
            torchfits.read("dummy.fits", mmap="sometimes")

    def test_read_hdu_auto_detects_compressed_image_extension(self):
        """`hdu='auto'` should resolve to the first payload HDU (fitsio-like behavior)."""
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            image = np.random.normal(size=(64, 64)).astype(np.float32)
            primary = fits.PrimaryHDU()
            compressed = fits.CompImageHDU(image, compression_type="RICE_1")
            fits.HDUList([primary, compressed]).writeto(f.name, overwrite=True)
            path = f.name

        try:
            empty_primary = torchfits.read(path, hdu=0, mmap="auto")
            auto_tensor = torchfits.read(path, hdu="auto", mmap="auto")
            none_tensor = torchfits.read(path, hdu=None, mmap="auto")

            assert isinstance(empty_primary, torch.Tensor)
            assert empty_primary.numel() == 0
            assert isinstance(auto_tensor, torch.Tensor)
            assert isinstance(none_tensor, torch.Tensor)
            assert tuple(auto_tensor.shape) == image.shape
            assert tuple(none_tensor.shape) == image.shape
        finally:
            if os.path.exists(path):
                os.unlink(path)
            torchfits.clear_file_cache()

    def test_read_header_auto_matches_detected_hdu(self):
        """`read_header(..., hdu='auto')` should return the detected payload header."""
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            image = np.random.normal(size=(32, 48)).astype(np.float32)
            primary = fits.PrimaryHDU()
            compressed = fits.CompImageHDU(image, compression_type="RICE_1")
            fits.HDUList([primary, compressed]).writeto(f.name, overwrite=True)
            path = f.name

        try:
            header = torchfits.read_header(path, hdu="auto")
            assert int(header.get("NAXIS", 0)) == 2
            assert int(header.get("NAXIS1", 0)) > 0
            assert int(header.get("NAXIS2", 0)) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)
            torchfits.clear_file_cache()


class TestCacheModuleFunctions:
    """The cache helpers that survive the 1.3.1 removal of the policy objects."""

    def test_clear_cache_resets_engine_counters(self, tmp_path):
        from torchfits.cache import clear_cache, get_cache_stats

        path = tmp_path / "img.fits"
        torchfits.write_tensor(str(path), torch.ones(4, 4), overwrite=True)
        torchfits.read(str(path), return_header=True, cache_capacity=8)
        torchfits.read(str(path), return_header=True, cache_capacity=8)
        assert get_cache_stats()["total_requests"] > 0
        clear_cache()
        stats = get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0


def test_cached_reads_are_isolated_from_caller_mutation(tmp_path):
    """In-place mutation of a returned result must not poison the cache.

    Regression: cached read results were handed out (and stored) by
    reference, so ``result['flux'].mul_(2)`` silently corrupted every later
    read served from the default on-by-default cache.
    """
    import torch

    import torchfits

    table_path = tmp_path / "alias.fits"
    torchfits.write(
        table_path.as_posix(),
        {"flux": torch.arange(10, dtype=torch.float32)},
        overwrite=True,
    )
    image_path = tmp_path / "img.fits"
    torchfits.write_tensor(image_path.as_posix(), torch.ones(4, 4), overwrite=True)

    expected = torch.arange(10, dtype=torch.float32)

    d1 = torchfits.read(table_path.as_posix(), hdu=1)
    d1["flux"].mul_(100)
    assert torch.equal(torchfits.read(table_path.as_posix(), hdu=1)["flux"], expected)

    d3 = torchfits.read(table_path.as_posix(), hdu=1)
    d3["flux"].mul_(-1)
    assert torch.equal(torchfits.read(table_path.as_posix(), hdu=1)["flux"], expected)

    i1 = torchfits.read(image_path.as_posix())
    i1.add_(5)
    assert torch.equal(torchfits.read(image_path.as_posix()), torch.ones(4, 4))


class TestRemovedCachePolicyObjects:
    """1.3.1 removed the cache-policy objects instead of keeping no-ops.

    torchfits' changelog (1.3.1rc2, Removed): "Remove ineffective cache-policy
    configuration". Pin that they stay gone, so a stale caller fails loudly
    instead of configuring nothing.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "CacheConfig",
            "CacheManager",
            "get_cache_manager",
            "configure_for_environment",
            "optimize_for_dataset",
            "get_optimal_cache_config",
            "configure_cache",
        ],
    )
    def test_policy_object_is_gone(self, name):
        assert not hasattr(torchfits.cache, name)

    def test_cache_stats_surface_engine_errors(self, monkeypatch):
        """Stats aggregation must not silently swallow engine failures."""
        import torchfits._io_engine.caches as caches

        def boom() -> dict:
            raise RuntimeError("engine broken")

        monkeypatch.setattr(caches, "get_cache_performance", boom)
        with pytest.raises(RuntimeError, match="engine broken"):
            torchfits.cache.get_cache_stats()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Deep-review unit 10, TE-002: the selective contract of clear_file_cache
# ---------------------------------------------------------------------------
#
# `clear_file_cache` takes six keyword-only flags -- the entire reason it
# accepts arguments at all -- and `docs/api-core-io.md` documents each one.
# Nothing pinned them. Verified by breaking the implementation: making
# `clear_python_caches` ignore all five Python-side keywords, and making
# `cpp=False` still clear the native cache, each left the whole cache-related
# test selection green (122 passed, 1 skipped).
#
# The caches are module-level OrderedDicts, so the flag -> cache mapping is
# asserted directly. That is deliberately implementation-coupled: the contract
# under test *is* which store each flag clears.


def _fill_all_caches() -> None:
    from torchfits._io_engine import caches

    caches.file_cache[("data",)] = None
    caches.image_meta_cache[("meta", 0)] = None
    caches.header_cards_cache[("meta", 0)] = None
    caches.cold_nommap_cache[("meta", 0)] = None
    caches.auto_mmap_cache[("meta", 0)] = None
    caches.hdu_type_cache[("hdu", 0)] = None
    caches.auto_hdu_cache[("auto",)] = None
    caches.cache_stats["hits"] += 7


_ALL_CACHES = (
    "file_cache",
    "image_meta_cache",
    "header_cards_cache",
    "cold_nommap_cache",
    "auto_mmap_cache",
    "hdu_type_cache",
    "auto_hdu_cache",
)


@pytest.mark.parametrize(
    ("flags", "preserved"),
    [
        ({"data": False}, ("file_cache",)),
        (
            {"meta": False},
            (
                "image_meta_cache",
                "header_cards_cache",
                "cold_nommap_cache",
                "auto_mmap_cache",
            ),
        ),
        ({"hdu_types": False}, ("hdu_type_cache", "auto_hdu_cache")),
    ],
)
def test_clear_file_cache_preserves_only_the_named_caches(flags, preserved):
    """Each keyword must gate exactly the caches it is documented to gate."""
    from torchfits._io_engine import caches

    torchfits.clear_file_cache()
    _fill_all_caches()
    torchfits.clear_file_cache(cpp=False, **flags)

    for name in _ALL_CACHES:
        size = len(getattr(caches, name))
        if name in preserved:
            assert size == 1, f"{flags}: {name} must be preserved, has {size} entries"
        else:
            assert size == 0, f"{flags}: {name} must be cleared, has {size} entries"
    # `stats` was not named, so it is reset.
    assert caches.cache_stats["hits"] == 0, f"{flags}: stats must reset"


def test_clear_file_cache_stats_false_keeps_counters():
    """``stats=False`` is the one flag that preserves rather than clears."""
    from torchfits._io_engine import caches

    torchfits.clear_file_cache()
    _fill_all_caches()
    torchfits.clear_file_cache(cpp=False, stats=False)

    assert caches.cache_stats["hits"] == 7, "stats=False must keep the counters"
    for name in _ALL_CACHES:
        assert len(getattr(caches, name)) == 0, f"{name} must still be cleared"


def test_clear_file_cache_cpp_flag_is_honoured():
    """``cpp=False`` must not touch the native cache, ``cpp=True`` must."""
    from unittest import mock

    fake_cpp = mock.MagicMock()

    torchfits.clear_file_cache(cpp=False, cpp_module=fake_cpp)
    fake_cpp.clear_shared_read_meta_cache.assert_not_called()

    torchfits.clear_file_cache(cpp=True, cpp_module=fake_cpp)
    fake_cpp.clear_shared_read_meta_cache.assert_called_once()


# --------------------------------------------------------------------------
# TS-014: `cache_subsystem_policy` is documented in `docs/api-core-io.md` and
# had no test. TS-002 pinned `clear_file_cache`'s *selective* keyword contract;
# this is the same contract one layer up -- the policy table that decides which
# keywords a named subsystem clears -- and it was equally unpinned. A regression
# that made `fits_header_metadata` clear `data` too, or clear nothing, would
# have been silent.
# --------------------------------------------------------------------------

# The measured table: subsystem name -> the flags it actually enables.
_POLICY_FLAGS = {
    "fits_image_data": {"data"},
    "fits_table_data": {"data"},
    "fits_header_metadata": {"meta", "hdu_types"},
    "fits_header_hdu_metadata": {"meta", "hdu_types"},
}


def _enabled(flags: dict) -> set:
    return {name for name, on in flags.items() if on}


@pytest.mark.parametrize("name, expected", sorted(_POLICY_FLAGS.items()))
def test_each_subsystem_enables_exactly_its_own_flags(name, expected):
    """The selective part: clearing `fits_image_data` must not touch metadata."""
    from torchfits.io import cache_subsystem_policy

    flags = cache_subsystem_policy(name)
    assert _enabled(flags) == expected, flags
    # The handle cache was removed, so the policy no longer names it.
    assert "handles" not in flags


def test_all_subsystem_enables_every_flag():
    from torchfits.io import cache_subsystem_policy

    flags = cache_subsystem_policy("all")
    assert all(flags.values()), flags
    assert set(flags) == {"data", "meta", "hdu_types", "stats", "cpp"}


def test_policy_returns_a_copy_not_the_shared_table():
    """Mutating the result must not corrupt the policy for the next caller."""
    from torchfits.io import cache_subsystem_policy

    first = cache_subsystem_policy("fits_image_data")
    first["data"] = False
    first["injected"] = True

    second = cache_subsystem_policy("fits_image_data")
    assert second["data"] is True
    assert "injected" not in second


def test_unknown_subsystem_lists_the_valid_names():
    """The error is the only place the four non-"all" names are discoverable."""
    from torchfits.io import cache_subsystem_policy

    with pytest.raises(KeyError) as err:
        cache_subsystem_policy("bogus")
    message = str(err.value)
    assert "bogus" in message
    for name in ["all", *_POLICY_FLAGS]:
        assert name in message, f"{name} missing from the error message"


@pytest.mark.parametrize("flag_name", ["data", "meta", "hdu_types", "handles"])
def test_clear_file_cache_flag_names_are_not_subsystem_names(flag_name):
    """A namespace trap worth pinning: the *keyword* names are not *subsystems*.

    `clear_file_cache(data=...)` takes these words as keywords, so
    `cache_subsystem_policy("data")` is the natural-looking call -- and it
    raises. The valid names are the `fits_*` ones.
    """
    from torchfits.io import cache_subsystem_policy

    with pytest.raises(KeyError):
        cache_subsystem_policy(flag_name)


def test_clear_cache_subsystem_forwards_the_policy_flags():
    """The policy is only useful if `clear_cache_subsystem` actually applies it."""
    from unittest import mock

    from torchfits._io_engine import caches

    with mock.patch.object(caches, "clear_file_cache") as clear:
        caches.clear_cache_subsystem("fits_header_metadata")
    kwargs = clear.call_args.kwargs
    assert kwargs["meta"] is True
    assert kwargs["hdu_types"] is True
    assert kwargs["data"] is False
    assert "handles" not in kwargs, "the removed handle cache must not be forwarded"
