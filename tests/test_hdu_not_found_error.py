"""torchfits.HDUNotFoundError: one type for unknown or out-of-range HDUs.

It must stay catchable as KeyError, IndexError, ValueError and RuntimeError
(the types different routes raised in 1.3.1), and carry the request.
"""

import pickle

import numpy as np
import pytest

import torchfits


def test_hdu_not_found_error_on_read_and_table_routes(tmp_path):
    path = str(tmp_path / "x.fits")
    torchfits.write(path, np.zeros((2, 2), dtype=np.float32), overwrite=True)
    for call in (
        lambda: torchfits.read(path, hdu=5),
        lambda: torchfits.read(path, hdu="NOPE"),
        lambda: torchfits.read_header(path, 5),
    ):
        with pytest.raises(torchfits.HDUNotFoundError) as excinfo:
            call()
        err = excinfo.value
        for base in (KeyError, IndexError, ValueError, RuntimeError):
            assert isinstance(err, base)
        assert err.path == path
        assert "x.fits" in str(err)


def test_hdu_not_found_error_for_request_and_pickle(tmp_path):
    err = torchfits.HDUNotFoundError.for_request("f.fits", 7, 2)
    assert (err.path, err.hdu, err.n_hdus) == ("f.fits", 7, 2)
    assert "out of range" in str(err)
    again = pickle.loads(pickle.dumps(err))
    assert (again.path, again.hdu, again.n_hdus) == ("f.fits", 7, 2)
