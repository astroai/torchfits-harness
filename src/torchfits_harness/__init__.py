"""Evaluation package for an installed torchfits subject.

This tree does not build torchfits or vendor CFITSIO. Install a torchfits
wheel first, then this package with ``pip install -e . --no-deps``.
"""

from __future__ import annotations


def subject() -> str:
    """Return the imported torchfits version and file path."""
    import torchfits

    return f"{torchfits.__version__} {torchfits.__file__}"
