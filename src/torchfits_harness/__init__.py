"""Evaluation package for an installed torchfits subject.

This tree does not build torchfits or vendor CFITSIO, and torchfits is not a
dependency: install the subject with ``pixi run subject <wheel-or-spec>``.
"""

from __future__ import annotations


def subject() -> str:
    """Return the imported torchfits version and file path."""
    import torchfits

    return f"{torchfits.__version__} {torchfits.__file__}"
