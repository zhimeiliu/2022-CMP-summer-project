"""Run tests on `patch_extraction.patch_extractor.extractor.PatchExtractor`."""
from pathlib import Path
import pytest

from lyzeum.patch_extraction import PatchExtractor


def test_patch_extractor_instantation():
    """Test the instantiation of `PatchExtractor` works."""
    PatchExtractor(
        patch_size=224,
        stride=2,
        min_patch_mag=5.0,
        max_patch_mag=20.0,
        overview_mag=2.5,
        background_tol=0.6,
        top_dir="here",
        cleanup_workers=1,
        qupath_binary="qupath",
    )


def test_patch_extractor_patch_size_arg():
    """Test the `patch_size` argument of `PatchExtractor`."""
    # Should accept a positive integer.
    PatchExtractor(patch_size=224)

    with pytest.raises(TypeError):
        PatchExtractor(patch_size=224.0)
    with pytest.raises(TypeError):
        PatchExtractor(patch_size="hello")
    with pytest.raises(ValueError):
        PatchExtractor(patch_size=-2)
    with pytest.raises(ValueError):
        PatchExtractor(patch_size=0)


def test_patch_extractor_stride_arg():
    """Test the `stride` argument of `PatchExtractor`."""
    patch_size = 224
    # Should accept a positive integer greater than zero.
    PatchExtractor(patch_size, 1)

    # Check the stride argument only accepts integer types.
    with pytest.raises(TypeError):
        PatchExtractor(patch_size=patch_size, stride=2.0)
    with pytest.raises(TypeError):
        PatchExtractor(patch_size=patch_size, stride="Batman")

    # Check the stride argument only accepts values >= 1.
    with pytest.raises(ValueError):
        PatchExtractor(patch_size=patch_size, stride=0)
    with pytest.raises(ValueError):
        PatchExtractor(patch_size=patch_size, stride=-1)


def test_patch_extractor_min_patch_mag_arg():
    """Test `min_patch_mag` argument."""
    patch_size = 224

    PatchExtractor(
        patch_size=patch_size,
        min_patch_mag=5.0,
        max_patch_mag=40.0,
    )

    with pytest.raises(TypeError):
        PatchExtractor(patch_size=patch_size, min_patch_mag=40)
    with pytest.raises(TypeError):
        PatchExtractor(patch_size=patch_size, min_patch_mag=1j)

    with pytest.raises(ValueError):
        PatchExtractor(patch_size=patch_size, min_patch_mag=-2.0)
    with pytest.raises(ValueError):
        PatchExtractor(patch_size=patch_size, min_patch_mag=0.5)


def test_patch_extractor_max_patch_mag_arg():
    """Test the `max_patch_mag` argument of `PatchExtractor`."""
    patch_size = 224

    # Should work with float greater than 1.
    PatchExtractor(
        patch_size=patch_size,
        min_patch_mag=5.0,
        max_patch_mag=40.0,
    )

    with pytest.raises(TypeError):
        PatchExtractor(patch_size=patch_size, max_patch_mag=40)
    with pytest.raises(TypeError):
        PatchExtractor(patch_size=patch_size, max_patch_mag=1j)

    with pytest.raises(ValueError):
        PatchExtractor(patch_size=patch_size, max_patch_mag=-2.0)
    with pytest.raises(ValueError):
        PatchExtractor(patch_size=patch_size, max_patch_mag=0.5)


def test_patch_extractor_patch_magnifcation_arguments():
    """Test min and max patch magnification arguments of `PatchExtractor`."""
    # Min mag cannot be greater than max mag
    with pytest.raises(ValueError):
        PatchExtractor(patch_size=224, min_patch_mag=40.0, max_patch_mag=20.0)

    PatchExtractor(patch_size=224, min_patch_mag=20.0, max_patch_mag=20.0)


def test_patch_extractor_overview_mag_arg():
    """Test the `overview_mag` argument of `PatchExtractor`."""
    patch_size = 224

    # Should work with float greater than 1.
    PatchExtractor(patch_size=patch_size, overview_mag=40.0)

    with pytest.raises(TypeError):
        PatchExtractor(patch_size=patch_size, overview_mag=40)
    with pytest.raises(TypeError):
        PatchExtractor(patch_size=patch_size, overview_mag=1j)

    with pytest.raises(ValueError):
        PatchExtractor(patch_size=patch_size, overview_mag=-2.0)
    with pytest.raises(ValueError):
        PatchExtractor(patch_size=patch_size, overview_mag=0.5)


def test_patch_extractor_background_tol_argument():
    """Test the `background_tol` argument of `PatchExtractor`."""
    patch_size = 224

    # Should work with float on 0 <= background_tol <= 1
    PatchExtractor(patch_size, background_tol=0.0)
    PatchExtractor(patch_size, background_tol=0.5)
    PatchExtractor(patch_size, background_tol=1.0)

    with pytest.raises(TypeError):
        PatchExtractor(patch_size, background_tol=1)
    with pytest.raises(TypeError):
        PatchExtractor(patch_size, background_tol="Robin")

    with pytest.raises(ValueError):
        PatchExtractor(patch_size, background_tol=1.001)
    with pytest.raises(ValueError):
        PatchExtractor(patch_size, background_tol=-0.001)


def test_patch_extractor_top_dir_argument():
    """Test the `top_dir` argument of `PatchExtractor`."""
    # Shouold work with Path or str
    PatchExtractor(patch_size=224, top_dir="Here")
    PatchExtractor(patch_size=224, top_dir=Path("Here"))

    with pytest.raises(TypeError):
        PatchExtractor(patch_size=224, top_dir=1234)
    with pytest.raises(TypeError):
        PatchExtractor(patch_size=224, top_dir=3.14)
    with pytest.raises(TypeError):
        PatchExtractor(patch_size=224, top_dir=[10])


def test_patch_extractor_cleanup_workers_argument():
    """Test the `cleanup_workers` argument of `PatchExtractor`."""
    # Should work for positive int or None
    PatchExtractor(patch_size=224, cleanup_workers=12)
    PatchExtractor(patch_size=224, cleanup_workers=None)

    with pytest.raises(TypeError):
        PatchExtractor(patch_size=224, cleanup_workers=1.0)
    with pytest.raises(TypeError):
        PatchExtractor(patch_size=224, cleanup_workers="Mr Freeze.")

    with pytest.raises(ValueError):
        PatchExtractor(patch_size=224, cleanup_workers=0)
    with pytest.raises(ValueError):
        PatchExtractor(patch_size=224, cleanup_workers=-2)


def test_patch_extractor_qupath_argument():
    """Test the `qupath` argument of `PatchExtractor`."""
    with pytest.raises(FileNotFoundError):
        PatchExtractor(patch_size=224, qupath_binary="/a/bad/path")
