"""Run tests on `PatchExtractor.__call__`."""
from pathlib import Path
from shutil import rmtree

import pytest
import numpy as np
from skimage.io import imsave

from lyzeum.patch_extraction import PatchExtractor

almost_blank_path = Path(".blank.tif")
noisy_path = Path(".noisy.tif")
top_dir = Path(".patches")


@pytest.fixture(scope="session", autouse=True)
def create_pretend_wsis():
    """Create some pretend WSIs to test calling patch extractor on."""
    size = (2 ** 12, 2 ** 12)
    almost_blank_img = np.full(size, 255, dtype=np.uint8)
    almost_blank_img[0, 0] = 0
    noisy_img = np.random.randint(256, size=size, dtype=np.uint8)

    imsave(almost_blank_path, almost_blank_img, check_contrast=False)
    imsave(noisy_path, noisy_img, check_contrast=False)

    yield

    almost_blank_path.unlink()
    noisy_path.unlink()


@pytest.fixture()
def delete_patch_dir():
    """Cleanup the generated patches after a test."""
    yield
    rmtree(top_dir)


def test_wsi_path_arg_path(delete_patch_dir):
    """Test the wsi_path argument in call."""
    extractor = PatchExtractor(top_dir=".patches")
    # Should work with a Path.
    extractor(noisy_path, user_supplied_mag=20.0)


def test_wsi_path_arg_str(delete_patch_dir):
    """Test the wsi_path argument in call."""
    extractor = PatchExtractor(top_dir=top_dir)
    # Should work with a Path.
    extractor(str(noisy_path), user_supplied_mag=20.0)


def test_wsi_path_arg_with_bad_inputs():
    """Test the wsi_path argument with bad types."""
    extractor = PatchExtractor(top_dir=top_dir)

    # Should break with non str or non Path
    with pytest.raises(TypeError):
        extractor(1)
    with pytest.raises(TypeError):
        extractor(["path"])

    # Should break for str or path which does not lead to a file
    with pytest.raises(FileNotFoundError):
        extractor(Path("/not/the/droids/you/are/looking/for.tif"))
    with pytest.raises(FileNotFoundError):
        extractor("/not/the/droids/you/are/looking/for.tif")

    # Should also break if the path is to a directory and not a file
    dir_path = Path(".a_folder_test/")
    dir_path.mkdir()
    with pytest.raises(IsADirectoryError):
        extractor(dir_path)
    dir_path.rmdir()


def test_region_arg_types(delete_patch_dir):
    """Test the region argument types."""
    extractor = PatchExtractor(top_dir=top_dir)

    # Should work with region as dictionary
    extractor(
        noisy_path,
        region={"left": 0, "top": 0, "width": 0, "height": 0},
        user_supplied_mag=20.0,
    )

    # Should break if region isn't a dictionary
    with pytest.raises(TypeError):
        extractor(
            noisy_path,
            region=[0, 0, 0, 0],
            user_supplied_mag=20.0,
        )

    with pytest.raises(TypeError):
        extractor(
            noisy_path,
            region=(0, 0, 0, 0),
            user_supplied_mag=20.0,
        )


def test_region_arg_keys(delete_patch_dir):
    """Test the keys of the region argument."""
    extractor = PatchExtractor(top_dir=top_dir)

    # Should work with the keys: "left", "top", "width", "height"
    extractor(
        noisy_path,
        region={"left": 0, "top": 0, "width": 0, "height": 0},
        user_supplied_mag=20.0,
    )

    # Should break with any other keys
    with pytest.raises(ValueError):
        extractor(
            noisy_path,
            region={"left": 0, "top": 0, "width": 0, "bob": 0},
            user_supplied_mag=20.0,
        )

    with pytest.raises(ValueError):
        extractor(
            noisy_path,
            region={"batman": 0, "top": 0, "width": 0, "height": 0},
            user_supplied_mag=20.0,
        )


def test_region_arg_value_types(delete_patch_dir):
    """Test the accepted value types of the region args."""
    extractor = PatchExtractor(top_dir=top_dir)

    # Should work with integers
    extractor(
        noisy_path,
        region={"left": 0, "top": 0, "width": 0, "height": 0},
        user_supplied_mag=20.0,
    )

    # Should break with any non-int keys
    with pytest.raises(TypeError):
        extractor(
            noisy_path,
            region={"left": 0.0, "top": 0.0, "width": 0.0, "height": 0.0},
            user_supplied_mag=20.0,
        )
    with pytest.raises(TypeError):
        extractor(
            noisy_path,
            region={"left": "0", "top": 0, "width": 0, "height": 0},
            user_supplied_mag=20.0,
        )


def test_region_arg_values(delete_patch_dir):
    """Test the values accepted by the region arg."""
    extractor = PatchExtractor(
        top_dir=top_dir,
        min_patch_mag=10.0,
        max_patch_mag=20.0,
    )

    # Should work zeros and positive integers
    # Should work with integers
    extractor(
        noisy_path,
        region={"left": 0, "top": 0, "width": 0, "height": 0},
        user_supplied_mag=20.0,
    )
    rmtree(top_dir)

    extractor(
        noisy_path,
        region={"left": 0, "top": 0, "width": 5000, "height": 5000},
        user_supplied_mag=20.0,
    )

    # Should break with negative coords
    with pytest.raises(ValueError):
        extractor(
            noisy_path,
            region={"left": 0, "top": 0, "width": -1, "height": 0},
            user_supplied_mag=20.0,
        )

    # Should break with zero width and non-zero height
    with pytest.raises(ValueError):
        extractor(
            noisy_path,
            region={"left": 0, "top": 0, "width": 0, "height": 5000},
            user_supplied_mag=20.0,
        )

    # Should break with non-zero width and zero height
    with pytest.raises(ValueError):
        extractor(
            noisy_path,
            region={"left": 0, "top": 0, "width": 5000, "height": 0},
            user_supplied_mag=20.0,
        )


def test_user_supplied_mag_type(delete_patch_dir):
    """Test the types accepted by the user_supplied_mag argument."""
    region = dict(zip(["left", "top", "width", "height"], 4 * [0]))
    extractor = PatchExtractor(
        top_dir=top_dir,
        min_patch_mag=10.0,
        max_patch_mag=20.0,
    )

    # Should work with floats
    extractor(noisy_path, region=region, user_supplied_mag=10.0)

    # Should break with non-floats
    with pytest.raises(TypeError):
        extractor(noisy_path, region=region, user_supplied_mag=10)
    with pytest.raises(TypeError):
        extractor(noisy_path, region=region, user_supplied_mag=10j)
    with pytest.raises(TypeError):
        extractor(noisy_path, region=region, user_supplied_mag="10")


def test_user_supplied_mag_values():
    """Test the values accepted by user_supplied_mag."""
    extractor = PatchExtractor(
        top_dir=top_dir,
        min_patch_mag=10.0,
        max_patch_mag=20.0,
    )

    # Should work with floats greater than 1.0
    extractor(noisy_path, user_supplied_mag=10.0)
    rmtree(top_dir)

    # Should break with floats less than 1.0
    with pytest.raises(ValueError):
        extractor(noisy_path, user_supplied_mag=0.99)
    with pytest.raises(ValueError):
        extractor(noisy_path, user_supplied_mag=0.0)
    with pytest.raises(ValueError):
        extractor(noisy_path, user_supplied_mag=-0.01)


def test_generate_patches_arg_types():
    """Test the types accepted by the generate patches arg."""
    extractor = PatchExtractor(
        top_dir=top_dir,
        min_patch_mag=10.0,
        max_patch_mag=20.0,
    )

    # Should work with bool
    extractor(noisy_path, user_supplied_mag=10.0, generate_patches=True)
    rmtree(top_dir)
    extractor(noisy_path, user_supplied_mag=10.0, generate_patches=False)
    rmtree(top_dir)

    # Should break with any non-bool
    with pytest.raises(TypeError):
        extractor(noisy_path, user_supplied_mag=10.0, generate_patches=1)
    with pytest.raises(TypeError):
        extractor(noisy_path, user_supplied_mag=10.0, generate_patches=1.0)
    with pytest.raises(TypeError):
        extractor(noisy_path, user_supplied_mag=10.0, generate_patches="True")


def test_with_foreground_less_than_background_tol():
    """Test with an image where no patches should survive thresholding."""
    extractor = PatchExtractor(
        top_dir=top_dir,
        min_patch_mag=1.0,
        max_patch_mag=1.0,
        overview_mag=1.0,
        background_tol=0.5,
    )

    with pytest.raises(RuntimeError):
        extractor(almost_blank_path, user_supplied_mag=1.0)

    rmtree(top_dir)
