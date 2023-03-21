"""Tests for functions in `pystain.utils`."""
from pathlib import Path
import pytest

from skimage import io
import numpy as np
import torch
from torch import from_numpy, uint8, float32, float64, int32, int64

from pystain import utils

dummy_path = Path(".dummy_img.png")


@pytest.fixture(scope="session", autouse=True)
def dummy_img():
    """Create a dummy image for the tests."""
    rng = np.random.default_rng()
    img = rng.integers(256, size=(100, 100, 3), dtype=np.uint8)
    if dummy_path.exists():
        dummy_path.unlink()
    io.imsave(dummy_path, img)

    yield

    dummy_path.unlink()


def test_receive_img_with_forbidden_types():
    """Test the accepted input types for `utilts.receive_img`."""
    # Checks with forbidden types
    with pytest.raises(TypeError):
        utils.receive_img(123)
    with pytest.raises(TypeError):
        utils.receive_img([124])
    with pytest.raises(TypeError):
        utils.receive_img({"hello": "world"})


def test_recieve_img_with_allowed_types():
    """Test `utils.receive_img` with allowed types."""
    # The types str, Path, np.ndarray and torch.Tensor are allowed
    utils.receive_img(str(dummy_path))
    utils.receive_img(dummy_path)
    utils.receive_img(io.imread(dummy_path))
    utils.receive_img(from_numpy(io.imread(dummy_path)).permute(2, 0, 1))


def test_receive_img_with_ndarray_shapes():
    """Test `utils.receive_img` with numpy arrays with good and bad shapes.

    The correct shape for a rgb image as a numpy array is (H, W, C), where C=3
    is the number of colours channels and H and W are the height and width (in pixels).


    """
    img = io.imread(dummy_path)
    # Correct way: shape is (H, W, C)
    utils.receive_img(img)

    # Wrong: shape of (C, H, W)
    with pytest.raises(RuntimeError):
        utils.receive_img(np.transpose(img, (2, 0, 1)))
    # Wrong: shape of (H, C, W)
    with pytest.raises(RuntimeError):
        utils.receive_img(np.transpose(img, (0, 2, 1)))


def test_recieve_img_with_tensor_shapes():
    """Test `utils.receive_img` with torch Tensors with good and bad shapes.

    The correct shape for a rgb image as a torch Tensor is (C, H, W), where
    C=3 is the number of colours, and H and W are the height and width (in
    pixels).

    """
    img = from_numpy(io.imread(dummy_path)).permute(2, 0, 1)

    # Correct way: shape of (C, H, W)
    utils.receive_img(img)

    # Wrong: shape of (H, W, C)
    with pytest.raises(RuntimeError):
        utils.receive_img(img.permute(1, 2, 0))
    # Wrong: shape of (H, C, W)
    with pytest.raises(RuntimeError):
        utils.receive_img(img.permute(1, 0, 2))


def test_receive_image_ndarray_types():
    """Test `utils.receive_img` with numpy arrays with different dtypes."""
    img = io.imread(dummy_path)

    # Correct way: type np.uint8
    utils.receive_img(img.astype(np.uint8))

    # Wrong ways: other dtypes
    with pytest.raises(TypeError):
        utils.receive_img(img.astype(float))
    with pytest.raises(TypeError):
        utils.receive_img(img.astype(int))
    with pytest.raises(TypeError):
        utils.receive_img(img.astype(complex))
    with pytest.raises(TypeError):
        utils.receive_img(img.astype(np.int32))
    with pytest.raises(TypeError):
        utils.receive_img(img.astype(np.int64))
    with pytest.raises(TypeError):
        utils.receive_img(img.astype(np.float64))


def test_receive_image_tensor_types():
    """Test `utils.receive_img` with pytorch tensors with different dtypes."""
    img = from_numpy(io.imread(dummy_path)).permute(2, 0, 1)

    # Correct way: torch.uint8
    utils.receive_img(img.type(uint8))

    # Wrong ways: other dtypes
    with pytest.raises(TypeError):
        utils.receive_img(img.type(float32))
    with pytest.raises(TypeError):
        utils.receive_img(img.type(float64))
    with pytest.raises(TypeError):
        utils.receive_img(img.type(int32))
    with pytest.raises(TypeError):
        utils.receive_img(img.type(int64))


def test_rgb_to_od():
    """Test the function `utils.rgb_to_od` returns acceptable values."""
    img = from_numpy(io.imread(dummy_path)).permute(2, 0, 1)
    optical_density = utils.rgb_to_od(img)

    msg = f"OD values should be >= 1e-6, got {optical_density.min()}"
    assert (optical_density >= 1e-6).all(), msg

    img[:] = 255
    optical_density = utils.rgb_to_od(img)
    msg = "Ouput after passing max intensity has wrong value."
    assert (optical_density == 1e-6).all(), msg


def test_od_to_rgb():
    """Test function `utils.od_to_rgb` returns expected values."""
    optical_density = torch.zeros(3, 256, 256)
    rgb = utils.od_to_rgb(optical_density)

    msg = f"Passing zeros should return ones, got {torch.unique(rgb)}"
    assert (rgb == 1.0).all(), msg

    optical_density = torch.rand(3, 256, 256) * 1000
    rgb = utils.od_to_rgb(optical_density)

    msg = "Passing random noise should give output on [0, 1)."
    msg += f" Max and min are {rgb.min()}, {rgb.max()}"
    assert (rgb >= 0.0).all() and (rgb <= 1.0).all(), msg


def test_get_tissue_mask_catches_empty_masks():
    """Test function `utils.get_tissue_mask` catches empty mask."""
    img_rgb = torch.zeros(3, 256, 256).type(uint8)
    _ = utils.get_tissue_mask(img_rgb)

    img_rgb += 255
    with pytest.raises(utils.EmptyTissueMaskError):
        utils.get_tissue_mask(img_rgb)


def test_eigenvecs_point_correct_way():
    """Test function `utils.eigenvecs_point_correct_way`."""
    eig_vecs = torch.ones(2, 2)
    utils.eigenvecs_point_correct_way(eig_vecs)
    assert (eig_vecs == 1).all()

    eig_vecs[0, 0] = -1
    utils.eigenvecs_point_correct_way(eig_vecs)
    assert (eig_vecs[:, 1] == 1).all()
    assert eig_vecs[0, 0] == 1
    assert eig_vecs[1, 0] == -1

    eig_vecs[:] = 1
    eig_vecs[0, 1] = -1
    utils.eigenvecs_point_correct_way(eig_vecs)
    assert (eig_vecs[:, 0] == 1).all()
    assert eig_vecs[0, 1] == 1
    assert eig_vecs[1, 1] == -1


def test_h_and_e_in_right_order():
    """Test function `utils.h_and_e_in_right_order`."""
    vec_1 = torch.ones(3, 1)
    vec_2 = torch.zeros(3, 1)

    he_matrix = utils.h_and_e_in_right_order(vec_1, vec_2)

    tst = torch.cat((vec_1, vec_2), dim=1).T
    assert (he_matrix == tst).all()

    he_matrix = utils.h_and_e_in_right_order(vec_2, vec_1)
    assert (he_matrix == tst).all()
