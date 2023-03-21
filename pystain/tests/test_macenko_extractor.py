"""Tests for callable object `pystain.macenko_extractor.MacenkoExtractor`."""
from pathlib import Path
import pytest

from skimage.io import imread

from torch import from_numpy, uint8, zeros, full

import pystain
from pystain import MacenkoExtractor

patch = (Path(pystain.__file__).parent / "../patches/patch-1.png").resolve()


def test_angular_percentile_arg():
    """Test the `angular_perctile` argument to `MacenkoExtractor`."""
    extractor = MacenkoExtractor(angular_percentile=0.5)
    assert extractor._angular_percentile == 0.5
    extractor = MacenkoExtractor(angular_percentile=0.123456)
    assert extractor._angular_percentile == 0.123456

    # Should break with non-floats
    with pytest.raises(TypeError):
        MacenkoExtractor(angular_percentile=1)
    with pytest.raises(TypeError):
        MacenkoExtractor(angular_percentile=0)
    with pytest.raises(TypeError):
        MacenkoExtractor(angular_percentile="0")

    # Should break with values outside [0, 1]
    with pytest.raises(ValueError):
        MacenkoExtractor(angular_percentile=1.1)
    with pytest.raises(ValueError):
        MacenkoExtractor(angular_percentile=-0.1)


def test_callable_arg_types():
    """Test the arguments `MacenkoExtractor` lets pass when called."""
    extractor = MacenkoExtractor()
    img_arr = imread(patch)
    img_tensor = from_numpy(img_arr).permute(2, 0, 1)

    # Should accept str, Path, np.ndarray and torch.Tensor
    _, _ = extractor(patch)
    _, _ = extractor(str(patch))
    _, _ = extractor(img_arr)
    _, _ = extractor(img_tensor)

    # Should reject any other type
    with pytest.raises(TypeError):
        extractor([patch, str(patch)])


def test_callable_arg_dtypes():
    """Test the dtypes of arrays and tensors `__call__` accepts."""
    extractor = MacenkoExtractor()
    img_arr = imread(patch)
    img_tensor = from_numpy(img_arr).permute(2, 0, 1)

    # Should accept ndarrays and Tensors with type uint8
    extractor(img_arr)
    extractor(img_tensor)

    # Should reject arrays and tensors with wrong dtypes
    with pytest.raises(TypeError):
        extractor(img_arr.astype(int))
    with pytest.raises(TypeError):
        extractor(img_arr.astype(float))

    with pytest.raises(TypeError):
        extractor(img_tensor.int())
    with pytest.raises(TypeError):
        extractor(img_tensor.float())


def test_callable_arg_shapes():
    """Test the shapes off arrays and tensor `__call__` accepts."""
    extractor = MacenkoExtractor()
    img_arr = imread(patch)
    img_tensor = from_numpy(img_arr).permute(2, 0, 1)

    # Should accept ndarrays with shape (H, W, C)
    extractor(img_arr)

    # Should accept Tensor of shape (C, H, W)
    extractor(img_tensor)

    # Should reject arrays with other shapes
    with pytest.raises(RuntimeError):
        extractor(img_arr.transpose(2, 0, 1))
    with pytest.raises(RuntimeError):
        extractor(img_arr.transpose(0, 2, 1))

    # Should reject Tensors with other shapes
    with pytest.raises(RuntimeError):
        extractor(img_tensor.permute(1, 2, 0))
    with pytest.raises(RuntimeError):
        extractor(img_tensor.permute(1, 0, 2))


def test_call_on_black_image():
    """Test `__call__` on a black image."""
    extractor = MacenkoExtractor()
    black_img = zeros(3, 256, 256, dtype=uint8)
    he_mat, concs = extractor(black_img)

    assert not he_mat.isnan().any()
    assert not concs.isnan().any()


def test_call_on_white_image():
    """Test `__call__` on a white image."""
    extractor = MacenkoExtractor()
    white_img = full((3, 256, 256), 255, dtype=uint8)
    he_mat, concs = extractor(white_img)

    assert (he_mat == 0).all()
    assert (concs == 0).all()
