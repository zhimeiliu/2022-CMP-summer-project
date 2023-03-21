"""Tests for `pystain.stain_transformer.StainTransformer`."""
from pathlib import Path

import pytest

from skimage.io import imread
from torch import from_numpy, zeros, uint8, full

import pystain
from pystain import StainTransformer

patch = (Path(pystain.__file__).parent / "../patches/patch-1.png").resolve()


def test_instantiation_args_are_set_correctly():
    """Test the instantiation arguments are set correctly."""
    transformer = StainTransformer(normalise=True)
    assert transformer._normalise == True
    transformer = StainTransformer(normalise=False)
    assert transformer._normalise == False

    transformer = StainTransformer(jitter=True)
    assert transformer._jitter == True
    transformer = StainTransformer(jitter=False)
    assert transformer._jitter == False

    transformer = StainTransformer(jitter_strength=0.1234)
    assert transformer._jitter_strength == 0.1234
    transformer = StainTransformer(jitter_strength=0.4321)
    assert transformer._jitter_strength == 0.4321


def test_types_of_normalise():
    """Test the jitter argument is checked correctly at instantiation."""
    # jitter argument should only accept bool.
    _ = StainTransformer(normalise=True)
    _ = StainTransformer(normalise=False)
    # jitter argument should reject any non-bool.
    with pytest.raises(TypeError):
        StainTransformer(normalise=1)
    with pytest.raises(TypeError):
        StainTransformer(normalise=1.0)
    with pytest.raises(TypeError):
        StainTransformer(normalise="True")


def test_types_of_jitter():
    """Test the jitter argument is checked correctly at instantiation."""
    # jitter argument should only accept bool.
    _ = StainTransformer(jitter=True)
    _ = StainTransformer(jitter=False)
    # jitter argument should reject any non-bool.
    with pytest.raises(TypeError):
        StainTransformer(jitter=1)
    with pytest.raises(TypeError):
        StainTransformer(jitter=1.0)
    with pytest.raises(TypeError):
        StainTransformer(jitter="True")


def test_types_values_of_jitter_strength():
    """Test jitter strength argument is checked correctly at instantiation."""
    # jitter strength argument should be a float
    _ = StainTransformer(jitter_strength=1.0)
    _ = StainTransformer(jitter_strength=0.5)
    _ = StainTransformer(jitter_strength=0.0)

    with pytest.raises(TypeError):
        StainTransformer(jitter_strength=1)
    with pytest.raises(TypeError):
        StainTransformer(jitter_strength=0)
    with pytest.raises(TypeError):
        StainTransformer(jitter_strength=1j)

    # Jitter strength should be on [0, 1]
    with pytest.raises(ValueError):
        StainTransformer(jitter_strength=-0.01)
    with pytest.raises(ValueError):
        StainTransformer(jitter_strength=-1.01)


def test_callable_arg_types():
    """Test arg types when `StainTransformer` is called."""
    transformer = StainTransformer()
    img_arr = imread(patch)
    img_tensor = from_numpy(img_arr).permute(2, 0, 1)

    # Should accept str, Path, np.ndarray and torch.Tensor
    _ = transformer(patch)
    _ = transformer(str(patch))
    _ = transformer(img_arr)
    _ = transformer(img_tensor)

    # Should reject any other type
    with pytest.raises(TypeError):
        transformer([patch, str(patch)])


def test_callable_arg_dtypes():
    """Test arg dtypes when `StainTransformer` gets called."""
    transformer = StainTransformer()
    img_arr = imread(patch)
    img_tensor = from_numpy(img_arr).permute(2, 0, 1)

    # Should reject arrays and tensors with wrong dtypes
    with pytest.raises(TypeError):
        transformer(img_arr.astype(int))
    with pytest.raises(TypeError):
        transformer(img_arr.astype(float))

    with pytest.raises(TypeError):
        transformer(img_tensor.int())
    with pytest.raises(TypeError):
        transformer(img_tensor.float())


def test_callable_arg_shapes():
    """Test the shapes allowed when `StainTransformer` gets called."""
    transformer = StainTransformer()
    img_arr = imread(patch)
    img_tensor = from_numpy(img_arr).permute(2, 0, 1)

    # Should accept arrays with shape (H, W, C), with C=3
    transformer(img_arr)

    # Should accept tensors with shape (C, H, W), with C=3.
    transformer(img_tensor)

    # Should reject arrays with other shapes
    with pytest.raises(RuntimeError):
        transformer(img_arr.transpose(2, 0, 1))
    with pytest.raises(RuntimeError):
        transformer(img_arr.transpose(0, 2, 1))

    # Should reject Tensors with other shapes
    with pytest.raises(RuntimeError):
        transformer(img_tensor.permute(1, 2, 0))
    with pytest.raises(RuntimeError):
        transformer(img_tensor.permute(1, 0, 2))


def test_call_with_black_image():
    """Test call works on a black image."""
    transformer = StainTransformer()
    black_img = zeros(3, 256, 256, dtype=uint8)

    tensor_img = transformer(black_img)

    assert not tensor_img.isnan().all()


def test_call_with_white_image():
    """Test call works on a white image."""
    transformer = StainTransformer()
    white_img = full((3, 256, 256), 255, dtype=uint8)

    # If we pass a white image, we should get one in return.
    tensor_img = transformer(white_img)
    assert (tensor_img == 1.0).all()
