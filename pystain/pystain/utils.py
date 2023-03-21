"""Utility functions for Stain manipulation with pytorch."""
from typing import Union
from pathlib import Path

import torch
from torch import Tensor, uint8, from_numpy, as_tensor

from numpy import ndarray
from skimage.io import imread


from skimage.color import rgb2lab


ImageInType = Union[str, Path, Tensor, ndarray]


def receive_img(img_in: ImageInType) -> Tensor:
    """Receive `img_in` and return a Tensor.

    Parameters
    ----------
    img_in : str, Path, Tensor, ndarray

        If `img_in`:

            — is a `str` or a `Path`, we load it using `skimage.io.imread`.
            — is a `Tensor`, it should have dtype `torch.uint8` and be of
              shape (C, H, W), where C=3 is the colour dimension and H and W
              are the height and width.
            — is an `ndarray`, it should have dtype `np.uint8` and be of shape
              (H, W, C), where C=3 is the colour dimension and H and W are the
              height and width.

    Returns
    -------
    img : Tensor
        The image as a `Tensor` with shape (C, H, W) and dtype torch.uint8.

    """
    if isinstance(img_in, (str, Path)):
        img = from_numpy(imread(str(img_in))).permute(2, 0, 1)
    elif isinstance(img_in, Tensor):
        img = img_in[:]
    elif isinstance(img_in, ndarray):
        _ndarray_rgb_shape_check(img_in)
        img = from_numpy(img_in).permute(2, 0, 1)
    else:
        msg = "img_in should be str, Path, ndarray or Tensor. "
        msg += f"Got {type(img_in)}"
        raise TypeError(msg)
    _is_uint8_tensor(img)
    _is_rgb_tensor(img)
    return img


def _ndarray_rgb_shape_check(img_array: ndarray):
    """Check the ndarray `img` has rgb shape.

    Parameters
    ----------
    img_array : ndarray
        An rgb image as numpy array of shape (H, W, C), where H and W are the
        height and width of the image and C=3 is the colour channel dimension.

    Raises
    ------
    RuntimeError
        If `img_array` does not have the shape of an rgb image.

    """
    if not (len((shape := img_array.shape)) == 3 and shape[-1] == 3):
        msg = f"ndarray image should have shape (H, W, 3), got {shape}."
        raise RuntimeError(msg)


def _is_uint8_tensor(tensor: Tensor):
    """Check `tensor` is of type `torch.uint8`.

    Parameters
    ----------
    tensor : Tensor
        An rgb image as a `torch.Tensor` of type `torch.uint8`.

    Raises
    ------
    TypeError
        If `tensor` is not of type `torch.uint8`.

    """
    if not (dtype := tensor.dtype) == uint8:
        raise TypeError(f"dtype should be uint8. Got shape {dtype}.")


def _is_rgb_tensor(tensor: Tensor):
    """Check `tensor` is rgb shape.

    Parameters
    ----------
    tensor : Tensor
        An rgb image as a `torch.Tensor`.

    Raises
    ------
    RuntimeError
        If `tensor` is not rgb-shaped (i.e. (C, H, W), where C=3).

    """
    if not (len((shape := tensor.shape)) == 3 and tensor.shape[0] == 3):
        msg = f"tensor shape should be CxHxW shape. Got {shape}"
        raise RuntimeError(msg)


def rgb_to_od(tensor_rgb: Tensor) -> Tensor:
    """Convert `tensor_rgb` to optical density.

    Parameters
    ----------
    tensor : Tensor
        An rgb image as a `torch.Tensor`.

    Returns
    -------
    Tensor
        `tensor_rgb` converted into units of optical density.

    """
    rgb_float = tensor_rgb.clip(1, 255).float() / 255.0
    return torch.maximum(-1.0 * torch.log(rgb_float), as_tensor(1e-6))


def od_to_rgb(img_od: Tensor):
    """Convert image from optical density space to RGB space.

    Parameters
    ----------
    img_od : Tensor
        Image in optical density space.

    Returns
    -------
    Tensor
        `img_od` converted to RGB space (on [0, 1]).

    """
    return torch.exp(-img_od)


class EmptyTissueMaskError(Exception):
    """Custom Exception to be used when an empty tissue mask is generated."""


def get_tissue_mask(img_rgb: Tensor, luminosity_thresh: float = 0.35):
    """Get a tissue mask.

    Parameters
    ----------
    img_rgb : Tensor
        RGB image.
    luminosity_thresh : float
        Luminsotiy threshold to distringuish between tissue (foreground) and
        background.

    Returns
    -------
    mask : Tensor
        A boolean mask giving the tissue (foreground) in `img_rgb`.

    """
    lab = from_numpy(rgb2lab(img_rgb.permute(1, 2, 0).numpy()))
    luminosity = lab[:, :, 0] / 255.0
    mask = luminosity < luminosity_thresh

    if mask.sum() == 0:
        raise EmptyTissueMaskError("Empty mask computed.")

    return mask


def eigenvecs_point_correct_way(eig_vecs: Tensor):
    """Make sure the eigenvectors point in the correct direction.

    Parameters
    ----------
    eig_vegs : Tensor
        Principal eigenvectors.

    """
    if eig_vecs[0, 0] < 0:
        eig_vecs[:, 0] *= -1
    if eig_vecs[0, 1] < 0:
        eig_vecs[:, 1] *= -1


def h_and_e_in_right_order(vec_1: Tensor, vec_2: Tensor):
    """Make sure the H and E vectors are in the right order.

    Parameters
    ----------
    vec_1 : Tensor
        Extracted stain vector.
    vec_2 : Tensor
        Extracted stain vector.

    Returns
    -------
    ordered_vectors : Tensor
        The input vectors in the correct orders.

    """
    if vec_1[0] > vec_2[0]:
        ordered_vectors = torch.cat((vec_1, vec_2), dim=1)
    else:
        ordered_vectors = torch.cat((vec_2, vec_1), dim=1)
    return ordered_vectors.T
