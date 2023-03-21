"""Helper functions for composing training and validation transforms."""
from torch.nn import Module
from torch import Tensor
from torchvision import transforms as tfms
from torchvision.transforms.functional import pad, center_crop

from numpy import sqrt, ceil


# from skimage.io import imread

from pystain import StainTransformer


class _ReflectivePadRotator(Module):
    """Image transform: reflectively pads, rotates and crops image.

    This avoids the weird empty bits of the images which come from randomly
    rotating it.

    """

    def __init__(self):
        """Build `_ReflectivePadRotator`."""
        super().__init__()
        self._rotator = tfms.RandomRotation(180.0)

    def forward(self, img: Tensor) -> Tensor:
        """Reflective pad, rotate and centre crop the image.

        Parameters
        ----------
        img : Tensor
            A mini-batch, or single image, as Tensor.

        Returns
        -------
        transformed : Tensor
            The augmented version of `img`.

        Raises
        ------
        AssertionError
            If the input batch, or image, is not square.

        """
        height, width = img.shape[-2], img.shape[-1]
        assert height == width, "Image shape should be square."

        padding = int(ceil(sqrt(2) * height)) - height
        transformed = pad(img, padding, padding_mode="reflect")
        transformed = self._rotator(transformed)
        transformed = center_crop(transformed, height)
        return transformed


def get_img_transforms(training: bool) -> tfms.Compose:
    """Return a composition of image transforms.

    Parameters
    ----------
    training : bool
        Are these transforms for a training or validation / inference loader?

    """
    transform_list = []

    transform_list.append(StainTransformer(jitter=False, normalise=True))
    # transform_list.append(imread)
    # transform_list.append(tfms.ToTensor())

    if training is True:
        transform_list.append(tfms.RandomHorizontalFlip())
        transform_list.append(tfms.RandomVerticalFlip())
        # transform_list.append(_ReflectivePadRotator())

    return tfms.Compose(transform_list)
