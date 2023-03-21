#!/usr/bin/env python
"""Pytorch-based tool for transforming H&E characteristics of RGB images."""
import torch
from torch import Tensor


from pystain.macenko_extractor import MacenkoExtractor
from pystain.utils import (
    ImageInType,
    receive_img,
    od_to_rgb,
)


class StainTransformer:
    """Class for manipulating the stain characteristics of an image.

    Parameters
    ----------
    jitter : bool, optional
        Determines if the stain characterisitics are jittered before returning
        a new image. The default is False.
    jitter_strength : float, optional
        The "strength" with which the stain characteristics are jittered. The
        stain matrix is multiplied by random noise on
        [1.0 - `jitter_strength`, 1.0 + `jitter_strength`)
    normalise : bool, optional
        Controls outputs be normalised to target characteristics.

    Examples
    --------
    TODO: Add examples.


    """

    def __init__(
        self,
        normalise: bool = True,
        jitter: bool = False,
        jitter_strength: float = 0.2,
    ):
        """Construct the stain transformer class."""
        self.extractor = MacenkoExtractor()
        self._normalise = normalise
        self._jitter = jitter
        self._jitter_strength = jitter_strength

        _check_is_bool(self._normalise)
        _check_is_bool(self._jitter)
        self._check_jitter_strength()

    _target_he = Tensor(
        [
            [0.5144, 0.7946, 0.3224],
            [0.0869, 0.9494, 0.3019],
        ],
    )
    _max_target_concs = Tensor([2.4639, 1.1895])

    def _check_jitter_strength(self):
        """Check the `jitter_strength` is an acceptable type and value.

        Raises
        ------
        TypeError
            If the `jitter_strength` is not a float.
        ValueError
            If the `jitter_strength` argument is not on [0, 1].

        """
        if not isinstance(self._jitter_strength, float):
            msg = "Jitter strength should be float, got "
            msg += f"{type(self._jitter_strength)}."
            raise TypeError(msg)
        if not 0.0 <= self._jitter_strength <= 1.0:
            msg = "jitter_strength should be on [0, 1], got "
            msg += f"{self._jitter_strength}."
            raise ValueError(msg)

    @staticmethod
    def _concs_to_rgb(concs: Tensor, he_matrix: Tensor) -> Tensor:
        """Map from concentrations to rgb space.

        Parameters
        ----------
        concs : Tensor
            H and E concentrations.
        he_matrix : Tensor
            The estimated H and E matrix.

        Returns
        -------
        Tensor
            The image associated with `concs` in rgb space (on [0, 1]).

        """
        new_rgb = od_to_rgb(torch.matmul(he_matrix.T, concs))
        return torch.clip(new_rgb, 0, 1)

    def _uniform_rand(self) -> Tensor:
        """Return a Tensor of uniform random numbers.

        Returns
        -------
        Uniform random numbers on [1.0 - self.jitter, 1.0 + self.jitter).

        """
        lower = 1.0 - self._jitter_strength
        upper = 1.0 + self._jitter_strength
        return lower + (upper - lower) * torch.rand(2, 1)

    def fit(self, target_rgb: ImageInType) -> None:
        """Extract stain characteristics from `norm_target`.

        Parameters
        ----------
        target_rgb : str or Path or numpy.ndarray or Tensor
            An rgb image to be used as the new target in stain normalisation.

        """
        self._target_he, target_concs = self.extractor(target_rgb)
        self._max_target_concs[:] = torch.quantile(target_concs, 0.99, dim=1)

    def __call__(self, src_rgb: ImageInType):
        """Modify the stain characteristics of `src_img`.

        Parameters
        ----------
        src_rgb : str, Path, Tensor, ndarray

            An rgb image to be transformed. If `src_rgb`:

                — is a `str` or a `Path`, we load it using
                  `skimage.io.imread`.
                — is a `Tensor`, it should have dtype `torch.uint8` and be of
                  shape (C, H, W), where C=3 is the colour dimension and H and
                  W are the height and width.
                — is an `ndarray`, it should have dtype `np.uint8` and be of
                  shape (H, W, C), where C=3 is the colour dimension and H and
                  W are the height and width.

        Returns
        -------
        Tensor
            A transformed version of the input image in rgb space (on [0, 1]).
            The result may have been normalised and have had its stain
            jittered (depending on user selected options).

        """
        src_img = receive_img(src_rgb)

        src_he_matrix, src_concs = self.extractor(src_img)
        max_src_concs = torch.quantile(src_concs, 0.99, dim=1)

        if self._normalise is True:
            max_src_concs[max_src_concs == 0] = 1
            src_concs /= max_src_concs.unsqueeze(-1)
            src_concs *= self._max_target_concs.unsqueeze(-1)
            he_matrix = torch.clone(self._target_he)
        else:
            he_matrix = torch.clone(src_he_matrix)

        if self._jitter is True:
            he_matrix *= self._uniform_rand()

        new_rgb = self._concs_to_rgb(src_concs, he_matrix)
        return new_rgb.reshape(src_img.shape)


def _check_is_bool(arg: bool) -> None:
    """Check `arg` is of type bool.

    Raises
    ------
    TypeError
        If arg is not a bool.

    """
    if not isinstance(arg, bool):
        raise TypeError(f"arg should be of type boool, got f{arg}.")
