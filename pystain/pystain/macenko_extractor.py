#!/usr/bin/env python
"""Callable Macenko stain extractor object.

Other useful implementations of Macenko's method include

    — https://github.com/Peter554/StainTools
    — https://github.com/TissueImageAnalytics/tiatoolbox
    — https://github.com/EIDOSlab/torchstain

which are based on the original paper:

    — [1] M. Macenko et al., "A method for normalizing histology slides for
    quantitative analysis," 2009 IEEE International Symposium on Biomedical
    Imaging: From Nano to Macro, 2009, pp. 1107-1110,
    doi: 10.1109/ISBI.2009.5193250.

"""
from typing import Tuple

import torch
from torch import Tensor


from pystain import utils


class MacenkoExtractor:
    """Mackenko stain extractor.

    Parameters
    ----------
    angular_percentile : float, optional
        Alpha parameter from Ref. [1]. Should be on (0, 1).

    """

    def __init__(self, angular_percentile: float = 0.99) -> None:
        """Construct macenko extractor."""
        self._angular_percentile = angular_percentile
        self._check_angular_percentile()

    def _check_angular_percentile(self) -> None:
        """Check the type and value of `_self.angular_percentile`.

        Raises
        ------
        TypeError
            If class argument `angular_percentile` is not a float.
        ValueError
            If the class argument `angular_percentile` is not on the interval
            [0, 1].

        """
        if not isinstance(self._angular_percentile, float):
            msg = "angular_percentile shoud be a foat, "
            msg += f"got {type(self._angular_percentile)}"
            raise TypeError(msg)
        if not 0 <= self._angular_percentile <= 1:
            msg = "angular_percentile should be on [0, 1], got "
            msg += f"{self._angular_percentile}"
            raise ValueError(msg)

    def _extract_he_matrix(self, od_vecs: Tensor) -> Tensor:
        """Extract the he_matrix from `img_in`.

        Parameters
        ----------
        od_vecs : Tensor
            The image in optical density space, of shape
            (3, num_pixels).

        Returns
        -------
        he_matrix : Tensor
            The H and E matrix estimated from `img_od`.

        """
        _, vectors = torch.linalg.eigh(torch.cov(od_vecs, correction=0))

        principle_eig_vecs = vectors[:, [2, 1]]
        utils.eigenvecs_point_correct_way(principle_eig_vecs)

        projected = torch.matmul(od_vecs.T, principle_eig_vecs).T

        phi = torch.atan2(projected[1, :], projected[0, :])

        min_phi = torch.quantile(phi, 1 - self._angular_percentile)
        max_phi = torch.quantile(phi, self._angular_percentile)

        vec_1 = torch.matmul(
            principle_eig_vecs,
            Tensor([torch.cos(min_phi), torch.sin(min_phi)]).unsqueeze(-1),
        )

        vec_2 = torch.matmul(
            principle_eig_vecs,
            Tensor([torch.cos(max_phi), torch.sin(max_phi)]).unsqueeze(-1),
        )

        he_matrix = utils.h_and_e_in_right_order(vec_1, vec_2)
        he_matrix /= torch.linalg.vector_norm(he_matrix, dim=1)[:, None]
        return he_matrix

    @staticmethod
    def _get_concentrations(img_od: Tensor, he_matrix: Tensor) -> Tensor:
        """Extact the stain concentrations from `img_od`.

        Parameters
        ----------
        img_od : Tensor
            The image in optical density space, with shape (3, num_pixels).
        he_matrix : Tensor
            The H & E matrix estimated from `img_od`.

        Returns
        -------
        Tensor
            The estimated H & E concentrations (of shape (2, num_pixels)).

        """
        lstsq_out = torch.linalg.lstsq(he_matrix.T, img_od, rcond=None)
        return lstsq_out.solution

    def _extract_stains(self, img_rgb: Tensor) -> Tuple[Tensor, Tensor]:
        """Extract the stain matrix and stain concentrations from `img_rgb`.

        Parameters
        ----------
        img_rgb : Tensor
            The image in RGB space (C, H, W).

        Returns
        -------
        he_matrix : Tensor
            The estimated H and E matrix.
        he_concentrations : Tensor
            The estimated concentrations at each pixel (Shape (2, N)).

        """
        img_od = utils.rgb_to_od(img_rgb).reshape(3, -1)

        try:
            mask = utils.get_tissue_mask(img_rgb).reshape(-1)
            he_matrix = self._extract_he_matrix(img_od[:, mask])
            he_concentrations = self._get_concentrations(img_od, he_matrix)
        except utils.EmptyTissueMaskError:
            he_matrix = torch.zeros(2, 3)
            he_concentrations = torch.zeros(2, img_od.shape[1])

        return he_matrix, he_concentrations

    def __call__(self, img_rgb: utils.ImageInType) -> Tuple[Tensor, Tensor]:
        """Calculate the HE matrix and concentrations from `img_in`.

        Parameters
        ----------
        img_rgb : str, Path, Tensor, ndarray

            If `img_rgb`:

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
        he_matrix : Tensor
            The H and E matrix estimated from `img_rgb`.
        he_concentrations : Tensor
            The H and E concentrations estimated from `img_rgb`.

        """
        img_rgb = utils.receive_img(img_rgb)
        he_matrix, he_concentrations = self._extract_stains(img_rgb)
        return he_matrix, he_concentrations
