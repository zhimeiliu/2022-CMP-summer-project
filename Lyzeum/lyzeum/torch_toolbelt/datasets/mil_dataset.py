"""Custom pytorch datasets."""
from typing import Tuple, Union, List
import warnings
from pathlib import Path
from collections import Counter
from zipfile import is_zipfile

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from torchvision.transforms import Compose

from lyzeum.misc import list_zipfile_contents


# pylint: disable=too-few-public-methods


class MultiInstanceDataset(Dataset):
    """Multiple instance learning dataset.

    Parameters
    ----------
    inputs : List[Union[str, Path]]
        List of paths tro zip files. Each zip files should contain the patches
        drawn from each input.
    targets : Tensor
        A `Tensor` holding the ground truths for each of the elements in
        `inputs`. Note, `len(inputs)` should equal `len(targets)`.
    bag_size : int
        Number of patches to randomly sample from each item in `inputs`.
    x_tfms : Compose
        Transforms to apply to each patch when it is sampled.

    """

    def __init__(
        self,
        inputs: List[Union[Path, str]],
        targets: List[Tensor],
        bag_size: int,
        x_tfms: Compose,
    ) -> None:
        """Construct the dataset class."""
        super().__init__()
        self.inputs = inputs
        self.targets = targets
        self.bag_size = bag_size
        self.x_tfms = x_tfms

        self._check_input_types()
        self._check_target_types()
        self._check_transform_type()
        self._check_input_and_target_lengths_match()
        self._bag_size_check()

    def _check_input_types(self):
        """Type check `self.inputs`.

        Raises
        ------
        TypeError
            If `self.inputs` is not a list.
        TypeError
            If the items in `self.inputs` are not path like.
        RuntimeError
            If not all items in  `self.inputs` are zip files.

        """
        if not isinstance(self.inputs, list):
            msg = f"Inputs should be a list. Got {type(self.inputs)}."
            raise TypeError(msg)

        if not all(map(lambda x: isinstance(x, (str, Path)), self.inputs)):
            unique = list(Counter(map(type, self.inputs)).keys())
            msg = "inputs should be a list of path-like. Got list containing: "
            msg += f"{unique}"
            raise TypeError(msg)

        if not all(map(is_zipfile, self.inputs)):
            msg = "Not all inputs are zip files. They should be"
            raise RuntimeError(msg)

    def _check_target_types(self):
        """Make sure `self.targets` is an accepted type.

        Raises
        ------
        TypeError
            If `self.targets` is not a Tensor.

        """
        if not isinstance(self.targets, Tensor):
            msg = f"Targets should be a Tensor. got {type(self.targets)}."
            raise TypeError(msg)

    def _check_transform_type(self):
        """Check `self.x_tfms` has type `Compose`.

        Raises
        ------
        TypeError
            If `self.x_tfms` is not an instance of Compose.

        """
        if not isinstance(self.x_tfms, Compose):
            msg = f"x_tfms should have type Compose. Got {type(self.x_tfms)}"
            raise TypeError(msg)

    def _check_input_and_target_lengths_match(self):
        """Check the sizes of the inputs and the targets match.

        Raises
        ------
        RuntimeError
            If the number of inputs does not match the number of targets.

        """
        if not (x_len := len(self.inputs)) == (y_len := len(self.targets)):
            msg = "Number of inputs and targets should be equal. "
            msg += f"Got {x_len} inputs and {y_len} targets."
            raise RuntimeError(msg)

    def _bag_size_check(self) -> None:
        """Warn if any patients have fewer inputs than the bag size."""
        patch_checks = list(
            map(
                lambda x: (len(list_zipfile_contents(x)) < self.bag_size),
                self.inputs,
            )
        )
        low_count = np.sum(patch_checks)

        if low_count != 0:
            msg = (
                f"{low_count} scans have fewer tiles than {self.bag_size}"
                + " tiles, which is the bag size."
            )
            warnings.warn(msg, UserWarning)

    def __len__(self) -> int:
        """Return the number of inputs in the data set.

        Returns
        -------
        int
            Number of patients in the dataset.

        """
        return len(self.inputs)

    def show_bag(self, idx: int) -> None:
        """Display a batch on-screen using matplotlib."""
        square_size = int(np.ceil(np.sqrt(self.bag_size)))
        bag, label = self.__getitem__(idx)
        bag_arr = np.transpose(bag.numpy(), (0, 2, 3, 1))

        fig, axes = plt.subplots(square_size, square_size, figsize=(7, 8))
        fig.subplots_adjust(left=0, right=1, bottom=0, hspace=0, wspace=0)
        for tile, axis in zip(bag_arr, axes.ravel()):
            axis.imshow(tile)
        for axis in axes.ravel():
            axis.set_xticks([])
            axis.set_yticks([])
        fig.suptitle(f"{label}")
        plt.show()

    def _get_single_patient_items(self, idx: int) -> Tuple[List[str], Tensor]:
        """Return the list of tile paths for a single patient.

        Parameters
        ----------
        idx : int
            Index of the input and target to be drawn.

        Returns
        -------
        List[str]
            List of the image paths inside the zip file.
        Tensor
            The n-hot-encoded target.

        """
        zip_path = self.inputs[idx]
        target = self.targets[idx]
        return list_zipfile_contents(zip_path), target

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Return a single bag of images.

        Parameters
        ----------
        idx : int
            Index of the item we wish to take the item from.

        Returns
        -------
        Tensor
            Tuple of tensors holding the input and label. The tuple contains
            tiles_pp randomly choosen tiles from a given patient, chosen
            with repetition.
        label : Tensor
            Binary truth: one-hot-encoding style.

        """
        all_scan_patches, patient_label = self._get_single_patient_items(idx)

        # sample_inds = torch.randperm(len(all_scan_patches))[: self.bag_size]
        # bag_paths = [patient_paths[choice_idx] for choice_idx in sample_inds]
        sample_inds = torch.randint(len(all_scan_patches), (self.bag_size,))
        bag_patches = map(lambda x: all_scan_patches[x], sample_inds)

        bag = torch.stack(list(map(self.x_tfms, bag_patches)), dim=0)
        return bag, patient_label
