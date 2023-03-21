"""Function to house `FullyConnectedDataset`."""
from typing import Optional, Union, Tuple

from torch import Tensor
from torch.utils.data import Dataset


class FCDataset(Dataset):
    """Basic dataset for fully connected network experiments.

    Parameters
    ----------
    inputs : Tensor
        Inputs stacked along the row dimension.
    targets : Tensor, optional
        Targets (ground-truths) stacked along the row dimension, or `None` if
        you are doing inference and have no ground truths. If `targets` is
        `None`, `self.__get_item__` returns inputs only, rather than an
        input-target tuple.

    """

    def __init__(self, inputs: Tensor, targets: Optional[Tensor] = None) -> None:
        """Construct FullyConnectedDataset."""
        self._inputs_and_targets_type_check(inputs, targets)
        self.inputs = inputs.clone()
        self.targets = targets.clone() if targets is not None else None

        self._input_and_target_count_check()

    @staticmethod
    def _inputs_and_targets_type_check(
        inputs: Tensor, targets: Union[Tensor, None],
    ) -> None:
        """Check `self.inputs` and `self._targets` are of corect types.

        Parameters
        ----------
        inputs : Tensor
            The inputs, or x values, for the Dataset.
        targets : Union[Tensor, None]
            The targets, or y values, for the Dataset.

        Raises
        ------
        TypeError
            If `self.inputs` is not a `Tensor`.
        TypeError
            If `self.targets` is neither a `Tensor` or `None`.

        """
        if not isinstance(inputs, Tensor):
            msg = f"inputs should be of type Tensor. Got {type(inputs)}."
            raise TypeError(msg)
        if not isinstance(targets, (Tensor, type(None))):
            msg = "targets should be of type Tensor, or None  if you are "
            msg += "doing inference with no ground-truths. Got "
            msg += f"{type(targets)}."
            raise TypeError(msg)

    def _input_and_target_count_check(self) -> None:
        """Check the number of the inputs and targets match.

        Raises
        ------
        RuntimeError
            If `len(self.inputs)` does not match `len(self.targets)`.

        """
        if self.targets is None:
            return

        if not (x_len := len(self.inputs)) == (y_len := len(self.targets)):
            msg = "Number of inputs and targets should be the same. Got "
            msg += f"{x_len} inputs and {y_len} targets."
            raise RuntimeError(msg)

    def __len__(self) -> int:
        """Return the number of entries in the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """Return the input and target as position `idx`.

        Parameters
        ----------
        idx : int
            The index of the item to return

        Returns
        -------
        Union[Tuple[Tensor, Tensor], Tensor]
            A tuple of `(self.inputs[idx], self.targets[idx])` or, if
            `self.targets is None`, `self.inputs[idx]`.

        """
        if self.targets is not None:
            return self.inputs[idx].clone(), self.targets[idx].clone()
        return self.inputs[idx].clone()
