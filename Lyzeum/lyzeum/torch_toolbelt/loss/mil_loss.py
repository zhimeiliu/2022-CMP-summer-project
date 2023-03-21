"""Loss function for the multiple instance learning scheme."""
from typing import Tuple, Optional


import torch
from torch import nn


class MILLoss(nn.Module):
    """Custom loss function for the multiple instance learning scheme.

    Parameters
    ----------
    alpha : int
        Top k of predictions in a positive bag to label as positive.
    beta : int
        Bottom k of patients in a positive bag to label as negative.
    loss_func : nn.Module
        Pytorch loss function.
    loss_weights : torch.Tensor, optional
        Weights to use in the loss function. Should have the same length as the
        number of classes.

    """

    def __init__(
        self,
        alpha: int,
        beta: int,
        loss_func: nn.Module,
        loss_weights: Optional[torch.Tensor] = None,
    ):
        """Build MILLoss."""
        super().__init__()
        self._alpha = alpha
        self._beta = beta
        self.loss_func = loss_func(reduction="none", weight=loss_weights)
        self._check_alpha_beta_arguments()

    def _check_alpha_beta_arguments(self):
        """Check `alpha` and `beta` are allowed values.

        Raises
        ------
        TypeError
            If `self._alpha` and `self._beta` are not both integers, raise.
        ValueError
            If `self._alpha` is not greater than zero, raise.
        ValueError
            If `self._beta` is not zero or more, raise.

        """
        if not (isinstance(self._alpha, int) and isinstance(self._beta, int)):
            msg = "(alpha, beta should be (int, int), got "
            msg += f"{(type(self._alpha), type(self._beta))})"
            raise TypeError(msg)

        if self._alpha <= 0:
            raise ValueError(f"Alpha must not be less than zero: {self._alpha}")
        if self._beta < 0:
            raise ValueError(f"Beta must be positive, got {self._beta}")

    @staticmethod
    def _check_tensor_dimensions(tensor: torch.Tensor, num_dims: int):
        """Check the dimensionality of `tensor` is as expected.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor whose dimensionality we wish to test.

        Raises
        ------
        RuntimeError
            If `tensor` is not of the correct dimensionality, raise.

        """
        if len(tensor.shape) != num_dims:
            msg = f"Tensor should have {num_dims} dims, not "
            msg += f"{len(tensor.shape)}"
            raise RuntimeError(msg)

    @staticmethod
    def _initialize_mask_and_labels(
        predictions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create Tensors to serve as the mask and proxy labels.

        Parameters
        ----------
        predictions : torch.Tensor
            The predictions from the network.

        Returns
        -------
        mask : torch.Tensor
            A mask determining which items are used in the loss computation.
        proxy_labels : torch.Tensor
           Tensor to hold the proxy labels we calculate based on the
           predictions.

        """
        mask = torch.zeros(
            (len(predictions), 1),
            device=predictions.device,
            dtype=predictions.dtype,
        )
        labels = torch.zeros(
            predictions.shape,
            device=predictions.device,
            dtype=predictions.dtype,
        )
        return mask, labels

    def _get_sorted_inds(
        self,
        tensor: torch.Tensor,
        largest_first: bool = False,
    ) -> torch.Tensor:
        """Return the indices of the values in `tensor` in sorted order.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor we wish to get the sorted indices from.
        largest_first : bool, optional
            Do we want the indices sorted by largest-first (descending order)
            or smallest-first (ascending order). This is the `largest`
            argument from `torch.topk`.

        Returns
        -------
        indices : torch.Tensor
            Indices of the

        """
        self._check_tensor_dimensions(tensor, 1)
        _, indices = torch.topk(
            tensor,
            k=len(tensor),
            largest=largest_first,
            sorted=True,
        )
        return indices

    @staticmethod
    def _check_prediction_and_target_sizes_match(
        predictions: torch.Tensor, target: torch.Tensor
    ):
        """Check the sizes of `predictions` and `target` match.

        Parameters
        ----------
        predictions : torch.Tensor
            Tensor holding the predictions.
        target : torch.Tensor
            One-dimensional tensor holding the target.

        Raises
        ------
        RuntimeError
            If the number of columns in `predictions` does not equal the
            length of `target`, raise.

        """
        if predictions.shape[1] != len(target):
            msg = f"Predictions has {predictions.shape} columns but "
            msg += f"targets has {len(target)}. These should match."
            raise RuntimeError(msg)

    @torch.no_grad()
    def _get_labels_and_mask(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the proxy labels and mask.

        Parameters
        ----------
        prediction : torch.Tensor
            The predictions from the current bag.
        target : torch.Tensor
            The bag label associated with patient from whom the bag came. The
            options are the vectors [1, 0] and [0, 1].

        Returns
        -------
        mask : torch.Tensor
            A masking tensor which provides the weights in use in the loss
            calculation. Since we dynamically select which bag elements
            contribute to the gradient flow, some items need to have zero
            weight.
        labels : torch.Tensor
            The proxy labels we calculate based on the bag.

        Notes
        -----
        We calculate the proxy labels based on the bag_label and predictions.
        If the patient is normal (negative), the entire bag receives the proxy
        label [1, 0]. If the patient is diseased (positive), the alpha % of the
        bag with the highest prediction receives the proxy label [0, 1] and
        the beta % of the bag with the lowest predictions recieve the proxy
        label [1, 0]. All other items in a positive bag recieve a weight of
        zero when computing the loss.

        """
        self._check_tensor_dimensions(target, 1)
        self._check_tensor_dimensions(predictions, 2)
        self._check_prediction_and_target_sizes_match(predictions, target)

        argmax_label = int(target.argmax().item())
        mask, labels = self._initialize_mask_and_labels(predictions)

        if argmax_label == 0:
            labels[:, argmax_label] = 1
            mask[:] = 1
        else:

            indices = self._get_sorted_inds(
                predictions[:, 0],
                largest_first=False,
            )

            labels[indices[: self._alpha], argmax_label] = 1
            mask[indices[: self._alpha]] = 1

            if self._beta > 0:
                labels[indices[-self._beta :], 0] = 1
                mask[indices[-self._beta :]] = 1

        return mask, labels

    def forward(
        self, prediction: torch.Tensor, bag_label: torch.Tensor
    ) -> torch.Tensor:
        """Compute the MIL loss.

        Parameters
        ----------
        prediction : torch.Tensor
            The predictions from the current bag.
        bag_label: torch.Tensor
            The label associated with the bag.

        Returns
        -------
        loss : torch.Tensor
            The loss from the items in the bag we allowed into backpropagation.

        """
        mask, labels = self._get_labels_and_mask(prediction, bag_label)
        loss = (self.loss_func(prediction, labels) * mask).sum() / (mask.sum())
        return loss
