"""Test the MILLoss class."""
import pytest

import torch
from torch import nn

from lyzeum.torch_toolbelt.loss import MILLoss

# pylint: disable=protected-access


def test_mil_loss_alpha_beta_types():
    """Test `MILLoss` with bad alpha and beta argument types."""
    with pytest.raises(TypeError):
        MILLoss(1.0, 1, nn.BCEWithLogitsLoss)
    with pytest.raises(TypeError):
        MILLoss(1, 0.0, nn.BCEWithLogitsLoss)
    with pytest.raises(TypeError):
        MILLoss("alpha", "beta", nn.BCEWithLogitsLoss)


def test_mil_loss_alpha_beta_values():
    """Test `MILLoss` with bad alpha and beta argument values."""
    loss = nn.BCEWithLogitsLoss
    with pytest.raises(ValueError):
        MILLoss(0, 1, nn.BCEWithLogitsLoss)
    with pytest.raises(ValueError):
        MILLoss(-1, 1, nn.BCEWithLogitsLoss)

    with pytest.raises(ValueError):
        MILLoss(0, 0, nn.BCEWithLogitsLoss)
    with pytest.raises(ValueError):
        MILLoss(-1, 0, nn.BCEWithLogitsLoss)
    with pytest.raises(ValueError):
        MILLoss(1, -1, nn.BCEWithLogitsLoss)

    MILLoss(1, 0, loss)
    MILLoss(1, 1, loss)


def test_mil_loss_check_tensor_dimensions_method():
    """Test the method `_check_tensor_dimensions` in `MILLoss`."""
    mil_loss = MILLoss(5, 0, nn.BCEWithLogitsLoss)

    mil_loss._check_tensor_dimensions(torch.zeros((10,)), 1)
    mil_loss._check_tensor_dimensions(torch.zeros((10, 5)), 2)
    mil_loss._check_tensor_dimensions(torch.zeros((10, 5, 2)), 3)
    mil_loss._check_tensor_dimensions(torch.zeros((10, 5, 3, 2)), 4)

    with pytest.raises(RuntimeError):
        mil_loss._check_tensor_dimensions(torch.zeros((10,)), 2)
    with pytest.raises(RuntimeError):
        mil_loss._check_tensor_dimensions(torch.zeros((10, 2)), 6)


def test_mil_loss_initialize_mask_and_labels_method_output_size():
    """Test output sizes of `_initialize_mask_and_labels` in `MILLoss`."""
    mil_loss = MILLoss(5, 0, nn.BCEWithLogitsLoss)

    preds = torch.rand(50, 2)
    mask, labels = mil_loss._initialize_mask_and_labels(preds)
    assert tuple(mask.shape) == (50, 1), "Unexpected mask shape"
    assert tuple(labels.shape) == (50, 2), "Unexpected labels shape"

    preds = torch.rand(100, 3)
    mask, labels = mil_loss._initialize_mask_and_labels(preds)
    assert tuple(mask.shape) == (100, 1), "Unexpected mask shape"
    assert tuple(labels.shape) == (100, 3), "Unexpected labels shape"

    preds = torch.rand(666, 666)
    mask, labels = mil_loss._initialize_mask_and_labels(preds)
    assert tuple(mask.shape) == (666, 1), "Unexpected mask shape"
    assert tuple(labels.shape) == (666, 666), "Unexpected labels shape"


def test_mil_loss_initialize_mask_and_labels_method_output_dtypes():
    """Test the output dtypes of `_get_mask_and_labels` method of `MILLoss`."""
    mil_loss = MILLoss(5, 0, nn.BCEWithLogitsLoss)

    for dtype in [int, float]:
        preds = torch.zeros((20, 2), dtype=dtype)
        mask, labels = mil_loss._initialize_mask_and_labels(preds)
        assert mask.dtype == preds.dtype, "Unexpected mask dtype."
        assert labels.dtype == preds.dtype, "Unexpected labels dtype."


def test_mil_loss_initialize_mask_and_labels_method_output_devices():
    """Test output device of `_get_mask_and_labels` method of MILLoss."""
    mil_loss = MILLoss(5, 0, nn.BCEWithLogitsLoss)

    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        preds = torch.zeros(10, device=device)
        mask, labels = mil_loss._initialize_mask_and_labels(preds)
        assert mask.device.type == device, "Unexpected mask device"
        assert labels.device.type == device, "Unexpected labels device"


def test_mil_loss_get_sorted_inds_method_input_dimension():
    """Test the method `_get_sorted_inds` in `MILLoss`."""
    mil_loss = MILLoss(1, 0, nn.BCEWithLogitsLoss)
    mil_loss._get_sorted_inds(torch.zeros(10))

    with pytest.raises(RuntimeError):
        mil_loss._get_sorted_inds(torch.zeros((10, 10)))
    with pytest.raises(RuntimeError):
        mil_loss._get_sorted_inds(torch.zeros((10, 10, 10)))
    with pytest.raises(RuntimeError):
        mil_loss._get_sorted_inds(torch.zeros((10, 1)))


def test_mil_loss_get_sorted_inds_method_return_values():
    """Test return values of `_get_sorted_inds` from `MILLoss`."""
    mil_loss = MILLoss(1, 0, nn.BCEWithLogitsLoss)

    tensor = torch.Tensor([10, 1, 20, 24])
    expected = torch.Tensor([3, 2, 0, 1])
    indices = mil_loss._get_sorted_inds(tensor, largest_first=True)
    check = (expected == indices).sum().item() == 4
    assert check is True, "_get_sorted_inds returning wrong result."

    tensor = torch.Tensor([10, 1, 20, 24])
    expected = torch.Tensor([1, 0, 2, 3])
    indices = mil_loss._get_sorted_inds(tensor, largest_first=False)
    check = (expected == indices).sum().item() == 4
    assert check is True, "_get_sorted_inds returning wrong result."


def test_mil_loss_get_labels_and_mask_output_values_with_bad_sizes():
    """Test the `get_labels_and_mask` method of `MILLoss` raises error.

    If the prediction dimension doesn't match the target dimensions we should
    raise an error.

    """
    mil_loss = MILLoss(1, 0, nn.BCEWithLogitsLoss)

    predictions = torch.rand(10, 3)
    target = torch.zeros(2)
    with pytest.raises(RuntimeError):
        mil_loss._get_labels_and_mask(predictions, target)

    predictions = torch.rand(10, 10)
    target = torch.zeros(11)
    with pytest.raises(RuntimeError):
        mil_loss._get_labels_and_mask(predictions, target)

    predictions = torch.rand(10, 3)
    target = torch.zeros(3)
    mil_loss._get_labels_and_mask(predictions, target)


def test_mil_loss_get_labels_and_mask_output_values_with_negative_bag():
    """Test output of `_get_labels_and_mask` method of `MILLoss`."""
    mil_loss = MILLoss(1, 0, nn.BCEWithLogitsLoss)

    preds = torch.zeros(10, 2)
    target = torch.Tensor([1, 0])
    mask, labels = mil_loss._get_labels_and_mask(preds, target)
    assert mask.sum() == len(mask), "Mask values are wrong"
    assert labels[:, 0].sum() == len(mask), "Missing positive labels."
    assert labels[:, 1:].sum() == 0, "Positive labels in the wrong place."


def test_mil_loss_get_labels_and_mask_output_values_with_positive_bag():
    """Test output of `_get_labels_and_bag` method of `MILLoss`."""
    alpha, beta = 5, 3

    mil_loss = MILLoss(alpha, beta, nn.BCEWithLogitsLoss)
    preds = torch.linspace(1, 20, 20).reshape(10, 2)
    target = torch.Tensor([0, 1])

    mask, labels = mil_loss._get_labels_and_mask(preds, target)

    # Test the mask is as expected:

    # The top alpha indices should equal 1
    assert (mask[:alpha] == 1).all(), "Unexpected alpha mask values."
    # The bottom beta indices should be 1
    assert (mask[-beta:] == 1).all(), "Unexpected beta mask values."
    # The other indices should be zero
    assert (mask[alpha:-beta] == 0).all(), "Unexpected mask values."

    # Test the labels are as expected:

    # Test the alpha part of the labels
    assert (labels[:alpha, 1] == 1).all(), "Unexpected alpha positive labels."
    assert (labels[:alpha, 0] == 0).all(), "Unexpected alpha negative labels."

    # Test the beta part of the labels
    assert (labels[-beta:, 1] == 0).all(), "Unexpected beta positive labels."
    assert (labels[-beta:, 0] == 1).all(), "Unexpected beta negative labels."

    # Test the empty part of the labels
    assert (labels[alpha:-beta, 1] == 0).all(), "Unexpected positive labels."
    assert (labels[alpha:-beta, 0] == 0).all(), "Unexpected negative labels."


def test_mil_loss_get_labels_and_mask_gradients_are_off():
    """Check `_get_labels_nad_mask` method of `MILLoss` yields no gradients.

    The tensors returned by `_get_labels_and_mask` should have gradients
    disabled, and therefore cannot contribute to backprop.

    """
    mil_loss = MILLoss(5, 0, nn.BCEWithLogitsLoss)
    preds = torch.rand((10, 2))
    target = torch.Tensor([0, 1])
    mask, labels = mil_loss._get_labels_and_mask(preds, target)

    assert mask.requires_grad is False, "Mask grads should not be enabled."
    assert labels.requires_grad is False, "Labels grads should not be enabled"


def test_mil_loss_weight_assignment():
    """Check we correctly assign weights to the loss function."""
    msg = "Loss weights have not been assigned properly."
    weight = torch.Tensor([1, 1])
    mil_loss = MILLoss(5, 0, nn.BCEWithLogitsLoss, loss_weights=weight)
    assert (mil_loss.loss_func.weight == weight).all().item(), msg

    weight = torch.Tensor([1, 2, 3, 4, 5])
    mil_loss = MILLoss(5, 0, nn.BCEWithLogitsLoss, loss_weights=weight)
    assert (mil_loss.loss_func.weight == weight).all().item(), msg

    weight = torch.Tensor([-2, 4, 5])
    mil_loss = MILLoss(5, 0, nn.BCEWithLogitsLoss, loss_weights=weight)
    assert (mil_loss.loss_func.weight == weight).all().item(), msg
