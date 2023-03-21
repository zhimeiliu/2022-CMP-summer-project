"""Tests for custom pytorch datasets."""
import pytest

import torch

from lyzeum.torch_toolbelt.datasets import FCDataset

# pylint: disable=redefined-outer-name


@pytest.fixture()
def inputs_and_targets():
    """Return the inputs and targets to test with."""
    return torch.rand(10, 4), torch.rand(10, 3)


def test_input_and_target_assignment(inputs_and_targets):
    """Test the inputs and targets are correctly assigned."""
    inputs, targets = inputs_and_targets
    dset = FCDataset(inputs, targets)

    assert (dset.inputs == inputs).all()
    assert (dset.targets == targets).all()


def test_input_types(inputs_and_targets):
    """Test the accepted types of inputs."""
    inputs, _ = inputs_and_targets

    # Should work with Tensors
    _ = FCDataset(inputs)

    # Should fail with anything else
    with pytest.raises(TypeError):
        _ = FCDataset(list(inputs))
    with pytest.raises(TypeError):
        _ = FCDataset(tuple(inputs))
    with pytest.raises(TypeError):
        _ = FCDataset(inputs.numpy())


def test_target_types(inputs_and_targets):
    """Test the accepted types of targets."""
    inputs, targets = inputs_and_targets

    # Should work with Tensors
    _ = FCDataset(inputs, targets=targets)
    # Should work with None
    _ = FCDataset(inputs, targets=None)

    # Should fail with anything else
    with pytest.raises(TypeError):
        _ = FCDataset(inputs, targets=list(targets))
    with pytest.raises(TypeError):
        _ = FCDataset(inputs, targets=tuple(targets))
    with pytest.raises(TypeError):
        _ = FCDataset(inputs, targets=targets.numpy())


def test_input_and_target_length_check(inputs_and_targets):
    """Test the inputs and target length checks."""
    inputs, targets = inputs_and_targets

    # Should work with matching lengths
    _ = FCDataset(inputs, targets)

    # Should break with mismatched lengths
    with pytest.raises(RuntimeError):
        FCDataset(inputs[1:], targets)
    with pytest.raises(RuntimeError):
        FCDataset(inputs, targets[1:])


def test_len_method(inputs_and_targets):
    """Test the len magic method of `FullyConnectedDataset`."""
    inputs, targets = inputs_and_targets
    dset = FCDataset(inputs, targets)
    assert len(dset) == len(inputs) == len(targets)


def test_get_item_method(inputs_and_targets):
    """Test the get item method of `FullyConnectedDataset`."""
    inputs, targets = inputs_and_targets

    dset = FCDataset(inputs, targets)

    for idx, (x_item, y_item) in enumerate(dset):
        assert (x_item == inputs[idx]).all()
        assert (y_item == targets[idx]).all()


def test_inference_mode_return(inputs_and_targets):
    """Test `FullConnectedDataset` returns a single item in inference mode."""
    inputs, _ = inputs_and_targets
    dset = FCDataset(inputs, targets=None)
    assert dset.targets is None

    for x_item in dset:
        assert isinstance(x_item, torch.Tensor)
