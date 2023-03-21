"""Tests for `ImageDataset`."""
from pathlib import Path
import pytest
import torch
from torchvision.transforms import Compose


from lyzeum.torch_toolbelt.datasets import ImageDataset


@pytest.fixture()
def inputs_and_targets():
    """Return some mock inputs and targets to work with."""
    inputs = list(map(lambda x: str(x) + ".png", range(20)))
    targets = torch.ones(20, 3)
    return inputs, targets


def test_init_variable_assignment(inputs_and_targets):
    """Make sure instantiation variables are assigned correctly."""
    inputs, targets = inputs_and_targets

    i_tfm = Compose([(lambda x: x * 2)])
    t_tfm = Compose([(lambda x: x * 3)])

    dset = ImageDataset(
        inputs,
        i_tfm,
        targets=targets,
        target_transforms=t_tfm,
    )

    assert inputs == dset.inputs, "inputs not assigned correctly."
    assert (targets == dset.targets).all(), "targets not assigned correctly."
    assert i_tfm == dset._img_tf, "image transforms not assigned correctly."
    assert t_tfm == dset._trgt_tf, "target transforms not assigned correctly."


def test_input_and_target_length_checks(inputs_and_targets):
    """Test the checks for mismatching input and target lengths."""
    inputs, targets = inputs_and_targets

    # Should work if inputs and targets are the same length
    _ = ImageDataset(inputs, Compose([]), targets=targets)
    # Should work if no targets are suppplied
    _ = ImageDataset(inputs, Compose([]), targets=None)

    # Shoud fail if inputs and targets are different lengths
    with pytest.raises(RuntimeError):
        _ = ImageDataset(inputs[:5], Compose([]), targets=targets)
    with pytest.raises(RuntimeError):
        _ = ImageDataset(inputs, Compose([]), targets=targets[:5])


def test_input_types(inputs_and_targets):
    """Test the checks for catching unexpected input types."""
    inputs, _ = inputs_and_targets

    # Should work for list of str
    _ = ImageDataset(inputs, Compose([]))
    # Should work for list of Path
    _ = ImageDataset(list(map(Path, inputs)), Compose([]))

    # Should fail if inputs isn't a list
    with pytest.raises(TypeError):
        _ = ImageDataset(tuple(inputs), Compose([]))

    # Should fail with a list of items which are neither str nor Path
    with pytest.raises(TypeError):
        _ = ImageDataset([1, 2, 3, 4], Compose([]))
    with pytest.raises(TypeError):
        _ = ImageDataset([1.0, 2.0, 3.0], Compose([]))


def test_target_types(inputs_and_targets):
    """Test the checks for catching unexpected target types."""
    inputs, targets = inputs_and_targets

    # Should work for tensor, list of str, list of Path and None
    _ = ImageDataset(inputs, Compose([]), targets=targets)

    target_str = list(map(str, inputs))
    _ = ImageDataset(inputs, Compose([]), targets=target_str)

    target_paths = list(map(Path, inputs))
    _ = ImageDataset(inputs, Compose([]), targets=target_paths)

    target_paths = list(map(Path, inputs))
    _ = ImageDataset(inputs, Compose([]), targets=None)

    # Should fail if inputs isn't a list or a Tensor
    with pytest.raises(TypeError):
        _ = ImageDataset(inputs, Compose([]), targets=tuple(targets))

    # Should fail if inputs is a list with any non-str or non-path items
    with pytest.raises(TypeError):
        _ = ImageDataset(inputs, Compose([]), targets=list(range(20)))


def test_len_method(inputs_and_targets):
    """Test the len magic method."""
    inputs, targets = inputs_and_targets

    dset = ImageDataset(inputs[:3], Compose([]))
    assert len(dset) == 3
    dset = ImageDataset(inputs[:10], Compose([]))
    assert len(dset) == 10
    dset = ImageDataset(inputs[:11], Compose([]))
    assert len(dset) == 11


def test_iter(inputs_and_targets):
    """Test iterating through the dataset yields the correct results."""
    inputs, targets = inputs_and_targets

    # Should work with both inputs and targets supplied
    dset = ImageDataset(
        inputs, Compose([]), targets=targets, target_transforms=Compose([])
    )
    for idx, (inpt, trgt) in enumerate(dset):
        assert inputs[idx] == inpt
        assert (targets[idx] == trgt).all()

    # Should work with only inputs suppplied
    dset = ImageDataset(inputs, Compose([]))
    for idx, inpt in enumerate(dset):
        assert inputs[idx] == inpt


def test_image_transform_typecheck(inputs_and_targets):
    """Test the `image_transform` types accepted."""
    inputs, _ = inputs_and_targets

    # Should work with Compose Type
    _ = ImageDataset(inputs, Compose([]))

    # Should fail with any other type
    with pytest.raises(TypeError):
        _ = ImageDataset(inputs, 1)
    with pytest.raises(TypeError):
        _ = ImageDataset(inputs, torch.nn.Sequential())


def test_target_transform_typecheck(inputs_and_targets):
    """Test the `target_transform` types accepted."""
    inputs, _ = inputs_and_targets

    # Should work with Compose and None
    _ = ImageDataset(inputs, Compose([]), target_transforms=Compose([]))
    _ = ImageDataset(inputs, Compose([]), target_transforms=None)

    # Should fail with any other type
    with pytest.raises(TypeError):
        _ = ImageDataset(inputs, Compose([]), target_transforms=1)
    with pytest.raises(TypeError):
        _ = ImageDataset(
            inputs,
            Compose([]),
            target_transforms=torch.nn.Sequential(),
        )


def test_img_tfms_are_applied(inputs_and_targets):
    """Test the image transfroms are applied when getting item."""
    inputs, _ = inputs_and_targets
    img_tfms = Compose([lambda x: "applied"])

    dset = ImageDataset(inputs, img_tfms)
    for x_item in dset:
        assert x_item == "applied"


def test_target_tfms_are_applied(inputs_and_targets):
    """Test the target transfroms are applied when getting item."""
    inputs, targets = inputs_and_targets
    trgt_tfms = Compose([lambda x: "applied"])

    dset = ImageDataset(
        inputs,
        Compose([]),
        targets=targets,
        target_transforms=trgt_tfms,
    )
    for _, y_item in dset:
        assert y_item == "applied"
