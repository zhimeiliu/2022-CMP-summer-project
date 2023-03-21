"""Tests for `MultiInstanceDataset`."""
from pathlib import Path
from shutil import rmtree, make_archive

import pytest

import numpy as np
from skimage.io import imsave, imread

import torch
from torchvision.transforms import Compose, ToTensor


from lyzeum.torch_toolbelt.datasets import MultiInstanceDataset


@pytest.fixture(scope="session", autouse=True)
def create_zipped_patch_dirs():
    """Create fake directory of zipped patches to test MIL model with."""
    patch_dir = Path(".patches/")
    img = np.random.randint(255, size=(256, 256, 3), dtype=np.uint8)

    for patient in range(10):
        directory = patch_dir / str(patient)
        directory.mkdir(parents=True)
        img_names = [directory / f"img_{idx}.png" for idx in range(10)]
        list(map(lambda x: imsave(x, img), img_names))
        make_archive(directory, "zip", directory)
        rmtree(directory)

    yield

    rmtree(patch_dir)


@pytest.fixture
def basic_args():
    """Return basic args required to instantiate `MultiInstanceDataset`."""
    inputs = list(Path(".patches").glob("**/*.zip"))
    targets = torch.randint(2, (len(inputs), 2))
    bag_size = 5
    x_tfms = Compose([imread, ToTensor(), lambda x: x * 0])
    return inputs, targets, bag_size, x_tfms


def test_argument_assignment(basic_args):
    """Check attributes are correctly assigned when arguments are passed."""
    inputs, targets, bag_size, x_tfms = basic_args
    data_set = MultiInstanceDataset(inputs, targets, bag_size, x_tfms)

    assert data_set.inputs == inputs
    assert (data_set.targets == targets).all()
    assert data_set.bag_size == bag_size
    assert data_set.x_tfms == x_tfms


def test_input_arg_types(basic_args):
    """Test the accepted and rejected input types."""
    inputs, targets, bag_size, x_tfms = basic_args

    # Should work when inputs is List[Union[str, Path]]
    # With Path:
    _ = MultiInstanceDataset(inputs, targets, bag_size, x_tfms)
    # With str:
    _ = MultiInstanceDataset(list(map(str, inputs)), targets, bag_size, x_tfms)

    # Should fail if inputs is not a list
    with pytest.raises(TypeError):
        _ = MultiInstanceDataset(tuple(inputs), targets, bag_size, x_tfms)

    # Should fail if the elements of inputs are not all zip files
    with pytest.raises(RuntimeError):
        _ = MultiInstanceDataset(
            inputs[:-1] + ["bob.txt"],
            targets,
            bag_size,
            x_tfms,
        )

    # Should fail with list of non-str/non-Path
    inputs = list(range(len(inputs)))
    with pytest.raises(TypeError):
        _ = MultiInstanceDataset(inputs, targets, bag_size, x_tfms)
    with pytest.raises(TypeError):
        _ = MultiInstanceDataset(
            list(map(float, inputs)),
            targets,
            bag_size,
            x_tfms,
        )


def test_target_arg_types(basic_args):
    """Test the accepted and rejected types of the `targets` argument."""
    inputs, targets, bag_size, x_tfms = basic_args

    # Should work with Tensors
    _ = MultiInstanceDataset(inputs, targets, bag_size, x_tfms)

    # Should fail with anything else
    with pytest.raises(TypeError):
        _ = MultiInstanceDataset(inputs, targets.numpy(), bag_size, x_tfms)
    with pytest.raises(TypeError):
        _ = MultiInstanceDataset(inputs, list(targets), bag_size, x_tfms)


def test_input_and_target_length_checks(basic_args):
    """Test the check for the input and target lengths matching."""
    inputs, targets, bag_size, x_tfms = basic_args

    # Should work with inputs and targets of same length
    _ = MultiInstanceDataset(inputs, targets, bag_size, x_tfms)

    # Should break if inputs and targets have different lengths
    with pytest.raises(RuntimeError):
        _ = MultiInstanceDataset(inputs[1:], targets, bag_size, x_tfms)
    with pytest.raises(RuntimeError):
        _ = MultiInstanceDataset(inputs, targets[1:], bag_size, x_tfms)


def test_len_method(basic_args):
    """Test the length of the dataset matches len of inputs and targets."""
    inputs, targets, bag_size, x_tfms = basic_args
    data_set = _ = MultiInstanceDataset(inputs, targets, bag_size, x_tfms)

    assert len(data_set) == len(inputs) == len(targets)


def test_x_tfms_are_applied(basic_args):
    """Test the `x_tfms` are applied."""
    inputs, targets, bag_size, x_tfms = basic_args
    data_set = MultiInstanceDataset(
        inputs,
        targets,
        bag_size,
        x_tfms,
    )

    for x_item, _ in data_set:
        assert (x_item == 0).all()


def test_one_iteration(basic_args):
    """Test one iteration of `MultiInstanceDataset`."""
    inputs, targets, bag_size, x_tfms = basic_args
    data_set = MultiInstanceDataset(inputs, targets, bag_size, x_tfms)

    for idx, (x_item, y_item) in enumerate(data_set):
        assert (y_item == targets[idx]).all()


def test_bag_size_warning(basic_args):
    """Check user is warned if patients have fewer patches than bag size."""
    inputs, targets, bag_size, x_tfms = basic_args

    # Should work with bag size smaller than length of inputs
    MultiInstanceDataset(inputs, targets, 1, x_tfms)

    with pytest.warns(UserWarning):
        data_set = MultiInstanceDataset(inputs, targets, 100, x_tfms)
