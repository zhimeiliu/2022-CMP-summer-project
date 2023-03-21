"""Tests for `ClassifierFC`."""
import pytest

from torch import nn

from lyzeum.torch_toolbelt.models import ClassifierFC


def test_layer_sizes_type():
    """Test the argument `layer_sizes` onlyy accepts lists."""
    # Should work with list
    _ = ClassifierFC([1, 2, 3])

    # Should fail with anything else
    with pytest.raises(TypeError):
        _ = ClassifierFC((10, 2, 3, 4, 5))
    with pytest.raises(TypeError):
        _ = ClassifierFC({10, 2, 3, 4, 5})
    with pytest.raises(TypeError):
        _ = ClassifierFC("1010")


def test_layer_sizes_entry_types():
    """Test the accepted types in the list `layer_sizes`."""
    # Should work with int
    _ = ClassifierFC([10, 5, 2])

    # Should fail with anything else
    with pytest.raises(TypeError):
        _ = ClassifierFC([10.0, 2.0])
    with pytest.raises(TypeError):
        _ = ClassifierFC([10, 2.0])
    with pytest.raises(TypeError):
        _ = ClassifierFC([10, 2, 2.0])


def test_layer_sizes_entry_values():
    """Test the accepted values in `layer_sizes`."""
    # Should work with int greater than or equal to 1.
    _ = ClassifierFC([10, 5, 1])

    # Should break with ints less than 1
    with pytest.raises(ValueError):
        _ = ClassifierFC([10, 5, 0, 2])
    with pytest.raises(ValueError):
        _ = ClassifierFC([-1, 5, 2])
    with pytest.raises(ValueError):
        _ = ClassifierFC([10, 5, -2])


def test_input_batchnorm_argument_types():
    """Test the argument `input_batchnorm` only accepts bools."""
    # Should work for bool
    _ = ClassifierFC([10, 10, 10], input_batchnorm=True)
    _ = ClassifierFC([10, 10, 10], input_batchnorm=True)

    # Should fail for anything else
    with pytest.raises(TypeError):
        _ = ClassifierFC([10, 10, 10], input_batchnorm=10)
    with pytest.raises(TypeError):
        _ = ClassifierFC([10, 10, 10], input_batchnorm=1.0)
    with pytest.raises(TypeError):
        _ = ClassifierFC([10, 10, 10], input_batchnorm="True")


def test_hidden_batchnorm_argument_types():
    """Test the argument `hidden_batchnorm` only accepts bools."""
    # Should work with bool
    _ = ClassifierFC([10, 10, 2], hidden_batchnorm=True)
    _ = ClassifierFC([10, 10, 2], hidden_batchnorm=False)

    # Should fail for anything else
    with pytest.raises(TypeError):
        _ = ClassifierFC([10, 10, 2], hidden_batchnorm=1.0)
    with pytest.raises(TypeError):
        _ = ClassifierFC([10, 10, 2], hidden_batchnorm=1)
    with pytest.raises(TypeError):
        _ = ClassifierFC([10, 10, 2], hidden_batchnorm="False")


def test_input_drouput_argument_types():
    """Test the argument `input_dropout` only accepts floats."""
    # Should work with float
    _ = ClassifierFC([10, 1], input_dropout=0.0)
    _ = ClassifierFC([10, 1], input_dropout=0.5)
    _ = ClassifierFC([10, 1], input_dropout=1.0)

    # Should otherwise fail
    with pytest.raises(TypeError):
        _ = ClassifierFC([10, 1], input_dropout=0)
    with pytest.raises(TypeError):
        _ = ClassifierFC([10, 1], input_dropout=0j)
    with pytest.raises(TypeError):
        _ = ClassifierFC([10, 1], input_dropout="0")


def test_input_dropout_argument_values():
    """Test the argument `input_dropout` takes the correct values only."""
    # Should work on [0, 1]
    _ = ClassifierFC([10, 2], input_dropout=0.0)
    _ = ClassifierFC([10, 2], input_dropout=0.5)
    _ = ClassifierFC([10, 2], input_dropout=1.0)

    # Should otherwise fail
    with pytest.raises(ValueError):
        _ = ClassifierFC([10, 2], input_dropout=-0.00001)
    with pytest.raises(ValueError):
        _ = ClassifierFC([10, 2], input_dropout=1.00001)


def test_hidden_drouput_argument_types():
    """Test the argument `hidden_dropout` only accepts floats."""
    # Should work with float
    _ = ClassifierFC([10, 10, 1], hidden_dropout=0.0)
    _ = ClassifierFC([10, 10, 1], hidden_dropout=0.5)
    _ = ClassifierFC([10, 10, 1], hidden_dropout=1.0)

    # Should otherwise fail
    with pytest.raises(TypeError):
        _ = ClassifierFC([10, 10, 1], hidden_dropout=0)
    with pytest.raises(TypeError):
        _ = ClassifierFC([10, 10, 1], hidden_dropout=0j)
    with pytest.raises(TypeError):
        _ = ClassifierFC([10, 10, 1], hidden_dropout="0")


def test_hidden_dropout_argument_values():
    """Test the argument `hidden_dropout` takes the correct values only."""
    # Should work on [0, 1]
    _ = ClassifierFC([10, 10, 2], hidden_dropout=0.0)
    _ = ClassifierFC([10, 10, 2], hidden_dropout=0.5)
    _ = ClassifierFC([10, 10, 2], hidden_dropout=1.0)

    # Should otherwise fail
    with pytest.raises(ValueError):
        _ = ClassifierFC([10, 10, 2], hidden_dropout=-0.00001)
    with pytest.raises(ValueError):
        _ = ClassifierFC([10, 10, 2], hidden_dropout=1.00001)


def test_model_with_no_batchnorm_no_dropout():
    """Test the contents of the model are correct."""
    layer_sizes = [20, 10, 5, 2]

    clf = ClassifierFC(
        layer_sizes,
        input_batchnorm=False,
        input_dropout=0.0,
        hidden_dropout=0.0,
        hidden_batchnorm=False,
    )

    assert len(clf.fwd) == 3
    for block in clf.fwd.children():
        layers = list(block.fwd.children())
        assert isinstance(layers[0], nn.Linear)


def test_model_with_input_batchnorm_no_dropout():
    """Test the contents of the model with input_batchnorm."""
    clf = ClassifierFC(
        [10, 10, 2],
        input_batchnorm=True,
        input_dropout=0.0,
        hidden_batchnorm=False,
        hidden_dropout=0.0,
    )

    first_block, other_blocks = clf.fwd.Block1, list(clf.fwd.children())[1:]

    # The first block should have a batchnorm
    assert isinstance(first_block.fwd[0], nn.BatchNorm1d)
    # The other blocks should not
    for block in other_blocks:
        for layer in block.fwd.children():
            assert not isinstance(layer, nn.BatchNorm1d)


def test_model_no_input_batchnorm_with_dropout():
    """Test the contents of the model with input_dropout."""
    clf = ClassifierFC(
        [10, 10, 2],
        input_batchnorm=False,
        input_dropout=0.1234321,
        hidden_batchnorm=False,
        hidden_dropout=0.0,
    )

    first_block, other_blocks = clf.fwd.Block1, list(clf.fwd.children())[1:]

    # The first block should have a dropout
    assert isinstance(first_block.fwd[0], nn.Dropout)
    assert first_block.fwd[0].p == 0.1234321
    # The other blocks should not
    for block in other_blocks:
        for layer in block.fwd.children():
            assert not isinstance(layer, nn.Dropout)


def test_model_with_input_batchnorm_and_dropout():
    """Test the contents of the model with input batchnorm and dropout."""
    clf = ClassifierFC(
        [10, 10, 2],
        input_batchnorm=True,
        input_dropout=0.123,
        hidden_batchnorm=False,
        hidden_dropout=0.0,
    )

    first_block, other_blocks = clf.fwd.Block1, list(clf.fwd.children())[1:]

    # The first block should have both batchnorm and dropout
    assert isinstance(first_block.fwd[0], nn.BatchNorm1d)
    assert isinstance(first_block.fwd[1], nn.Dropout)
    assert first_block.fwd[1].p == 0.123

    # The other blocks should have neither batchnorm nor dropout
    for block in other_blocks:
        for layer in block.fwd.children():
            assert not isinstance(layer, nn.BatchNorm1d)
            assert not isinstance(layer, nn.Dropout)


def test_model_contents_with_hidden_optional_layers():
    """Test the model contents with optional hidden layers included."""
    clf = ClassifierFC(
        [20, 5, 5, 2],
        input_batchnorm=False,
        hidden_batchnorm=True,
        input_dropout=0.0,
        hidden_dropout=0.1234,
    )

    first_block, other_blocks = clf.fwd.Block1, list(clf.fwd.children())[1:]

    # First block should not have batchnorm or dropout
    for layer in first_block.fwd.children():
        assert not isinstance(layer, nn.BatchNorm1d)
        assert not isinstance(layer, nn.Dropout)

    # Additional blocks should have batchnorms and dropouts
    for block in other_blocks:
        assert isinstance(block.fwd[0], nn.BatchNorm1d)
        assert isinstance(block.fwd[1], nn.Dropout)
        assert block.fwd[1].p == 0.1234


def test_model_with_all_optional_layers():
    """Test the model with both input and hidden batchnorms and dropouts."""
    clf = ClassifierFC(
        [20, 10, 2],
        input_batchnorm=True,
        input_dropout=0.123,
        hidden_batchnorm=True,
        hidden_dropout=0.456,
    )

    first_block, other_blocks = clf.fwd.Block1, list(clf.fwd.children())[1:]

    # First block should have batchnorm and dropout (with distinct p)
    assert isinstance(first_block.fwd[0], nn.BatchNorm1d)
    assert isinstance(first_block.fwd[1], nn.Dropout)
    assert first_block.fwd[1].p == 0.123

    # Second block should have both batchnorm and dropout (with distinct p)
    for block in other_blocks:
        assert isinstance(block.fwd[0], nn.BatchNorm1d)
        assert isinstance(block.fwd[1], nn.Dropout)
        assert block.fwd[1].p == 0.456


def test_relu_positions():
    """Test the positions of the ReLU layers."""
    clf = ClassifierFC(
        [50, 25, 2],
        input_batchnorm=True,
        input_dropout=0.5,
        hidden_batchnorm=True,
        hidden_dropout=0.5,
    )

    last_block = list(clf.fwd.children())[-1]
    other_blocks = list(clf.fwd.children())[:-1]

    # Last block should have no relu
    for layer in last_block.fwd.children():
        assert not isinstance(layer, nn.ReLU)

    # Other blocks should all end in ReLU
    for block in other_blocks:
        last_layer = list(block.fwd.children())[-1]
        assert isinstance(last_layer, nn.ReLU)


def test_linear_sizes():
    """Test the sizes of the linear layers are as expected."""
    layer_sizes = [256, 128, 64, 32, 16, 4, 2]
    clf = ClassifierFC(
        layer_sizes,
        input_batchnorm=False,
        input_dropout=0.0,
        hidden_batchnorm=False,
        hidden_dropout=0.0,
    )

    for idx, block in enumerate(clf.fwd.children()):

        assert isinstance(block.fwd[0], nn.Linear)
        assert block.fwd[0].in_features == layer_sizes[idx]
        assert block.fwd[0].out_features == layer_sizes[idx + 1]
