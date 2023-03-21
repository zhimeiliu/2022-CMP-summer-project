"""Tests for ClassifierCNN."""

import pytest

from lyzeum.torch_toolbelt.models import ClassifierCNN


def test_encoder_argument():
    """Test the types and values accepted by enoder argument."""
    # Should work with acceptable str
    _ = ClassifierCNN("resnet18", 2)

    # Should break with non-str
    with pytest.raises(TypeError):
        _ = ClassifierCNN(123, 2)

    # Should break with unaccepted model type
    with pytest.raises(ValueError):
        _ = ClassifierCNN("Batman", 5)


def test_num_classes_argument():
    """Test the num_classes argument."""
    # Should work with positive int
    _ = ClassifierCNN("resnet18", 1)
    _ = ClassifierCNN("resnet18", 2)
    _ = ClassifierCNN("resnet18", 10)

    # Should break with int less than 1
    with pytest.raises(ValueError):
        _ = ClassifierCNN("resnet18", 0)
    with pytest.raises(ValueError):
        _ = ClassifierCNN("resnet18", -1)
    with pytest.raises(ValueError):
        _ = ClassifierCNN("resnet18", -2)

    # Should break with non int
    with pytest.raises(TypeError):
        _ = ClassifierCNN("resnet18", 1.0)
    with pytest.raises(TypeError):
        _ = ClassifierCNN("resnet18", "2")


def test_pretrained_arg():
    """Test the pretrained arg."""
    # Should work with any bool
    _ = ClassifierCNN("resnet18", 5, pretrained=True)
    _ = ClassifierCNN("resnet18", 5, pretrained=False)

    # Should fail with any non-boolean
    with pytest.raises(TypeError):
        _ = ClassifierCNN("resnet18", 5, pretrained=1)
    # Should fail with any non-boolean
    with pytest.raises(TypeError):
        _ = ClassifierCNN("resnet18", 5, pretrained="True")


def test_pool_style_arg():
    """Test the pool_style argument."""
    # The three accepted values
    _ = ClassifierCNN("resnet18", 3, pool_style="avg")
    _ = ClassifierCNN("resnet18", 3, pool_style="max")
    _ = ClassifierCNN("resnet18", 3, pool_style="concat")

    # Anything else should fail
    with pytest.raises(ValueError):
        _ = ClassifierCNN("resnet18", 3, pool_style="bob-the-builer")


def test_clf_hidden_sizes_type():
    """Test the type of the clf_hidden_sizes argument."""
    # Should only accept a list
    _ = ClassifierCNN("resnet18", 10, clf_hidden_sizes=[10, 5])

    # Should reject aanything else
    with pytest.raises(TypeError):
        _ = ClassifierCNN("resnet18", 10, clf_hidden_sizes=(10, 5))
    with pytest.raises(TypeError):
        _ = ClassifierCNN("resnet18", 10, clf_hidden_sizes={10, 5})
