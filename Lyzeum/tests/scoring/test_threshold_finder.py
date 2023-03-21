"""Tests for `ThresholdFinder`."""
import pytest
import pandas as pd

from lyzeum.scoring import ThresholdFinder


def test_forbidden_prediction_values():
    """Test the values accepted for the predictions."""
    finder = ThresholdFinder()

    # Should break with a prediction less than zero
    data = pd.DataFrame(
        columns=["normal", "diagnosis"],
        data=zip([-0.1, 0.0, 0.5, 1.0], ["normal", "normal"] * 2),
    )
    with pytest.raises(ValueError):
        finder.set_best_thresholds(data, ["normal"])

    # Should break with a prediction greater than one
    data = pd.DataFrame(
        columns=["coeliac", "diagnosis"],
        data=zip([0.0, 0.0, 0.5, 1.1], ["coeliac", "coeliac"] * 2),
    )
    with pytest.raises(ValueError):
        finder.set_best_thresholds(data, ["coeliac"])


def test_diagnoses_are_str():
    """Test we catch forbidden ground truth values."""
    finder = ThresholdFinder()

    # Should break if the diagnoses are not str
    data = pd.DataFrame(
        columns=["coeliac", "diagnosis"], data=zip(([0.1, 0.2]), ("Hello", 2.0))
    )
    with pytest.raises(TypeError):
        finder.set_best_thresholds(data, ["coeliac"])


def test_with_extra_labels_passed():
    """Test we raise an error when the labels don't match the pred df."""
    finder = ThresholdFinder()

    data = pd.DataFrame(
        columns=["normal", "diagnosis"], data=zip((0.1, 1.0), ("normal", "coeliac"))
    )

    # Should break if we ask for an extra class to be thresholded
    with pytest.raises(RuntimeError):
        finder.set_best_thresholds(data, ["normal", "coeliac", "EXTRA"])
