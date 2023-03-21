"""Tests for `PredictionScorer`."""
import pytest

import pandas as pd

from lyzeum.scoring import PredictionScorer


def test_threshold_dict_argument_types():
    """Test the types of the argument `threshold_dict`."""
    # Should work with Dict[str, float]
    _ = PredictionScorer({"hello": 0.5})

    # Should break with non-dict
    with pytest.raises(TypeError):
        _ = PredictionScorer([1, 2, 3])
    with pytest.raises(TypeError):
        _ = PredictionScorer({"hello", "there"})


def test_threshold_dict_argument_key_types():
    """Test the allowed key types of the `threshold_dict` argument."""
    # Should work with keys of type str
    _ = PredictionScorer({"string_key": 0.1})

    # Should break with any other key type
    with pytest.raises(TypeError):
        _ = PredictionScorer({0: 0.1, 1: 0.2})
    with pytest.raises(TypeError):
        _ = PredictionScorer({"Hello": 0.1, 1: 0.2})


def test_threshold_dict_argument_value_types():
    """Test the allowed value types of the `threshold_dict` argument."""
    # Should work with float vals
    _ = PredictionScorer({"coeliac": 0.5, "normal": 0.75})

    # Should break with vals of non-float
    with pytest.raises(TypeError):
        _ = PredictionScorer({"coeliac": 0, "normal": 1})
    with pytest.raises(TypeError):
        _ = PredictionScorer({"coeliac": 0, "normal": 1.0})
    with pytest.raises(TypeError):
        _ = PredictionScorer({"coeliac": 0.0, "normal": 1})


def test_label_type_checking():
    """Make sure the passed labels can only be List[str]."""
    scorer = PredictionScorer({"normal": 0.5, "coeliac": 0.5})

    predictions = pd.DataFrame(
        columns=["diagnosis", "coeliac", "normal"],
        data=zip(["normal", "coeliac"], [0.2, 0.5], [0.1, 0.2]),
    )

    # Should work when labels arg is a list of str.
    scorer.score_predictions(predictions, ["normal", "coeliac"])

    # Should break if labals are is not a list
    with pytest.raises(TypeError):
        scorer.score_predictions(predictions, {"normal", "coeliac"})

    # Should break if labels list contains non-str
    with pytest.raises(TypeError):
        scorer.score_predictions(predictions, ["normal", 1])
    with pytest.raises(TypeError):
        scorer.score_predictions(predictions, [1, "normal"])


def test_labels_and_diagnoses_match():
    """Test the passed labels and diagnoses in `prediction` must match."""
    scorer = PredictionScorer({"normal": 0.5, "coeliac": 0.5})

    predictions = pd.DataFrame(
        columns=["diagnosis", "coeliac", "normal"],
        data=zip(["normal", "coeliac"], [0.2, 0.5], [0.1, 0.2]),
    )

    # Should work if the passed labels match the items in diagnosis col
    scorer.score_predictions(predictions, ["normal", "coeliac"])

    # Should break if the passed labels don't match diagnosis col
    with pytest.raises(RuntimeError):
        scorer.score_predictions(predictions, ["normal", "bob"])
    with pytest.raises(RuntimeError):
        scorer.score_predictions(predictions, ["normal", "coeliac", "bob"])
