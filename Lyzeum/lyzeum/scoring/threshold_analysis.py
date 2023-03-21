"""Module housing an object for doing multi-class ROC analysis."""
from typing import List, Dict, Any, Tuple

import pandas as pd
from pandas import Series, DataFrame


from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
)


class ThresholdFinder:
    """Threhold finding tool.

    Tool for finding the best thresholds in multi-class binary classifications.

    """

    def __init__(self):
        """Construct `ThresholdFinder`."""
        self.thresholds: Dict[str, float]
        self.thresholds = {}

    @staticmethod
    def _find_best_threshold(preds: Series, truths: Series) -> float:
        """Find the 'best' threshold for a binary classifier.

        Parameters
        ----------
        preds : pd.Series
            The predictions (proxy for probability).
        truths : pd.Series
            The binary truths associated with `preds`.

        Returns
        -------
        float
            The best threshold to separate positive and negative `preds`.

        Notes
        -----
        Currently we define the best threshold as the one which maximises
        the difference between (TPR - FPR).

        """
        _value_check_predictions(preds)
        fpr, tpr, thresholds = roc_curve(truths, preds)
        best_index = (tpr - fpr).argmax()
        return float(thresholds[best_index])

    def set_best_thresholds(self, predictions: DataFrame, labels: List[str]):
        """Find the best thresholds from the columns in `predictions`.

        Parameters
        ----------
        predictions : DataFrame
            A DataFrame holding the predictions and ground truths for each
            class.

        """
        _check_list_of_str(labels)
        _check_list_of_str(predictions.label.to_list())
        _check_labels_and_diagnoses_match(predictions.label, labels)

        for label in labels:
            preds = predictions[label]
            predictions[f"truth_{label}"] = predictions.label == label
            truths = predictions[f"truth_{label}"]
            self.thresholds[label] = self._find_best_threshold(preds, truths)

    def return_thresholds(self) -> Dict[str, float]:
        """Return a copy of the threshold dictionary.

        Returns
        -------
        Dict[str, float]
            The decision thresholds for each class.

        """
        return self.thresholds.copy()


class PredictionScorer:
    """Object for returning metrics on multi-class predictions.

    Parameters
    ----------
    threshold_dict : dict[str, float]
        Dictionary holding the class names and their decision thresholds.


    """

    def __init__(self, threshold_dict: Dict[str, float]):
        """Set up ThresholdAnalyser."""
        self._thresholds = threshold_dict
        self._type_check_threshold_dict()

    def _type_check_threshold_dict(self):
        """Check `self._thresholds` is dict of key, float pairs.

        Raises
        ------
        TypeError
            If `self._thresholds` is not a dict.
        TypeError
            If the keys of `self._thresholds` are not all str.
        TypeError
            If the values of `self._thresholds` are not all float.

        """
        if not isinstance(self._thresholds, dict):
            got = type(self._thresholds)
            msg = f"threshold_dict should be dict. Got {got}."
            raise TypeError(msg)

        keys, values = zip(*self._thresholds.items())
        if not all(map(lambda x: isinstance(x, str), keys)):
            msg = "threshold_dict keys should be str. Got types "
            msg += f"'{list(map(type, keys))}'"
            raise TypeError(msg)

        if not all(map(lambda x: isinstance(x, float), values)):
            msg = "threshold_dict keys should be float. Got types "
            msg += f"'{list(map(type, values))}'"
            raise TypeError(msg)

    def score_predictions(
        self,
        predictions: pd.DataFrame,
        labels: List[str],
    ) -> Tuple[Dict[str, Any], DataFrame]:
        """Obtain some scores from the predictions after thresholding.

        Parameters
        ----------
        predictions : pd.DataFrame
            DataFrame holding the predictions for each input.
        labels : List[str]
            The class labels whose predictions are to be scored.

        Returns
        -------
        score_dict : dict[Any]
            A dictionary containing the `predictions` datafram (with the
            decisions added) and a variety of scoring metrics.

        """
        predictions = predictions.copy()

        self._type_check_threshold_dict()
        _check_list_of_str(labels)
        _check_list_of_str(predictions.label.to_list())
        _check_labels_and_diagnoses_match(predictions.label, labels)

        score_dict: Dict[str, Dict[str, float]]
        score_dict = {}

        for label in labels:
            decision_key = f"decision_{label}"

            score: Dict[str, float]
            score = {}

            predictions[f"truth_{label}"] = predictions.label == label

            predictions[decision_key] = predictions[label].apply(
                lambda x: 0 if x <= self._thresholds[label] else 1
            )

            score["roc_auc"] = roc_auc_score(
                predictions[f"truth_{label}"],
                predictions[label],
            )

            score["accuracy"] = accuracy_score(
                predictions[f"truth_{label}"],
                predictions[decision_key],
            )

            score["precision"] = precision_score(
                predictions[f"truth_{label}"],
                predictions[decision_key],
            )

            score["recall"] = recall_score(
                predictions[f"truth_{label}"],
                predictions[decision_key],
            )

            score_dict[label] = score

        return score_dict, predictions


def _check_list_of_str(input_list: List[str]):
    """Check `input_list` is a list of str.

    Parameters
    ----------
    input_list : List[str]
        Check `input_list` is a list of str.

    Raises
    ------
    TypeError
        If `input_list` is not a list.
    TypeError
        If `input_list` does not only contain strings.

    """
    if not isinstance(input_list, list):
        msg = f"input_list should be list: got '{type(input_list)}.'"
        raise TypeError(msg)
    if not all(map(lambda x: isinstance(x, str), input_list)):
        types = list(map(type, input_list))
        msg = f"input_list should be list of str. Got '{types}'."
        raise TypeError(msg)


def _check_labels_and_diagnoses_match(
    diagnoses: Series,
    labels: List[str],
) -> None:
    """Check the diagnoses and the passed labels match.

    Parameters
    ----------
    diagnoses : Series
        Series holding the label for each scan.
    labels : List[str]
        The classes we want to obtain thresholds for.

    """
    unique_diagnoses = sorted(list(diagnoses.unique()))
    if not sorted(labels) == unique_diagnoses:
        msg = f"User requested to threshold on the labels '{labels}' "
        msg += f"but the data contains the labels '{unique_diagnoses}'."
        raise RuntimeError(msg)


def _value_check_predictions(preds: Series) -> None:
    """Check the predictions are on [0, 1].

    Parameters
    ----------
    preds : Series
        The predictions to be thresholded.

    Raises
    ------
    ValueError
        If preds are not on [0, 1].

    """
    if not preds.between(0.0, 1.0).all():
        msg = "Prediction values should be on [0, 1]. "
        msg += f"Got range [{preds.min()}, {preds.max()}]"
        raise ValueError(msg)
