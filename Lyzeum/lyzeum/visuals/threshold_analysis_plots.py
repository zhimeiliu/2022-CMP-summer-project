"""ROC analysis plotting tool."""
from typing import Union, List, Any, Dict
from string import ascii_lowercase
from pathlib import Path
from itertools import product

from numpy import linspace
from pandas import DataFrame, Series

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

from lyzeum.visuals.utils import (
    set_axis_aspect,
    set_xlims_and_ticks,
    set_ylims_and_ticks,
)

# pylint: disable=too-many-locals

_cmap = plt.cm.PuRd_r(linspace(0, 1, 7))  # pylint: disable=no-member


def threshold_analysis_plots(
    pred_df: DataFrame,
    classes: Union[str, List[str]],
    save_path: Union[str, Path],
    dpi: int = 500,
) -> None:
    """Generate an ROC plot.

    Parameters
    ----------
    pred_df : DataFrame
        DataFrame containing the predictions and the truths. The ground truths
        should be saved as str in the 'label' column. The predictions
        should be saved in columns, 'normal', 'coeliac', etc, and supplied
        in the `classes` argument.
    classes : List[str]
        Class labels to perform ROC analysis on.
    save_path : Union[str, Path]
        File path for saving the plot.
    dpi : int, optional
        DPI to use when saving the figure.

    """
    classes = _process_classes_arg(classes)
    if "valid_fold" not in pred_df.keys():
        pred_df["valid_fold"] = 1

    figsize = 1.6 * len(classes), 3.5
    fig, axes = plt.subplots(2, len(classes), figsize=figsize)
    fig.subplots_adjust(
        wspace=0,
        hspace=0.15,
        bottom=0.075,
        left=0.125,
        top=0.975,
        right=0.975,
    )

    folds = sorted(pred_df.valid_fold.unique())

    for (idx, label), fold in product(enumerate(classes), folds):

        fold_df = pred_df.loc[pred_df.valid_fold == fold].reset_index()

        pred = fold_df[label]
        fold_df[f"truth_{label}"] = fold_df.label == label
        truth = fold_df[f"truth_{label}"]

        roc_dict = _roc_info(pred, truth)

        pr_dict = _precision_recall_info(pred, truth)

        axes[0, idx].plot(
            roc_dict["fpr"],
            roc_dict["tpr"],
            color=_cmap[fold - 1],
            label=f"AUC = {round(roc_dict['auc'], 3):.3f}",
            lw=0.75,
        )

        axes[1, idx].plot(
            pr_dict["recall"],
            pr_dict["precision"],
            color=_cmap[fold - 1],
            label=f"AUC = {round(pr_dict['auc'], 3):.3f}",
            lw=0.75,
        )

    letters = iter(ascii_lowercase[: len(axes.ravel())])
    hor = [0.45, 0.05]
    for row, (col, label) in product(range(2), enumerate(classes)):
        axes[row, col].text(
            hor[row],
            0.5,
            f"({next(letters)}) --- {label.capitalize()}",
            transform=axes[row, col].transAxes,
            fontsize=8,
        )

        axes[row, col].legend(
            loc="lower left",
            fontsize=6,
            bbox_to_anchor=(hor[row] - 0.05, 0),
        )

    axes[0, 0].set_ylabel("True positive rate", labelpad=0)
    axes[1, 0].set_ylabel("Precision", labelpad=0)

    # Formatting specific to ROC plots
    for axis in axes[0, :].ravel():
        axis.set_xlabel("False positive rate", labelpad=0)
        set_ylims_and_ticks(
            axis,
            bottom=0.7,
            top=1.02,
            major=linspace(0.7, 1.0, 4),
        )
        set_xlims_and_ticks(axis, left=0.0, right=0.4)

    # Formatting specific to PR plots
    for axis in axes[1, :].ravel():
        set_xlims_and_ticks(axis, left=0.6, right=1.0)
        axis.set_xlabel("Recall", labelpad=0)
        axis.set_ylim(bottom=0.6, top=1.02)

    for axis in axes[:, 1:].ravel():
        axis.set_yticklabels([])

    for axis in axes.ravel():
        set_axis_aspect(axis, ratio=1.0)

    for axis in axes[:, :-1].ravel():
        axis.xaxis.get_major_ticks()[-1].set_visible(False)

    _process_save_path(save_path)
    fig.savefig(save_path, dpi=dpi)


def _process_save_path(save_path: Union[str, Path]) -> None:
    """Process `save_path` for saving the figure.

    Parameters
    ----------
    save_path : str or Path
        The path to save the figure as.

    """
    if not Path(save_path).parent.exists():
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)


def _process_classes_arg(classes: Union[str, List[str]]) -> List[str]:
    """Process the classes argument.

    Parameters
    ----------
    classes : str or List[str]
        Classes argument passed to `roc_plot`.

    """
    if isinstance(classes, str):
        return [classes]
    if _is_list_of_str(classes):
        return classes

    raise TypeError(f"classes should be a list of str. Got '{classes}'.")


def _is_list_of_str(to_check: Any) -> bool:
    """Check if `to_check` is a list of str.

    Parameters
    ----------
    to_check : Any
        A variable to check.

    Returns
    -------
    bool
        Is `to_check` a list of str?

    """
    if not isinstance(to_check, list):
        return False
    if not all(map(lambda x: isinstance(x, str), to_check)):
        return False
    return True


def _roc_info(preds: Series, truths: Series) -> Dict[str, float]:
    """Return ROC information necessary for plotting.

    Parameters
    ----------
    preds : Series
        Predictions (float on [0, 1]).
    truths : Series
        Binary ground truths.

    Returns
    -------
    fpr : ndarray
        False positive rates from ROC analysis.
    tpr : ndarray
        True positive rates from ROC analysis.
    roc_auc : float
        The area under the ROC curve.

    """
    fpr, tpr, _ = roc_curve(truths, preds)
    roc_auc = roc_auc_score(truths, preds)
    return {"fpr": fpr, "tpr": tpr, "auc": roc_auc}


def _precision_recall_info(preds: Series, truths: Series) -> Dict[str, float]:
    """Return precision--recall curve info.

    Parameters
    ----------
    preds : Series
        The predictions (float on [0, 1]).
    truths : Series
        The ground truths (binary)


    """
    precision, recall, _ = precision_recall_curve(truths, preds)
    area = auc(recall, precision)
    return {"precision": precision, "recall": recall, "auc": area}
