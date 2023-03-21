"""Functions for splitting multi-source WSI data for training and testing."""
from itertools import product
from typing import Union, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

from torch import eye

# pylint: disable=too-many-arguments


def patch_metadata_from_dir(
    patch_root: Union[str, Path],
    magnification: float,
    exclude_sources: Optional[Union[List[str], str]] = None,
) -> pd.DataFrame:
    """Obtain all relevant patch metadata from patches in `patch_root`.

    Parameters
    ----------
    patch_root : PathLike
        Path to the top directory in which the paths are saved.
    magnification : float
        The level of magnification to take the patches at.
    exclude_sources : List[str] or str, optional
        Source you wish to exclude when loading the data.

    Returns
    -------
    metadata : pd.DataFrame
        A DataFrame holding the metadata for each patch. The columns are:
        `patch_path`, `patch_filename`, `scan`, `case_id`, `label`,
        `source` and`top_dir`.

    Raises
    ------
    FileNotFoundError
        `patch_root` must exist.
    Exception
        If not patches are found when searching the tree, raise.

    Notes
    -----
    We assume the patches are saved in a structure conforming to:

    `'top_dir/patches/source/label/case_id/scan/mag_X.zip`,

    where `top_dir` can be any arbitrary path itself.

    """
    patch_root = Path(patch_root).resolve()

    if not patch_root.exists():
        raise FileNotFoundError(f"{patch_root} does not exist.")

    glob_pattern = f"*/*/*/*/mag_{float(magnification)}.zip"

    patch_zips = list(patch_root.glob(glob_pattern))
    assert len(patch_zips) != 0, "Never found any patch zips"

    metadata = pd.DataFrame()
    metadata["zip_file"] = patch_zips
    metadata["top_dir"] = patch_root

    items = ["scan", "case_id", "label", "source"]
    metadata[items] = (
        metadata.zip_file.apply(lambda x: list(x.parents)[0:4])
        .apply(lambda x: [item.name for item in x])
        .to_list()
    )
    if exclude_sources is not None:
        metadata = metadata.loc[
            ~metadata.source.isin(pd.Series(exclude_sources))
        ].reset_index(drop=True)

    return metadata.astype(str)


def split_data(
    metadata: pd.DataFrame,
    valid_split: Optional[float] = None,
    cv_folds: Optional[int] = None,
    train_sources: Optional[Union[List[str], str]] = None,
    valid_sources: Optional[Union[List[str], str]] = None,
    seed: int = 123,
):
    """Split the data in preparation for a training experiment.

    The data are split on the case_id level.

    Notes
    -----
    There are three main options available:
        — A standard train-valid split, where each data source and label
          is given proportional representation in each split.
        — A cross-validation experiment, where the same proportional
          representation of each label and data source is applied as
          before.
        — A split where the training and validation sources are explicitly
          specified. I.e., you want to train on data from certain sources and
          validate on data from another source(s).

    Raises
    ------
    AssertionError
        If the user requests a standard train-test split, but not all case_ids
        are correctly assigned to a split, we raise an AssertionError.

    """
    metadata.sort_values(by="case_id", inplace=True)
    metadata.reset_index(drop=True, inplace=True)
    rng = np.random.default_rng(seed)

    check_split_options_are_compatible(
        valid_split=valid_split,
        cv_folds=cv_folds,
        train_sources=train_sources,
        valid_sources=valid_sources,
    )

    if valid_split is not None:
        metadata["split"] = "unassigned"
        _standard_train_valid_split(metadata, valid_split, rng)
        msg = "At least one case_id has not been assigned a split."
        assert not (metadata.split == "unassigned").any(), msg

    if train_sources is not None and valid_sources is not None:
        metadata["split"] = "unassigned"
        _split_by_sources(metadata, train_sources, valid_sources)
        msg = "At least one case_id has not been assigned a split."
        assert not (metadata.split == "unassigned").any(), msg

    if cv_folds is not None:
        metadata["fold"] = "unassigned"
        _cross_validation_split(metadata, cv_folds, rng)
        msg = "At least one case_id has not been assigned a fold."
        assert not metadata.fold.isin(["unassigned"]).any(), msg


def check_split_options_are_compatible(
    valid_split: Optional[float] = None,
    cv_folds: Optional[int] = None,
    train_sources: Optional[Union[List[str], str]] = None,
    valid_sources: Optional[Union[List[str], str]] = None,
):
    """Check the split options to `split_data` are compatible.

    Parameters
    ----------
    valid_split : float, optional
        Fraction of case_ids to put in the validation set.
    cv_folds : int, optional
        Number of cross-validation folds to use.
    train_sources : List[str] or str, optional
        Sources to assign to the training set.
    valid_sources : List[str] or str, optional
        Sources to assign to the validation set.

    Raises
    ------
    RuntimeError
        Asking for more than one kind of split is not possible and thus raises
        a RuntimeError.

    """
    options = [
        valid_split is not None,
        cv_folds is not None,
        (train_sources is not None) or (valid_sources is not None),
    ]
    if np.sum(options) != 1:
        raise RuntimeError("Incompatible split options requested.")


def _standard_train_valid_split(
    metadata: pd.DataFrame,
    valid_split: float,
    rng: np.random.Generator,
):
    """Split the case_ids into a standard train-test split.

    Parameters
    ----------
    metadata : pd.DataFrame
        The metadata for the patches.
    valid_split : float
        The fraction of the data to put in the validation set.
    rng : np.random.Generator
        Numpy random number generator.

    Raises
    ------
    TypeError
        `valid_split` must be a float.
    ValueError
        If `valid_split` is not on the interval [0, 1] an incorrect value has
        been passed.

    """
    if not isinstance(valid_split, float):
        raise TypeError(f"valid split {valid_split} must be float")
    if not 0.0 <= valid_split <= 1.0:
        msg = f"Valid split {valid_split} must be on [0, 1]"
        raise ValueError(msg)

    labels, sources = metadata.label.unique(), metadata.source.unique()
    for label, source in product(labels, sources):

        subset = _get_subset_of_patients_to_assign(metadata, source, label)
        subset = _get_subset_of_patients_to_assign(metadata, source, label)
        subset = _random_shuffle_df(subset, rng)

        split_ind = int(len(subset) * (1.0 - valid_split))
        train_case_ids = subset.case_id.to_numpy()[:split_ind]
        valid_case_ids = subset.case_id.to_numpy()[split_ind:]
        metadata.loc[metadata.case_id.isin(train_case_ids), "split"] = "train"
        metadata.loc[metadata.case_id.isin(valid_case_ids), "split"] = "valid"


def _split_by_sources(
    metadata: pd.DataFrame,
    train_sources: Union[str, List[str]],
    valid_sources: Union[str, List[str]],
):
    """Split the training and validation data by source.

    Parameters
    ----------
    metadata : pd.DataFrame
        The patch metadata.
    train_sources : Union[List[str], str]
        The sources to be in the training set.
    valid_sources : Union[List[str], str]
        The sources to be in the validation set.

    """
    if not isinstance(train_sources, (list, str)):
        msg = "train_sources should be a list of str."
        msg += f"Got '{type(train_sources)}'"
        raise TypeError(msg)

    if not isinstance(valid_sources, (list, str)):
        msg = "valid_sources should be a list of str."
        msg += f"Got '{type(valid_sources)}'"
        raise TypeError(msg)

    train = set(pd.Series(train_sources))
    valid = set(pd.Series(valid_sources))

    if len((intersect := train.intersection(valid))) != 0:
        msg = "There is an intersection between the training and validation "
        msg += f"splits. The following sources intersect : '{intersect}'."
        raise ValueError(msg)

    metadata.loc[metadata.source.isin(train), "split"] = "train"
    metadata.loc[metadata.source.isin(valid), "split"] = "valid"


def _cross_validation_split(
    metadata: pd.DataFrame,
    cv_folds: int,
    rng: np.random.Generator,
):
    """Split the case_ids into `cv_folds`.

    Parameters
    ----------
    metadata : pd.DataFrame
        The metadata for the patches.
    cv_folds : int
        The number of cross validattion folds we wish to split the data into.

    Raises
    ------
    TypeError
        If cv_folds is not an integer a TyperError is raised.
    ValueError
        If cv_folds is not greater than zero we raise a type error.


    """
    if not isinstance(cv_folds, int):
        raise TypeError(f"cv_fold should be an integer: {cv_folds}")
    if cv_folds < 2:
        raise ValueError("cv_folds must be greater than 1.")

    labels, sources = metadata.label.unique(), metadata.source.unique()

    for label, source in product(labels, sources):
        subset = _get_subset_of_patients_to_assign(metadata, source, label)
        subset = _random_shuffle_df(subset, rng)
        subset["fold"] = (np.arange(len(subset)) % cv_folds) + 1

        for fold, grouped in subset.groupby("fold"):
            metadata.loc[metadata.case_id.isin(grouped.case_id), "fold"] = fold


def _get_subset_of_patients_to_assign(
    metadata: pd.DataFrame,
    source: str,
    label: str,
) -> pd.DataFrame:
    """Extract the susbet of case_ids to assign to a split.

    Parameters
    ----------
    metadata : pd.DataFrame
        DataFrame holding the scan-level metadata.
    source : str
        The data source whose scans we wish to assign to a split.
    label : str
        The label of the scans we wish to assign to a split.

    Returns
    -------
    unique_cases : pd.DataFrame
        A DataFrame of case_ids which belong to `source`, bear the label
        `label`, and have not yet been assigned.

    Notes
    -----
    In some cases, multiple scans with the same case_id have different
    labels. Cases which get assigned to a split based on one label, should not
    then be reassigned based on another scan with a different label. The
    first operation in this functions, which filters out only unassigned cases,
    avoids this problem.

    """
    assert not ("split" in metadata and "fold" in metadata)
    assert "split" in metadata or "fold" in metadata
    split_key = "split" if "split" in metadata else "fold"

    unassigned = metadata.loc[metadata[split_key] == "unassigned"]

    cases = unassigned.loc[
        (unassigned.source == source) & (unassigned.label == label),
        ["case_id"],
    ]
    return cases.drop_duplicates(subset="case_id").reset_index(drop=True)


def _random_shuffle_df(
    data_frame: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a randomly order version of `data_frame`.

    Parameters
    ----------
    data_frame : pd.DataFrame
        DataFrame to be shuffled.

    Returns
    -------
    pd.DataFrame
        A randomly shuffled version of `data_frame`.

    """
    shuffled_inds = rng.permutation(len(data_frame))
    return data_frame.iloc[shuffled_inds].reset_index(drop=True)


def separate_splits(
    metadata: pd.DataFrame,
    valid_fold: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Turn the training and validation splits into separate data frames.

    Parameters
    ----------
    metadata : pd.DataFrame
        DataFrame holding the patch metadata.
    valid_fold : int
        The integer index of the validation fold.

    Returns
    -------
    train : pd.DataFrame
        DataFrame holding the metadata for the training patches.
    valid : pd.DataFrame
        DataFrame holding the metadata for the validation patches.

    """
    if valid_fold is not None:
        assert isinstance(valid_fold, int), "valid_fold should be an integer."
        metadata["split"] = "unassigned"
        metadata.loc[metadata.fold == valid_fold, "split"] = "valid"
        metadata.loc[metadata.fold != valid_fold, "split"] = "train"

    train = metadata.loc[metadata.split == "train"].reset_index(drop=True)
    valid = metadata.loc[metadata.split == "valid"].reset_index(drop=True)

    return train, valid


def add_n_hot_encoding(metadata: pd.DataFrame, labels: List[str]) -> None:
    """Add the labels as n-hot-encoded vectors.

    Add the column `"truth"` with n-hot-encoded vectors and individual binary
    truths for each condition in labels.

    Parameters
    ----------
    metadata : pd.DataFrame
        Patch-level metadata storing all keyy information in columns.
    labels : list of str
        List of the labels to include in the n-hot-encoding.

    Raises
    ------
    ValueError
        If `labels` contains non-unique elements, raise.
    RuntimeError
        If there are labels in metadata not in `labels`, raise.

    Notes
    -----
    We modify `metadata` in place.

    """
    if not len(np.unique(labels)) == len(labels):
        msg = f"labels must contain unique elements only, got {labels}."
        raise ValueError(msg)

    if not metadata.label.isin(labels).all():
        msg = "At leat one label not included in labels. "
        msg += f"Got {labels} but metadata contains "
        msg += f"{metadata.label.unique()}"
        raise RuntimeError(msg)

    identity = eye(len(labels))
    metadata["target"] = metadata.label.apply(
        lambda x: identity[labels.index(x)].numpy()
    )
    metadata[[f"truth_{lab}" for lab in labels]] = metadata.target.to_list()


if __name__ == "__main__":
    pass
    # data = patch_metadata_from_dir("/media/jim/HDD-1/patches_224_168", 8)
    # split_data(data, valid_split=0.3)
    # print(
    #     data.groupby(by=["source", "split"]).case_id.nunique() / data.case_id.nunique()
    # )

    # split_data(
    #     data,
    #     train_sources="heartlands",
    #     valid_sources="addenbrookes",
    # )
    # split_data(
    #     data,
    #     train_sources=["heartlands", "addenbrookes"],
    #     valid_sources="addenbrookes",
    # )
    # print(data.groupby(by=["source", "split"]).case_id.nunique())
