"""Tests for experiments.data_splitting.py functions."""
from pathlib import Path
from shutil import rmtree, make_archive
from itertools import product

import pandas as pd
import numpy as np
from skimage import io

import pytest

from lyzeum.experiment_tools.data_splitting import split_data
from lyzeum.experiment_tools.data_splitting import patch_metadata_from_dir
from lyzeum.experiment_tools.data_splitting import add_n_hot_encoding


patch_dir = Path(".patches")


# pylint: disable=redefined-outer-name, unused-argument


@pytest.fixture
def create_synthetic_metadata() -> pd.DataFrame:
    """Create a fake metadata data frame to test the splitting with.

    Returns
    -------
    metadata : pd.DataFrame
        DataFrame holding 'fake' entries to test the data splitting algorithm
        on.

    """
    sources = ["scotland", "england", "ireland", "wales"]
    labels = ["cold", "flu", "covid", "coeliac"]
    metadata = pd.DataFrame()
    metadata[["source", "label"]] = list(product(sources, labels)) * 20
    metadata["case_id"] = list(map((lambda x: f"case_id_{x}"), list(metadata.index)))
    metadata["scan"] = metadata.case_id.str.replace("case_id", "scan")

    return metadata


@pytest.fixture
def create_fake_patches(create_synthetic_metadata):
    """Create a fake directory of patches for testing on."""
    # Function to save png image
    def _save_img(save_path: Path):
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)

        io.imsave(
            save_path,
            np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        )

    metadata = create_synthetic_metadata
    metadata["patch_filename"] = list(
        np.repeat([["1.png", "2.png", "3.png"]], len(metadata), 0)
    )
    metadata = metadata.explode("patch_filename").reset_index(drop=True)

    metadata["patch_path"] = metadata.apply(
        lambda row: (
            Path(
                patch_dir,
                row.source,
                row.label,
                row.case_id,
                row.scan,
                "mag_20.0",
            )
            / Path(
                row.patch_filename,
            )
        ),
        axis=1,
    ).reset_index(drop=True)
    metadata.patch_path.apply(_save_img)

    # Zip the patch dirs and delete them
    pattern = f"*/*/*/*/mag_{20.0}"
    patch_dirs = list(patch_dir.glob(pattern))
    list(map(lambda x: make_archive(str(x), "zip", x), patch_dirs))
    list(map(rmtree, patch_dirs))

    yield

    rmtree(".patches")


def test_patch_metadata_from_dir(create_fake_patches):
    """Test the function `patch_metadata_from_dir`."""
    metadata = patch_metadata_from_dir(patch_dir, 20.0)
    sources = ["scotland", "england", "ireland", "wales"]
    labels = ["cold", "flu", "covid", "coeliac"]

    assert metadata.source.isin(sources).all()
    assert metadata.label.isin(labels).all()
    assert metadata.case_id.nunique() == len(metadata)
    assert metadata.zip_file.apply(lambda x: x.endswith(".zip")).all()


def test_split_data_with_train_test_split_types(create_synthetic_metadata):
    """Test `split_data` with the standard train-test split request."""
    metadata = create_synthetic_metadata

    # Should work with float
    split_data(metadata, valid_split=0.5)

    # Should fail with non-float
    with pytest.raises(TypeError):
        split_data(metadata, valid_split=1)
    with pytest.raises(TypeError):
        split_data(metadata, valid_split=1j)
    with pytest.raises(TypeError):
        split_data(metadata, valid_split="Hello")


def test_split_data_with_train_test_split_values(create_synthetic_metadata):
    """Test `split_data` with standard train--test split values."""
    metadata = create_synthetic_metadata

    # Should work with floats on [0, 1]
    split_data(metadata, valid_split=0.0)
    split_data(metadata, valid_split=0.01)
    split_data(metadata, valid_split=0.99)
    split_data(metadata, valid_split=1.0)

    # Should break for inputs not on [0, 1]
    with pytest.raises(ValueError):
        split_data(metadata, valid_split=-0.01)
    with pytest.raises(ValueError):
        split_data(metadata, valid_split=1.01)


def test_split_data_with_train_test_split_behaviour(create_synthetic_metadata):
    """Test the behaviour of `split_data`."""
    metadata = create_synthetic_metadata

    # Testing with 50:50 split
    split_data(metadata, valid_split=0.5)
    # There should be 160 case_ids in each split
    assert (metadata.groupby("split").case_id.nunique() == 160).all()
    # There should be 80 case_ids with each of the possible labels
    assert (metadata.groupby("label").case_id.nunique() == 80).all()
    # There should be 40 case_ids with each label in each split
    assert (metadata.groupby(["split", "label"]).case_id.nunique() == 40).all()
    # There should be 40 case_ids from each source in each split
    assert (metadata.groupby(["split", "source"]).case_id.nunique() == 40).all()
    # There should be 20 case_ids from each source with each label
    assert (metadata.groupby(["label", "source"]).case_id.nunique() == 20).all()
    # There should be 10 case_ids in each split with each label from
    # each source
    assert (
        metadata.groupby(["split", "label", "source"]).case_id.nunique() == 10
    ).all()
    # There should be four kinds of label from each source
    assert (metadata.groupby("source").label.nunique() == 4).all()


def test_split_data_with_cv_split_types(create_synthetic_metadata):
    """Test `split_data` with a cross validation split types."""
    metadata = create_synthetic_metadata

    # Should work with integers
    split_data(metadata, cv_folds=2)
    split_data(metadata, cv_folds=3)

    with pytest.raises(TypeError):
        split_data(metadata, cv_folds=1.0)
    with pytest.raises(TypeError):
        split_data(metadata, cv_folds="1")


def test_split_data_with_cv_split_values(create_synthetic_metadata):
    """Test `split_data` with cross validation split values."""
    metadata = create_synthetic_metadata

    # Should work with positive integers greater than 1
    split_data(metadata, cv_folds=2)
    split_data(metadata, cv_folds=3)

    # Should break with integers of 1 or less
    with pytest.raises(ValueError):
        split_data(metadata, cv_folds=1)
    with pytest.raises(ValueError):
        split_data(metadata, cv_folds=0)
    with pytest.raises(ValueError):
        split_data(metadata, cv_folds=-1)


def test_split_data_with_cv_split_behaviour(create_synthetic_metadata):
    """Test `split_data` with cross validation split behaviour."""
    metadata = create_synthetic_metadata
    split_data(metadata, cv_folds=4)

    # There should be 80 case_ids in each fold
    assert (metadata.groupby("fold").case_id.nunique() == 80).all()
    # There should be 20 case_ids of each label in each fold
    assert (metadata.groupby(["fold", "label"]).case_id.nunique() == 20).all()
    # There should be 5 from each source, of each label, in each fold
    assert (metadata.groupby(["fold", "label", "source"]).case_id.nunique() == 5).all()


def test_split_data_by_source_types(create_synthetic_metadata):
    """Test split_data with split byy source argument types."""
    metadata = create_synthetic_metadata

    # Should work with str
    split_data(
        metadata,
        train_sources="scotland",
        valid_sources=["ireland", "wales", "england"],
    )

    split_data(
        metadata,
        train_sources=["ireland", "wales", "england"],
        valid_sources="scotland",
    )

    # Should work with list of str
    split_data(
        metadata,
        train_sources=["scotland", "england"],
        valid_sources=["wales", "ireland"],
    )

    # Should break with any other type
    with pytest.raises(TypeError):
        split_data(
            metadata,
            train_sources=1,
            valid_sources=["wales", "ireland"],
        )
    with pytest.raises(TypeError):
        split_data(
            metadata,
            train_sources=["wales", "ireland"],
            valid_sources=1,
        )


def test_split_data_split_by_source_values(create_synthetic_metadata):
    """Test split_data by source with bad values."""
    metadata = create_synthetic_metadata

    # Should work if all sources are accounted for, and don't intersect
    split_data(
        metadata,
        train_sources=["scotland", "england"],
        valid_sources=["wales", "ireland"],
    )

    # Should break if there is an intersection
    with pytest.raises(ValueError):
        split_data(
            metadata,
            train_sources="scotland",
            valid_sources="scotland",
        )
    with pytest.raises(ValueError):
        split_data(
            metadata,
            train_sources=["scotland", "england"],
            valid_sources=["scotland", "england", "wales"],
        )

    # Should break if sources go unaccounted for
    with pytest.raises(AssertionError):
        split_data(metadata, train_sources="scotland", valid_sources="wales")


def test_split_data_by_source_behaviour(create_synthetic_metadata):
    """Test the behaviour of split_data by source."""
    metadata = create_synthetic_metadata

    split_data(
        metadata,
        train_sources=["wales", "ireland"],
        valid_sources=["scotland", "england"],
    )
    assert (metadata.groupby("split").source.nunique() == 2).all()
    assert (metadata.groupby("split").source.nunique() == 2).all()

    split_data(
        metadata,
        train_sources=["wales"],
        valid_sources=["scotland", "england", "ireland"],
    )
    assert metadata.loc[metadata.split == "train"].source.nunique() == 1
    assert metadata.loc[metadata.split == "valid"].source.nunique() == 3


def test_split_data_with_bad_options(create_synthetic_metadata):
    """Test `split_data` with incompatible choices of split options."""
    metadata = create_synthetic_metadata

    with pytest.raises(RuntimeError):
        split_data(metadata, valid_split=0.5, cv_folds=4)
    with pytest.raises(RuntimeError):
        split_data(metadata, valid_split=0.3, train_sources="bob")
    with pytest.raises(RuntimeError):
        split_data(metadata, valid_split=0.3, valid_sources="bob")
    with pytest.raises(RuntimeError):
        split_data(
            metadata,
            valid_split=0.3,
            valid_sources=["bob"],
            train_sources=["roger"],
            cv_folds=10,
        )
    with pytest.raises(RuntimeError):
        split_data(metadata)


def test_add_n_hot_encoding_with_missing_label():
    """Test `add_n_hot_encoding` with missing labels."""
    metadata = pd.DataFrame()
    metadata["label"] = ["normal", "normal", "coeliac", "ambiguous"]

    with pytest.raises(RuntimeError):
        add_n_hot_encoding(metadata, ["normal", "coeliac"])


def test_add_n_hot_encoding_with_duplicate_labels():
    """Test `add_n_hot_encoding` with non-unique labels."""
    metadata = pd.DataFrame()
    metadata["label"] = ["normal", "normal", "coeliac", "ambiguous"]

    with pytest.raises(ValueError):
        add_n_hot_encoding(metadata, ["normal", "normal"])


def test_add_n_hot_encoding_targets_are_correct():
    """Test `add_n_hot_encoding` gives correct targets."""
    metadata = pd.DataFrame()
    metadata["label"] = ["normal", "normal", "coeliac", "ambiguous"]
    add_n_hot_encoding(metadata, ["normal", "coeliac", "ambiguous"])

    normal_trgt, coeliac_trgt, ambig_trgt = tuple(np.eye(3, dtype=int))

    norm_check = (
        metadata.loc[metadata.label == "normal", "target"]
        .apply(lambda x: (x == normal_trgt).all())
        .all()
    )
    assert bool(norm_check) is True, "Unexpected target for normal label."

    coel_check = (
        metadata.loc[metadata.label == "coeliac", "target"]
        .apply(lambda x: (x == coeliac_trgt).all())
        .all()
    )
    assert bool(coel_check) is True, "Unexpected target for coeliac label."

    ambig_check = (
        metadata.loc[metadata.label == "ambiguous", "target"]
        .apply(lambda x: (x == ambig_trgt).all())
        .all()
    )
    assert bool(ambig_check) is True, "Unexpected target for ambiguous label"
