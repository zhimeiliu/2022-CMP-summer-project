"""Tests for `experiments.experiment.py`."""
import argparse
from pathlib import Path

import pytest

from lyzeum.experiment_tools.experiment import ExperimentRunner


def mimic_args() -> argparse.Namespace:
    """Mimic the c-l args needed to instantiate `ExperimentErunner`."""
    args = argparse.Namespace()
    args.valid_split = None
    args.cv_folds = None
    args.train_sources = None
    args.valid_sources = None
    return args


def test_check_save_dir_method():
    """Test `FileExistsError' gets thrown if save dir exists."""
    args = mimic_args()
    args.valid_split = 0.3
    args.save_dir = Path(".test_save_dir/")
    args.save_dir.mkdir(exist_ok=True)

    runner = ExperimentRunner(args)
    with pytest.raises(FileExistsError):
        runner.check_save_dir_does_not_exist()

    args.save_dir.rmdir()


def test_train_method_with_bad_split_options():
    """Test the `train` method throws errors with bad split options."""
    # Ask for standard train-valid split and cross validation.
    args = mimic_args()
    args.valid_split = 0.3
    args.cv_folds = 10
    with pytest.raises(RuntimeError):
        ExperimentRunner(args)

    # Ask for standard train-valid split and specific sources.
    args = mimic_args()
    args.valid_split = 0.3
    args.train_sources = "heartlands"
    args.valid_split = "addenbrookes"
    with pytest.raises(RuntimeError):
        ExperimentRunner(args)

    # Ask for cross-validation and specify train-valid sources
    args = mimic_args()
    args.cv_folds = 10
    args.train_sources = "heartlands"
    args.valid_split = "addenbrookes"
    with pytest.raises(RuntimeError):
        ExperimentRunner(args)
