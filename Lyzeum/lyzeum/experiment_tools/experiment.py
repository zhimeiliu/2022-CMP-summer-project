"""Generic class for running an experiment."""
import argparse
from pathlib import Path
import pandas as pd

from lyzeum.experiment_tools.data_splitting import (
    separate_splits,
    check_split_options_are_compatible,
)


def create_argument_parser() -> argparse.ArgumentParser:
    """Process the minimum command-line arguments for a training experiment.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser with the basic arguments needed for a classification experiment

    """
    parser = argparse.ArgumentParser(
        description="Train a classification experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "patch_root",
        help="Path to the parent directory the patches are saved in.",
        type=str,
    )

    parser.add_argument(
        "--magnification",
        help="Magnification to take the patches at.",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--encoder",
        help="Type of encoder to use in the experiment",
        type=str,
        default="resnet50",
    )

    parser.add_argument(
        "--valid-split",
        help="Fraction of the data to put in the validation split.",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--cv-folds",
        help="Number of folds to use in a cross-validation experiment",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--train-sources",
        help="Sources to use as the training set",
        type=str,
        default=None,
        nargs="*",
    )

    parser.add_argument(
        "--valid-sources",
        help="Sources to use in the validation set",
        type=str,
        default=None,
        nargs="*",
    )

    parser.add_argument(
        "--rng-seed",
        help="Integer seed for the random number generators",
        type=int,
        default=123,
    )

    parser.add_argument(
        "--exclude-sources",
        help="Sources you wish to exclude from the experiment.",
        type=str,
        default=None,
        nargs="*",
    )

    parser.add_argument(
        "--num-workers",
        help="Number of workers for each data loader to spawn.",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--epochs", help="Number of epochs to train for.", type=int, default=3,
    )

    parser.add_argument(
        "--freeze-epochs",
        help="Number of epochs to train with the encoder parameters frozen.",
        type=int,
        default=0,
    )

    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)

    return parser


class ExperimentRunner:
    """Generic class for running an experiment.

    Parameters
    ----------
    args : argpase.Namespace
        Command-line arguments necessary for running the experiment.

    """

    def __init__(self, args: argparse.Namespace):
        """Construct the experiment class."""
        self.args = args
        self.fold = 1
        check_split_options_are_compatible(
            self.args.valid_split,
            self.args.cv_folds,
            self.args.train_sources,
            self.args.valid_sources,
        )

    def check_save_dir_does_not_exist(self):
        """Check the output directory does not exist.

        Raises
        ------
        FileExistsError
            If the experiment's output directory already exists, raise.

        """
        if Path(self.args.save_dir).exists():
            raise FileExistsError(self.args.save_dir)

    def train_on_single_split(
        self, train_df: pd.DataFrame, valid_df: pd.DataFrame,
    ):
        """Train the experiment on a single split.

        Parameters
        ----------
        train_df : pd.DataFrame
            ....
        valid_df : pd.DataFrame
            ....

        Notes
        -----
        This function should be overloaded and in it all of the training
        happens, from the instantiation of dataloaders to the saving of the
        model.

        """
        raise NotImplementedError()

    def train(self, metadata: pd.DataFrame):
        """Train the model.

        Parameters
        ----------
        metadata : pd.DataFrame
            Metadata for the patches to be used in the experiment.

        """
        self.check_save_dir_does_not_exist()
        if self.args.valid_split is not None:
            train_df, valid_df = separate_splits(metadata)
            self.train_on_single_split(train_df, valid_df)

        elif self.args.cv_folds is not None:
            for fold in sorted(metadata.fold.unique()):
                self.fold = fold
                metadata["valid_fold"] = fold
                train_df, valid_df = separate_splits(metadata, valid_fold=fold)
                self.train_on_single_split(train_df, valid_df)

        elif (self.args.train_sources and self.args.valid_sources) is not None:
            train_df, valid_df = separate_splits(metadata)
            self.train_on_single_split(train_df, valid_df)

        else:
            raise ValueError("Unrecognised split options.")
