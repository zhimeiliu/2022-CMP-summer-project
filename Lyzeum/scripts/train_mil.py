#!/usr/bin/env python3.9
"""Train the multiple instance classifier."""
import argparse
from pathlib import Path

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything, loggers

import numpy as np


from lyzeum.experiment_tools import data_splitting
from lyzeum.experiment_tools.data_splitting import add_n_hot_encoding
from lyzeum.experiment_tools.experiment import (
    create_argument_parser,
    ExperimentRunner,
)

from lyzeum.torch_toolbelt.models import ClassifierCNN
from lyzeum.torch_toolbelt.image_transforms import get_img_transforms
from lyzeum.torch_toolbelt.datasets import MultiInstanceDataset

from lyzeum.torch_toolbelt.loss import MILLoss


from lyzeum.visuals.loss_plotting import plot_losses


# pylint: disable=too-many-ancestors,too-many-arguments,arguments-differ


def process_command_line_args() -> argparse.Namespace:
    """Process the command-line arguments."""
    parser = create_argument_parser()

    parser.add_argument(
        "--save-dir",
        help="Directory to put the experiment's output in",
        type=str,
        default="output_data/",
    )

    parser.add_argument(
        "--bag-size",
        help="Number of tiles to randomly sample from patient as a bag",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--alpha",
        help="Number of items in positive bags to label as positive.",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--beta",
        help="Number of items in positive bags to mark as negative",
        type=int,
        default=0,
    )

    return parser.parse_args()


class MILModel(LightningModule):
    """Multiple instance learning model.

    Parameters
    ----------
    loss_func : nn.BCEWithLogitsLoss
        Loss function
    freeze_epochs : int
        Number of epochs to train only the classifier (and not the encoder)
        for.
    pretrained_encoder : bool
        Should we use pytorch's pretrained version of the encoder?
    classifier_batchnorms : bool
        Should we put batchnorms before the activations?

    """

    def __init__(
        self,
        loss_func,
        encoder: str,
        freeze_epochs: int = 0,
        pretrained: bool = True,
        learning_rate: float = 1e-4,
    ):
        """Construct model."""
        super().__init__()

        self.model = ClassifierCNN(
            encoder=encoder,
            pretrained=pretrained,
            num_classes=2,
            # clf_hidden_sizes=[4096, 4096],
            # clf_hidden_dropout=0.5,
        )
        self.my_loss = loss_func
        self.learning_rate = learning_rate
        self.freeze_epochs = freeze_epochs

    def forward(self, batch):
        """Pass a batch of inputs through the network."""
        frozen_encoder = self.current_epoch < self.freeze_epochs
        logits = self.model.forward(batch, frozen_encoder=frozen_encoder)
        return logits

    def training_step(self, batch, batch_idx):
        """Perform one training step."""
        assert self.model.training is True, "Should be in training mode."
        bag, label = batch
        bag, label = bag.squeeze(), label.squeeze()
        return {"loss": self.my_loss(self.forward(bag), label)}

    def training_epoch_end(self, outputs):
        """End of training epoch."""
        avg_loss = torch.Tensor([x["loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss, prog_bar=True)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """Perform one validation step."""
        assert self.model.training is False, "Should be in eval model."
        bag, label = batch
        bag, label = bag.squeeze(), label.squeeze()
        return {"valid_loss": self.my_loss(self.forward(bag.squeeze()), label)}

    def validation_epoch_end(self, outputs):
        """End of epoch validation process."""
        avg_loss = torch.tensor([x["valid_loss"] for x in outputs]).mean()
        self.log("valid_loss", avg_loss, prog_bar=True)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        """Do inference on a batch."""
        self.eval()
        return self.forward(batch).cpu()

    def configure_optimizers(self):
        """Return the optimiser."""
        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        # milestones = [self.freeze_epochs] if self.freeze_epochs != 0 else []
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optim,
        #     milestones,
        #     gamma=0.01,
        #     verbose=False,
        # )
        # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        #     optim, lr_lambda=lambda epoch: 0.85
        # )
        return {
            "optimizer": optim,
            # "scheduler": scheduler,
            # "interval": "epoch",
            # "frequency": 1,
        }


class MILExperiment(ExperimentRunner):
    """Class for running the MIL experiment."""

    def __init__(self, args: argparse.Namespace):
        """Construct `MILExperiment`."""
        super().__init__(args)
        self.model_class = MILModel

    def train_on_single_split(self, train_df, valid_df):
        """Train the model on the `train_df`-`valid_df` split.

        Parameters
        ----------
        train_df : pd.DataFrame
            Scan level metadata where the patch_path column is a list of
            patches from each scan and the rest of the columns give the
            patient's metadata.
        valid_df : pd.DataFrame
            Same as `train_df` but for the validation set.

        """
        out_dir = Path(self.args.save_dir).resolve()

        train_set = MultiInstanceDataset(
            train_df.zip_file.to_list(),
            torch.as_tensor(np.vstack(train_df.target)),
            self.args.bag_size,
            x_tfms=get_img_transforms(True),
        )

        if len(valid_df) == 0:
            valid_set = None
        else:
            valid_set = MultiInstanceDataset(
                valid_df.zip_file.to_list(),
                torch.as_tensor(np.vstack(valid_df.target.to_list())),
                self.args.bag_size,
                x_tfms=get_img_transforms(False),
            )

        train_loader = DataLoader(
            train_set,
            shuffle=True,
            batch_size=1,
            num_workers=self.args.num_workers,
            prefetch_factor=1,
        )

        valid_loader = DataLoader(
            valid_set,
            shuffle=False,
            batch_size=1,
            num_workers=self.args.num_workers,
            prefetch_factor=1,
        )

        mil_loss = MILLoss(
            self.args.alpha,
            self.args.beta,
            nn.BCEWithLogitsLoss,
        )

        model = MILModel(
            mil_loss,
            self.args.encoder,
            pretrained=True,
            freeze_epochs=self.args.freeze_epochs,
            learning_rate=self.args.lr,
        )

        print(model)

        trainer = Trainer(
            gpus=1,
            max_epochs=self.args.epochs,
            deterministic=True,
            checkpoint_callback=False,
            auto_lr_find=True,
            logger=loggers.CSVLogger(str(out_dir / "log")),
        )

        if valid_set is None:
            trainer.fit(model, train_loader)
        else:
            trainer.fit(model, train_loader, valid_loader)

        torch.save(model.model, out_dir / f"model_{self.fold}.pth")
        train_df.to_csv(out_dir / f"train_data_{self.fold}.csv", index=False)
        valid_df.to_csv(out_dir / f"valid_data_{self.fold}.csv", index=False)


def get_patch_metadata(args: argparse.Namespace) -> pd.DataFrame:
    """Load the patch metadata and split it based on args.

    Parameters
    ----------
    args : argprase.Namespace
        Command line arguments.

    Returns
    -------
    metadata : pd.DataFrame
        Patch-wise metadata, were the column 'patch_path' holds a list of
        patch_paths.

    """
    metadata = data_splitting.patch_metadata_from_dir(
        args.patch_root,
        args.magnification,
        exclude_sources=args.exclude_sources,
    )

    data_splitting.split_data(
        metadata,
        valid_split=args.valid_split,
        cv_folds=args.cv_folds,
        train_sources=args.train_sources,
        valid_sources=args.valid_sources,
        seed=args.rng_seed,
    )

    return metadata


def run_experiment(args: argparse.Namespace):
    """Run experiment."""
    metadata = get_patch_metadata(args)
    labels = ["normal", "coeliac"]
    metadata = metadata.loc[metadata.label.isin(labels)].reset_index(drop=True)
    add_n_hot_encoding(metadata, labels)

    experiment = MILExperiment(args)
    experiment.train(metadata)

    torch.save(labels, f"{args.save_dir}/classes.pth")
    plot_losses(args.save_dir)


if __name__ == "__main__":
    command_line_arguments = process_command_line_args()
    seed_everything(command_line_arguments.rng_seed)
    run_experiment(command_line_arguments)
