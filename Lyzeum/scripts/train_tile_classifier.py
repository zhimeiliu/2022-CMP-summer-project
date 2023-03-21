"""Classification epxeriment training code."""
import argparse
from typing import List
from pathlib import Path

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything, loggers

from sklearn.metrics import roc_auc_score, roc_curve

from experiments import data_splitting
from experiments.data_splitting import add_n_hot_encoding

# from experiments.data_splitting import separate_splits
from experiments.experiment import create_argument_parser, ExperimentRunner
from experiments.prediction_visuals import plot_losses
from experiments.prediction_visuals import plot_patient_level_roc

# from experiments.prediction_visuals import PredDistPlotter
# from experiments.prediction_visuals import PatchPlotter
# from experiments.prediction_visuals import PredictionViewer
# from experiments.utils.normalise import get_channel_statistics
from experiments.utils import torch_datasets
from experiments.utils.networks import ClassifierCNN
from experiments.utils.image_transforms import get_img_transforms

# pylint: disable=arguments-differ, too-many-ancestors, unused-argument
# pylint: disable=too-many-arguments


def process_command_line_args() -> argparse.Namespace:
    """Process the command-line arguments.

    Returns
    -------
    argparse.Namespace
        Command-line arguments needed for a classification experiment.

    Notes
    -----
    Any bespoke arguments needed for this experiment can be added before
    parsing.

    """
    parser = create_argument_parser()

    parser.add_argument(
        "--save-dir",
        help="Directory to save the model output in",
        type=str,
        default="tile_classifier_output",
    )

    parser.add_argument(
        "--batch-size",
        help="Batch size to use in training",
        type=int,
        default=64,
    )

    return parser.parse_args()


class LightningClassifier(LightningModule):
    """Multiple instance learning model.

    Parameters
    ----------
    encoder : str
        String determining which encoder we should use. See
        experiments.utils.networks for options.
    loss_func : nn.BCEWithLogitsLoss
        Pytorch loss function.
    freeze_epochs : int, optional
        Number of epochs to train for with the body frozen.
    pretrained : bool, optional
        Should we endow the encoder with pytorch's pretrained weights.
    learning_rate : float, optional
        Learning rate to train with.

    """

    def __init__(
        self,
        encoder: str,
        loss_func,
        freeze_epochs: int = 0,
        pretrained: bool = True,
        learning_rate: float = 1e-4,
    ):
        """Construct model."""
        super().__init__()
        self.cnn_classifier = ClassifierCNN(
            encoder,
            2,
            pretrained=pretrained,
        )
        self.my_loss = loss_func
        self.learning_rate = learning_rate
        self.freeze_epochs = freeze_epochs

    def forward(self, batch):
        """Pass a batch of inputs through the network."""
        logits = self.cnn_classifier(
            batch,
            self.trainer.current_epoch < self.freeze_epochs,
        )
        return logits

    def training_step(self, batch, batch_idx):
        """Perform one training step."""
        self.train()
        img_batch, labels = batch
        preds = self.forward(img_batch)
        num_correct = (preds.detach().argmax(dim=1) == labels.argmax(dim=1)).sum()
        return {
            "loss": self.my_loss(preds, labels),
            "correct": num_correct,
            "total": len(preds),
        }

    def training_epoch_end(self, outputs):
        """End of training epoch."""
        avg_loss = torch.Tensor([x["loss"] for x in outputs]).mean()
        count = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])
        self.log("train_loss", avg_loss, prog_bar=True)
        self.log("train_acc", count / total, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        """Perform one validation step."""
        self.eval()
        img_batch, labels = batch
        with torch.no_grad():
            preds = self.forward(img_batch)
            num_correct = (preds.argmax(dim=1) == labels.argmax(dim=1)).sum()
            return {
                "valid_loss": self.my_loss(preds, labels),
                "correct": num_correct,
                "total": len(preds),
            }

    def validation_epoch_end(self, outputs):
        """End of epoch validation process."""
        avg_loss = torch.tensor([x["valid_loss"] for x in outputs]).mean()
        count = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])
        self.log("valid_loss", avg_loss, prog_bar=True)
        self.log("valid_acc", count / total, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        """Do inference on a batch."""
        self.eval()
        with torch.no_grad():
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
        #     gamma=0.1,
        #     verbose=False,
        # )
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optim,
            lambda epoch: 0.85,
        )
        return {
            "optimizer": optim,
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }


class TileClfExperiment(ExperimentRunner):
    """Class for training the simple tile classifier.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    """

    def __init__(self, args):
        """Construct the tile classifier training class."""
        super().__init__(args)
        self.model_class = LightningClassifier

    def train_on_single_split(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
    ):
        """Train an experiment using the given train-valid split."""
        train_set = torch_datasets.SimpleImageDataset(
            train_df.patch_path,
            train_df.target,
            image_transforms=get_img_transforms(True),
        )

        valid_set = torch_datasets.SimpleImageDataset(
            valid_df.patch_path,
            valid_df.target,
            image_transforms=get_img_transforms(False),
        )

        train_loader, valid_loader = (
            DataLoader(
                train_set,
                shuffle=True,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                prefetch_factor=1,
            ),
            DataLoader(
                valid_set,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                prefetch_factor=1,
            ),
        )

        infer_loader = DataLoader(
            torch_datasets.InferImgDataset(
                valid_df.patch_path.to_list(),
            ),
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            prefetch_factor=1,
        )

        model = self.model_class(
            self.args.encoder,
            torch.nn.BCEWithLogitsLoss(),
            freeze_epochs=0,
        )

        print(model)

        trainer = Trainer(
            gpus=1,
            max_epochs=self.args.epochs,
            deterministic=True,
            checkpoint_callback=False,
            auto_lr_find=True,
            logger=loggers.CSVLogger(self.args.save_dir, "log"),
            # limit_train_batches=0.01,
            # limit_val_batches=0.01,
        )
        # trainer.tune(model, train_loader, valid_loader)
        trainer.fit(model, train_loader, valid_loader)
        preds = torch.cat(trainer.predict(model, infer_loader))

        valid_df[["pred_normal", "pred_coeliac"]] = preds.sigmoid().numpy()

        pred_path = Path(self.args.save_dir, "preds.csv")
        if pred_path.exists():
            valid_df.to_csv(pred_path, mode="a", header=False, index=False)
        else:
            valid_df.to_csv(pred_path, index=False)


def load_patch_metadata(args: argparse.Namespace) -> pd.DataFrame:
    """Load the patch metadata and split it based on requested args."""
    metadata = data_splitting.patch_metadata_from_dir(
        args.patch_root,
        args.downscaling,
        exclude_sources=args.exclude_sources,
    )
    data_splitting.split_data(
        metadata,
        valid_split=args.valid_split,
        cv_folds=args.cv_folds,
        train_sources=args.train_sources,
        valid_sources=args.valid_sources,
    )
    return metadata


def patient_level_accuracy(patch_predictions: pd.DataFrame):
    """Compute the patient-level accuracy.

    Parameters
    ----------
    patch_predictions : pd.DataFrame
        DataFrame holding the patch-level predictions.

    """
    if "valid_fold" not in patch_predictions.keys():
        patch_predictions["valid_fold"] = 1

    for _, fold_df in patch_predictions.groupby("valid_fold"):

        preds = fold_df[["patient", "pred_coeliac"]].groupby("patient").mean()

        targets = fold_df[["patient", "truth_coeliac"]].drop_duplicates(
            subset="patient"
        )

        merged = preds.merge(targets, on="patient")
        fpr, tpr, thresh = roc_curve(merged.truth_coeliac, merged.pred_coeliac)

        best_thresh = thresh[np.argmax(tpr - fpr)]
        best_thresh = 0.5
        merged["decision"] = merged.pred_coeliac.apply(
            lambda x: 0 if x < best_thresh else 1
        )

        accuracy = (merged.truth_coeliac == merged.decision).mean()
        roc_auc = roc_auc_score(merged.truth_coeliac, merged.pred_coeliac)

        print(f"Decision threshold: {best_thresh:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"ROC AUC: {roc_auc:.3f}")
        print("Incorrect patients:")
        print(merged.loc[merged.truth_coeliac != merged.decision, "patient"])
        print("\n\n\n")


def run_experiment(args: argparse.Namespace):
    """Run the experiment."""
    metadata = load_patch_metadata(args)
    diagnoses = ["normal", "coeliac"]
    metadata = metadata.loc[metadata.diagnosis.isin(diagnoses)].reset_index(drop=True)
    add_n_hot_encoding(metadata, diagnoses)

    # print(metadata.groupby(["source", "diagnosis", "split"]).patient.nunique())

    experiment = TileClfExperiment(args)
    # experiment.train(metadata)

    pred_df = pd.DataFrame(pd.read_csv(Path(args.save_dir, "preds.csv")))

    pred_df["pred"] = pred_df.pred_coeliac.to_numpy()
    pred_df["label"] = pred_df.diagnosis.to_numpy()

    patient_level_accuracy(pred_df)
    plot_losses(args.save_dir)
    plot_patient_level_roc(
        pred_df,
        "pred_coeliac",
        "truth_coeliac",
        args.save_dir,
    )
    # viewer = PredictionViewer(pred_df, Path(args.save_dir, "views"), 4, 224)
    # viewer.generate_all_plots()
    # dist_plotter = PredDistPlotter(pred_df, Path(args.save_dir, "dists"))
    # dist_plotter.generate_all_plots()
    # patch_plotter = PatchPlotter(
    #     pred_df,
    #     Path(args.save_dir, "patches"),
    #     num_patches=5,
    # )
    # # patch_plotter.generate_all_plots()


if __name__ == "__main__":
    arguments = process_command_line_args()
    seed_everything(arguments.rng_seed)
    run_experiment(arguments)
