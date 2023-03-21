#!/usr/bin/env python
"""Script for running inference using a pretrained pytorch model."""
from typing import List, Union
import argparse
from argparse import Namespace

from pathlib import Path


import pandas as pd
from pandas import DataFrame
import torch

from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import Module

from numpy import ndarray


from lyzeum.experiment_tools.data_splitting import patch_metadata_from_dir
from lyzeum.torch_toolbelt.datasets import ImageDataset
from lyzeum.torch_toolbelt.image_transforms import get_img_transforms
from lyzeum.misc import list_zipfile_contents

from lyzeum.visuals import PredDistPlotter, PredictionViewer, threshold_analysis_plots
from lyzeum.scoring.threshold_analysis import ThresholdFinder, PredictionScorer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _parse_arguments() -> Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with an existing model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "model_dir",
        type=str,
        help="The directory holding the model's output.",
    )

    parser.add_argument(
        "patch_parent_dir",
        type=str,
        help="Top directory the inference patches are saved in.",
    )

    parser.add_argument(
        "---infer_csv",
        type=str,
        help="Path to a csv file of scan metdata to do inference on."
        + "If None, the default behaviour is to look for and use"
        + " files containing 'valid_data' in the 'model-dir' arg.",
        default=None,
    )

    parser.add_argument(
        "--validate",
        type=bool,
        help="If validating, the thresholds will be calculated from the data",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--do-inference",
        type=bool,
        help="Should we do inference on each scan? Tries to load otherwise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--produce-plots",
        type=bool,
        help="Should we produce plots?",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--mag",
        type=float,
        help="Magnification to take patches at.",
        default="10.0",
    )

    parser.add_argument(
        "--infer-out-dir",
        type=str,
        help="Directory to save the inference results in.",
        default="inference_out_dir",
    )

    parser.add_argument(
        "--bs",
        type=int,
        help="Batch size to do inference with.",
        default=100,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of dataloader workers to use.",
        default=6,
    )

    parser.add_argument(
        "--means-csv",
        type=str,
        help="File name of the csv the scan means will be output in.",
        default="scan_means.csv",
    )

    return parser.parse_args()


def _process_inference_csv(
    infer_csv: Union[Path, None],
    model_dir: Path,
) -> DataFrame:
    """Process the csv file to use for inference.

    Parameters
    ----------
    infer_csv : Union[Path, None]
        The path to the csv file to use for inference. If None, we search the
        model directory for csv files containing 'valid_data' and use those.
    model_dir : Path
        Path to the folder the model is saved in.

    Returns
    -------
    inference_df : DataFrame
        DataFrame containing the inference scan metadata.

    """
    if infer_csv is None:
        files = filter(
            lambda x: "valid_data" in str(x),
            model_dir.glob("*.csv"),
        )

        data = pd.concat(
            list(map(pd.read_csv, files)),
            axis=0,
            ignore_index=True,
        )
    elif isinstance(infer_csv, (str, Path)):
        data = pd.read_csv(infer_csv)
    else:
        raise RuntimeError("No inference csv supplied.")

    if "fold" not in data.keys():
        data["fold"] = 1

    return data


def _patch_metadata_from_scan_df(scan_df: DataFrame) -> DataFrame:
    """Get patch-level metadata from `scan_df`.

    The patches live in a zip file, so to access them, we must retrieve
    their filemes from the zip.

    Parameters
    ----------
    scan_df : DataFrame
        DataFrame holding scan-level metadata.

    Returns
    -------
    DataFrame
        Patch-level metadata.

    """
    patch_df = scan_df.copy()
    patch_df["patch_path"] = patch_df.zip_file.apply(list_zipfile_contents)
    return patch_df.explode("patch_path").reset_index(drop=True)


@torch.no_grad()
def _predict_on_single_scan(
    model: Module,
    scan_df: pd.DataFrame,
    batch_size: int,
    workers: int,
) -> ndarray:
    """Use `model` to infer on a single scan.

    Parameters
    ----------
    model : Module
        A trained pytorch model which returns logits.
    scan_df : pd.DataFrame
        A data frame holding the patch metadata for a single scan.

    Returns
    -------
    Tensor
        The model's predictions for each patch in the scan.

    """
    assert model.training is False, "The model is not in training mode!!!"

    prediction_list: List[Tensor]
    prediction_list = []

    dataset = ImageDataset(
        scan_df.patch_path.to_list(),
        get_img_transforms(False),
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )

    for batch in data_loader:
        prediction_list.append(model(batch.to(DEVICE)).cpu())

    return torch.cat(tuple(prediction_list), dim=0).sigmoid().numpy()


def _save_single_scan_means(
    patch_pred_path: Path,
    keys: List[str],
    save_path: Path,
) -> None:
    """Save the mean predictions."""
    pred_df = pd.read_csv(patch_pred_path)
    means = list(map(lambda x: pred_df[x].mean(), keys))
    mean_df = pred_df.drop_duplicates(subset="scan").reset_index(drop=True)
    mean_df[keys] = means

    save_path.parent.mkdir(exist_ok=True)

    if not save_path.exists():
        mean_df.to_csv(save_path, index=False)
    else:
        mean_df.to_csv(save_path, index=False, mode="a", header=False)


def _save_single_scan_predictions(
    model: Module,
    keys: List[str],
    scan_df: DataFrame,
    args: Namespace,
    csv_path: Path,
):
    """Save the predictions for a single scan."""
    if not args.do_inference:
        return

    scan_df[keys] = _predict_on_single_scan(
        model,
        scan_df,
        args.bs,
        args.num_workers,
    )

    csv_path.parent.mkdir(exist_ok=True, parents=True)
    scan_df.to_csv(csv_path, index=False)


def _process_single_scan(
    args: Namespace,
    model: Module,
    case_id: str,
    scan: str,
    scan_df: DataFrame,
):
    """Infer, save and make plots for a single scan.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.
    model : Module
        Pytorch model to infer with.
    case_id : str
        Unique id of the case.
    scan : str
        The name of the scan.
    scan_df : DataFrame
        `DataFrame` holding the patch-wise metadata for the scan.

    """
    keys = torch.load(Path(args.model_dir) / "classes.pth")
    csv_path = Path(args.infer_out_dir, "predictions", case_id, scan + ".csv")

    _save_single_scan_predictions(
        model,
        keys,
        scan_df,
        args,
        csv_path,
    )

    _save_single_scan_means(
        csv_path,
        keys,
        Path(args.infer_out_dir, args.means_csv),
    )

    _single_scan_plots(csv_path, keys, args, case_id, scan)


def _single_scan_plots(
    pred_csv_path: Path,
    pred_keys: List[str],
    args: Namespace,
    case_id,
    scan,
) -> None:
    """Produce the plots for a single scan.

    Parameters
    ----------
    pred_csv_path : Path
        Path the csv file holding the predictions for the scan
    pred_keys : List[str]
        The prediction column headins in the csv at `pred_csv_path`.
    args : Namespace
        The command-line arguments.
    case_id : str
        The case_id associated with the scan.
    scan : str
        The file name of the scan.

    """
    if not args.produce_plots:
        return

    scan_df = pd.read_csv(pred_csv_path)

    label = scan_df.label.unique()[0]
    file_name = f"{label}---{case_id}---{scan}.pdf"

    dist_plotter = PredDistPlotter(pred_keys)
    dist_plotter.produce_plot(
        scan_df,
        Path(args.infer_out_dir) / "dists" / file_name,
    )

    heatmapper = PredictionViewer(pred_key="coeliac")
    heatmapper.produce_plot(
        scan_df,
        Path(args.infer_out_dir) / "heatmaps" / file_name,
    )


def _score_predictions(model_dir: Path, predictions: DataFrame):
    """Score the model's predictions."""
    decisions, score_data = [], []
    threshold_dict = torch.load(str(model_dir / "thresholds.pth"))
    labels = list(threshold_dict.keys())
    scorer = PredictionScorer(threshold_dict)

    if "valid_fold" not in predictions.keys():
        predictions["valid_fold"] = 0

    for fold, pred_df in predictions.groupby(by="valid_fold"):
        score_dict, preds = scorer.score_predictions(pred_df, labels)
        score_df = pd.DataFrame(score_dict).T.reset_index()
        score_df["fold"] = fold
        score_df = score_df.rename(columns={"index": "label"})
        score_data.append(score_df)
        decisions.append(preds)

    results = pd.concat(score_data, ignore_index=True)
    for label in labels:
        print(results.loc[results.label == label])
        print(results.loc[results.label == label].set_index("fold").mean())

    decided_df = pd.concat(decisions, axis=0, ignore_index=True)

    print(
        decided_df.loc[
            decided_df.decision_coeliac != decided_df.truth_coeliac,
            ["case_id", "scan", "label"],
        ].sort_values(by="case_id")
    )


def _determine_thresholds(
    determine_thresholds: bool,
    predictions: DataFrame,
    model_dir: Path,
):
    """Determine the decision-making thresholds.

    Parameters
    ----------
    determine_thresholds : bool
        If True, the thresholds are determined from the data, and saved in the
        model's output directory. If False, the thresholds are loaded from the
        model's output directory.
    predictions : DataFrame
        The scan-level predictions.
    model_dir : Path
        The directory the model is saved in.

    """
    if determine_thresholds is True:
        labels = torch.load(str(model_dir / "classes.pth"))
        finder = ThresholdFinder()
        finder.set_best_thresholds(predictions, labels)
        threshold_dict = finder.return_thresholds()
        if (save_path := Path(model_dir, "thresholds.pth")).exists():
            raise FileExistsError(f"Thresholds already saved to {save_path}.")
        torch.save(threshold_dict, save_path)


def _load_model(file_path: Path) -> Module:
    """Load a Pytorch model using `torch.load`.

    Parameters
    ----------
    file_path : Path
        Where the model lives.

    Returns
    -------
    model : Module
        Our model, ready to go!

    """
    model = torch.load(file_path).to(DEVICE).eval().requires_grad_(False)
    assert model.training is False
    return model


def _infer_on_single_fold(
    args: Namespace,
    infer_df: DataFrame,
    fold: int,
):
    """Infer on a single cross-validation fold.

    Parameters
    -----------
    args : Namespace
        The command-line arguments.
    infer_df : DataFrame
        DataFrame holding the metadata for the scans we want to infer on.
    fold : int
        The cross-validation fold. If we are not cross-validating, set
        `fold == 1`.

    """
    model = _load_model(Path(args.model_dir) / f"model_{fold}.pth")

    local_patches = patch_metadata_from_dir(args.patch_parent_dir, args.mag)
    new_top_dir = local_patches.top_dir.unique()[0]
    infer_df["new_top_dir"] = new_top_dir

    infer_df.zip_file = infer_df.apply(
        lambda x: x.zip_file.replace(x.top_dir, x.new_top_dir), axis=1
    )

    for (case_id, scan), scan_df in infer_df.groupby(by=["case_id", "scan"]):

        infer_df = _patch_metadata_from_scan_df(scan_df)
        _process_single_scan(args, model, case_id, scan, infer_df)


def _overwrite_data_check(args: Namespace) -> None:
    """Check data won't be overwritten."""
    preds_exist = Path(args.infer_out_dir, "predictions").exists()

    if preds_exist and args.do_inference:
        raise FileExistsError("Predicitons already saved. Refusing overwrite.")


def _infer_on_all_scans(args: Namespace):
    """Do inference on a single scan."""
    data = _process_inference_csv(args.infer_csv, Path(args.model_dir))

    scan_means_path = Path(args.infer_out_dir, args.means_csv)

    _overwrite_data_check(args)

    for fold, data_frame in data.groupby(by="fold"):
        _infer_on_single_fold(args, data_frame, fold)

    mean_df = pd.read_csv(scan_means_path)

    threshold_analysis_plots(
        mean_df,
        torch.load(Path(args.model_dir, "classes.pth")),
        Path(args.infer_out_dir) / "roc.pdf",
    )

    _determine_thresholds(
        args.validate,
        mean_df,
        Path(args.model_dir),
    )

    _score_predictions(Path(args.model_dir), mean_df)


if __name__ == "__main__":
    parsed_cl = _parse_arguments()
    _infer_on_all_scans(parsed_cl)
