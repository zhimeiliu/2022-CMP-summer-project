# #!/usr/bin/env python3
"""Module with usefull tool for visualising experimental predictions."""
import os
from typing import Union, Tuple, List, Any
from pathlib import Path
import regex

import pandas as pd

import numpy as np
from numpy import ndarray

from skimage import io
from sklearn.metrics import roc_curve, roc_auc_score

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure


from matplotlib import checkdep_usetex
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from .base_vis import BaseVis

plt.style.use((Path(__file__).parent / Path("matplotlibrc")).resolve())
plt.switch_backend("agg")

plt.rcParams.update({"text.usetex": checkdep_usetex(True)})


def plot_losses(out_dir: Union[Path, str]) -> None:
    """Plot the training and validation."""
    fig, axes = plt.subplots(1, 2, figsize=(3.375, 1.8))
    log_csv_iter = sorted(Path(out_dir).glob("log/default/*/metrics.csv"))

    folds = len(log_csv_iter)

    for fold, path in enumerate(log_csv_iter):
        log = pd.DataFrame(pd.read_csv(path))

        axes[0].plot(
            np.arange(log.epoch.max() + 1),
            log.train_loss.dropna(),
            "--o",
            ms=3,
            label=f"Fold {fold}" if folds > 1 else "__nolegend__",
        )

        if "valid_loss" in log:

            axes[1].plot(
                np.arange(log.epoch.max() + 1),
                log.valid_loss.dropna(),
                "--o",
                ms=3,
                label=f"Fold {fold}" if folds > 1 else "__nolegend__",
            )

    y_labels = iter(["Training loss", "Validation loss"])
    for axis in axes.ravel():
        axis.legend()
        axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Epochs")
        axis.set_ylabel(next(y_labels))

    fig.tight_layout(pad=0.1, w_pad=0.3)
    fig.savefig(f"{out_dir}/loss.pdf", dpi=1000)
