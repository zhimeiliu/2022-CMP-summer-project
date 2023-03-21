"""Prediction distribution plotter."""
from typing import Any, Union, List, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import numpy as np
from numpy import ndarray


import pandas as pd


from .base_vis import BaseVis


class PredDistPlotter(BaseVis):
    """Class for plotting patient's prediction distribution.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Data frame holding the predictions for each patch and the
        corresponsing metadata.
    pred_keys : list of str
        List of the prediction keys to use. For multi-class problems you
        may wish to pass, say, `["pred_normal", "pred_coeliac"]`.
    bin_width : float, optional
        Bin width to use when computing the prediction distributions.

    """

    def __init__(
        self,
        pred_keys: Union[str, List[str]],
        bin_width: float = 0.05,
        dpi: int = 500,
    ):
        """Construct PredDistPLotter."""
        super().__init__(dpi=dpi)
        self.bin_width = bin_width
        self.pred_keys = self._process_pred_keys(pred_keys)

    _colours = (
        np.array(
            [
                [231, 225, 239],
                [221, 28, 119],
                [201, 148, 199],
            ]
        )
        / 255.0
    )

    def _histogram_predictions(
        self,
        predictions: pd.Series,
    ) -> Tuple[ndarray, ndarray]:
        """Compute a 1D histogram of the predictions.

        Parameters
        ----------
        predictions : pd.Series
            Predictions whose distribution we wish to obtain.

        Returns
        -------
        prob_density : ndarray
            The prediction distribution in unit of probability density.
        bin_middles : ndarray
            The bin middles to plot `prob_density` against.

        """
        edges = np.arange(0, 1 + self.bin_width, self.bin_width)
        bin_middles = edges[1:] - 0.5 * self.bin_width
        prob_density, _ = np.histogram(predictions, bins=edges, density=True)
        return prob_density, bin_middles

    def produce_plot(self, data_frame, file_name):
        """Plot the prediction distribution for a single patient."""
        scale = 1.6
        figure, axes = plt.subplots(
            1,
            len(self.pred_keys),
            figsize=(len(self.pred_keys) * scale, scale),
        )

        letters = iter(["(a) --- ", "(b) --- ", "(c) --- "])
        for idx, key in enumerate(self.pred_keys):
            density, middles = self._histogram_predictions(data_frame[key])
            axes[idx].bar(
                middles,
                density,
                width=self.bin_width,
                color=self._colours[idx],
                edgecolor="k",
                lw=0.5,
            )

            if idx == 0:
                axes[idx].set_ylabel(r"Probability density", labelpad=0.0)
            axes[idx].set_xlabel(r"Prediction", labelpad=0.0)

            axes[idx].text(
                0.1,
                0.875,
                next(letters) + key.capitalize(),
                transform=axes[idx].transAxes,
            )

            # Do this after adding labels
            self._format_single_axis(axes[idx])
            figure.tight_layout(pad=0.6)
            self._save_fig(figure, Path(file_name))
        self._figure_cleanup(axes)

    @staticmethod
    def _format_single_axis(axis: Axes) -> None:
        """Format `axis`."""
        axis.set_xticks(np.linspace(0, 1, 6))
        axis.set_xlim(
            left=axis.get_xticks().min(),
            right=axis.get_xticks().max(),
        )
        axis.set_ylim(bottom=0, top=20)
        aspect = abs(np.diff(axis.get_xlim())) / abs(np.diff(axis.get_ylim()))
        axis.set_aspect(aspect)
