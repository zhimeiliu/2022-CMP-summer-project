"""Module to house the base visualisation class."""
from typing import Union, List, Any
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.pyplot import Figure

from matplotlib import checkdep_usetex


plt.style.use((Path(__file__).parent / Path("matplotlibrc")).resolve())
plt.switch_backend("agg")

plt.rcParams.update({"text.usetex": checkdep_usetex(True)})


# pylint: disable=too-few-public-methods


class BaseVis:
    """Base class for creating prediction visualisations.

    Parameters
    ----------
    dpi : int, optional
        The dpi to save the plot with.


    """

    def __init__(self, dpi: int = 500):
        """Construct BaseVisualiser."""
        self.dpi = dpi

    def _save_fig(
        self,
        fig: Figure,
        file_name: Path,
    ):
        """Save `fig` to file."""
        self._create_dir(Path(file_name).parent)
        fig.savefig(file_name, dpi=self.dpi)

    @staticmethod
    def _create_dir(directory: Union[str, Path]):
        """Create the `directory` if it doesn't already exist."""
        Path(directory).mkdir(exist_ok=True)

    @staticmethod
    def _figure_cleanup(axes: Union[np.ndarray, Axes]):
        """Close open figures and clean the axes.

        Parameters
        ----------
        axes : ndarray
            Array of matplotlib axes.

        """
        if isinstance(axes, np.ndarray):
            for axis in axes.ravel():
                axis.cla()
                axis.remove()
        elif isinstance(axes, Axes):
            axes.cla()
            axes.remove()
        else:
            msg = f"axes should be ndarray or Axes. Got {type(axes)}"
            raise TypeError(msg)

        plt.close("all")

    @staticmethod
    def _is_string(input_str: Any) -> bool:
        """Check if `input_str` is a string or not."""
        return isinstance(input_str, str)

    def _is_list_of_string(self, str_list: Any):
        """Check if `str_list` is a list of str."""
        is_list = isinstance(str_list, list)
        all_str = all(map(self._is_string, str_list))
        return is_list and all_str

    def _process_pred_keys(
        self,
        pred_keys: Union[str, List[str]],
    ) -> List[str]:
        """Process the `pred_keys` argument.

        Parameters
        ----------
        pred_keys : str or list of str
            The predicition keys we wish to visualise.

        Returns
        -------
        list of str
            List of the prediction keys passed by the user.

        Raises
        ------
        TypeError
            If the items in `pred_keys` are not str or list of str.

        """
        key_list: List[str]
        if self._is_string(pred_keys):
            key_list = [str(pred_keys)]
        elif self._is_list_of_string(pred_keys):
            key_list = list(pred_keys)
        else:
            msg = "pred_keys should be str or list of str. "
            msg += f"Got {type(pred_keys)}."
            raise TypeError(msg)
        return key_list
