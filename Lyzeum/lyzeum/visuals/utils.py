"""Helper function for plotting."""
from typing import Optional
from matplotlib.axes import Axes
from numpy import array, diff, ndarray


def set_axis_aspect(axis: Axes, ratio: float = 1.0) -> None:
    """Set the 'on-screen' aspect ratio of `axis`.

    Parameters
    ----------
    axis : Axes
        The axis whose aspect ratio should be set.
    ratio : float
        The ratio to use: width / height

    """
    width = abs(diff(array(axis.get_xlim())))
    height = abs(diff(array(axis.get_ylim())))

    axis.set_aspect((width / height) / ratio)


def set_xlims_and_ticks(
    axis: Axes,
    left: Optional[float] = None,
    right: Optional[float] = None,
    major: Optional[ndarray] = None,
) -> None:
    """Set the x-limits and ticks of `axis`.

    Parameters
    -----------
    axis : Axes
        The axis being formatted.
    left : float, optional
        The left-hand x-limit.
    right : float, optional
        The right-hand x-limit.
    major : ndarray, optional
        Array of x_ticks to use.

    """
    if major is not None:
        axis.set_xticks(major)

    axis.set_xlim(left=left, right=right)


def set_ylims_and_ticks(
    axis: Axes,
    bottom: Optional[float] = None,
    top: Optional[float] = None,
    major: Optional[ndarray] = None,
) -> None:
    """Set the y-limits and ticks of `axis`.

    Parameters
    ----------
    axis : Axes
        The axis to be formatted.
    bottom : float, optional
        The bottom y-limit.
    top : float, optional
        The top y-limit.
    major : ndarray, optional
        Array of y_ticks to use.

    """
    if major is not None:
        axis.set_yticks(major)

    axis.set_ylim(bottom=bottom, top=top)
