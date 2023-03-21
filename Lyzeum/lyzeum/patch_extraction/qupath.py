"""Qupath helper functions."""
from typing import Dict, Union
from pathlib import Path
import subprocess
import urllib.parse
import regex

from lyzeum.patch_extraction.exceptions import QupathException

PathLike = Union[str, Path]

# pylint: disable=too-many-arguments


def create_qupath_command(
    wsi: PathLike,
    tile_dir: PathLike,
    downscaling: float,
    patch_dir: int,
    stride: int,
    qupath: PathLike,
    patch_format: str,
    task: str,
    region: Dict[str, int],
) -> str:
    """Produce the command-line command to tile a WSI with qupath.

    Parameters
    ----------
    wsi : PathLike
        Path to the WSI in question.
    patch_dir : PathLike
        The directory in which the tiles should be saved.
    downscaling : int
        The level of downscaling to generate the tiles at.
    tile_size : int
        The side length (int pixels) of the square tiles.
    stride : int
        The stride, in pixels, of the sliding window used to generate the
        tiles.
    qupath : PathLike
        Path to the qupath binary file.
    patch_format : str
        Format to save the patches in, i.e. `png`.
    task : str
        The task you with qupath to perform: `"generate_patches"` or
        `"generate_overview"`.
    region : dict[str, int]
        A dictionary of keys `left`, `top`, `width` and `height` specifying
        the region of interest in the WSI. If None, the entire WSI is tiles.


    """
    if not patch_format.startswith("."):
        patch_format = "." + patch_format

    groovy_script = Path(__file__).parent / "wsi_patch_extractor.groovy"
    _check_qupath_task(task)

    command = f"{qupath} script {groovy_script}"
    arg_list = [
        _get_image_uri(wsi),
        downscaling,
        patch_dir,
        patch_dir - stride,
        tile_dir,
        patch_format,
        region["left"],
        region["top"],
        region["width"],
        region["height"],
        task,
    ]
    arg_string = ",".join([f"'{arg}'" for arg in arg_list])
    command += f" --args [{arg_string}]"
    return command


def _get_image_uri(wsi: PathLike) -> str:
    """Express `wsi` as a uri string."""
    return urllib.parse.quote(f"file://{wsi}")


def _check_qupath_task(task: str):
    """Raise an exception if the qupath task is not an option.

    Parameters
    ----------
    task : str
        The task qupath should perform: `"generate_overview" or
        "generate_patches"`.

    Raise
    -----
    ValueError
        If `task` is not one of the accepted options we raise a ValueError.

    """
    options = ["generate_patches", "generate_overview"]
    if task not in options:
        raise ValueError(f"{task} not in options: {options}")


def get_wsi_mag(wsi: PathLike, qupath: PathLike) -> float:
    """Return the magnification of the slide `wsi`.

    Parameters
    ----------
    wsi : path_like
        Path to the whole slide image we are dealing with.
    qupath : path_like
        Path to the qupath executable binary.

    Returns
    -------
    float
        The estimated magnification of the slide.

    """
    groovy_script = Path(__file__).parent / "get_magnification.groovy"
    qupath_command = f"{qupath} script {groovy_script}"
    qupath_command += f" --args {_get_image_uri(wsi)}"

    output = run_command(qupath_command, return_stdout=True)
    pattern = "slide magnification<<<(.*?)>>>slide magnification"
    return float(regex.search(pattern, output).group(1))


def run_command(
    qupath_command: str,
    return_stdout: bool = False,
) -> Union[str, None]:
    """Run the qupath command."""
    completed_process = subprocess.run(
        qupath_command,
        shell=True,
        check=False,
        capture_output=True,
    )
    if completed_process.returncode != 0:
        print(completed_process.stdout, "\n\n")
        print(completed_process.stderr, "\n\n")
        msg = f"Quapth error running command:\n{qupath_command}"
        raise QupathException(msg)
    if return_stdout is True:
        return str(completed_process.stdout)
    return None
