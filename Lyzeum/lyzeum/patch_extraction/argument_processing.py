"""Process the arguments required by PatchExtractor."""
import os
from pathlib import Path
from typing import Tuple, Union, Optional, Dict
from shutil import which

from numpy import isnan


def process_patch_size(patch_size: int) -> int:
    """Process the `patch_size` argument to `PatchExtractor`.

    Parameters
    ----------
    patch_size : int
        Size of the patches we wish to extract.

    Returns
    -------
    patch_size
        See parameters.

    Raises
    ------
    TypeError
        If `patch_size` is not an integer.
    ValueError
        If `patch_size` is less than 2.

    """
    if not isinstance(patch_size, int):
        raise TypeError(f"patch_size must be an int, not {type(patch_size)}")
    if patch_size < 2:
        msg = f"Patch size must be >= 2, got {patch_size}"
        raise ValueError(msg)
    return patch_size


def process_stride(stride: int) -> int:
    """Process `stride` argument to `PatchExtractor`.

    Parameters
    ----------
    stride : int
        The stride to use when generating patches using a sliding window.

    Returns
    -------
    stride : int
        See parameters.


    Raises
    ------
    TypeError
        If the `stride` is not an integer.
    ValueError
        If the `stride` is not at least 1.

    """
    if not isinstance(stride, int):
        raise TypeError(f"stride must be a integer, not {type(stride)}")
    if stride < 1:
        raise ValueError(f"stride must be at least 1, not {stride}")
    return stride


def process_mag_var(mag: float) -> float:
    """Check `mag` is an acceptable type and value.

    Parameters
    ----------
    mag : float
        Magnification argument to be checked.

    Returns
    -------
    mag : float
        See parameters.

    Raises
    ------
    TypeError
        If `mag` is not a float.
    ValueError
        If `mag` is < 1.0

    """
    if not isinstance(mag, float):
        msg = f"Magnification var {mag} should be float, got {type(mag)}."
        raise TypeError(msg)

    if mag < 1.0:
        raise ValueError(f"Magnification must be >= 1.0, got {mag}.")

    return mag


def process_min_max_mags(min_mag: float, max_mag: float) -> Tuple[float, float]:
    """Check the requested min and max mags are compatible.

    Parameters
    ----------
    min_mag : float
        Minimum user-requested magnifcation for the patches.
    max_mag : float
        Maximum user-requested magnifcation for the patches.

    Returns
    -------
    min_mag : float
        See parameters.
    max_mag : float
        See parameters.

    Raises
    ------
    ValueError
        If `min_mag` exceeds `max_mag`.

    """
    process_mag_var(min_mag)
    process_mag_var(max_mag)

    if min_mag > max_mag:
        msg = "min_patch_magnification exceeds max_patch_magnification. "
        msg += f"Got min = {min_mag} and max = {max_mag}."
        raise ValueError(msg)

    return min_mag, max_mag


def process_background_tol(background_tol: float) -> float:
    """Process the `background_tol` argument.

    Parameters
    ----------
    background_tol : float
        The background tolerance to use (fraction of total number of pixels.)

    Returns
    -------
    background_tol : float
        See parameters.

    Raises
    ------
    stuff


    """
    if not isinstance(background_tol, float):
        msg = f"background_tol should be a float, got {type(background_tol)}"
        raise TypeError(msg)

    if not 0 <= background_tol <= 1:
        msg = f"background_tol should be on (0, 1), got {background_tol}"
        raise ValueError(msg)

    return background_tol


def process_qupath_binary(qupath_in: Union[str, Path]) -> Path:
    """Check the `qupath_binary` exists and is executable.

    Parameters
    ----------
    qupath_in : str or Path
        Path to the executable qupath binary file.

    Returns
    -------
    qupath_binary : str or Path
        Path(qupath_in).

    Raises
    ------
    TypeError
        If `qupath` in is not a Path or str.
    FileNotFoundError
        If the qupath binary has not been found or is not executable.

    """
    if not isinstance(qupath_in, (Path, str)):
        msg = f"Qupath binary should be Path or str, got {type(qupath_in)}"
        raise TypeError(msg)

    qupath_binary = which(qupath_in)
    if qupath_binary is None:
        msg = f"Qupath file {qupath_in} not found, or isn't executable."
        raise FileNotFoundError(msg)
    return Path(qupath_binary)


def check_path_like(path_like: Union[str, Path]):
    """Make sure `path_like` is a str or a path.

    Parameters
    ----------
    path_like : str or Path
        Varible to check is path-like.

    Returns
    -------
    Path
        Path(path_like)

    Raises
    ------
    TypeError
        If `path_like` is not a str or a Path.


    """
    if not isinstance(path_like, (str, Path)):
        msg = f"path_like '{path_like}' should be str or Path, got "
        msg += f"{type(path_like)}"
        raise TypeError(msg)


def process_cleanup_workers(requested_workers: Optional[int] = None) -> int:
    """Determine a safe number of workers to use in the cleanup process.

    Parameters
    ----------
    requested_workers : int
        Number of workers the user requests for the cleanup process.

    Returns
    -------
    cleanup_workers : int
        The minimum between the requested number of workers and the maximum
        number of workers the process maye safely use.

    Raises
    ------
    TypeError
        If cleanup workers is not an int or None.
    ValueError
        If cleanup workers is not a positive integer.

    """
    if not isinstance(requested_workers, (int, type(None))):
        msg = "cleanup_workers should be int or None, got"
        msg += f" {type(requested_workers)}."
        raise TypeError(msg)

    # pid = os.getpid()
    # max_possible = len(os.sched_getaffinity(pid))

    # cleanup_workers = (
    #     requested_workers if requested_workers is not None else max_possible
    # )
    cleanup_workers = requested_workers

    if cleanup_workers < 1:
        msg = f"cleanup_workers should exceed 0, got {cleanup_workers}."
        raise ValueError(msg)

    # return min(cleanup_workers, max_possible)
    return cleanup_workers


def process_zip_patches(zip_patches: bool) -> bool:
    """Process the `zip_patches` argument of `PatchExtractor`.

    Parameters
    ----------
    zip_patches : bool
        Bool determining if the patch directory is zipped or not.

    Raises
    ------
    TypeError
        If `zip_patches` is not a bool.

    """
    if not isinstance(zip_patches, bool):
        msg = f"zip_patches arg should be bool. Got '{type(zip_patches)}.'"
        raise TypeError(msg)
    return zip_patches


def process_wsi_arg(wsi_in: Union[str, Path]) -> Path:
    """Process `wsi_in` argument given to `PatchExtractor` when called.

    Parameters
    ----------
    wsi_in : str or Path
        Path to the whole slide image file to be processed.

    Returns
    -------
    wsi : Path
        Absolute path to the wsi file.

    Raises
    ------
    FileNotFoundError
        If path-like `wsi_in` does not exist.
    IsADirectoryError
        If `wsi_in` leads to a directory (and not a file).

    """
    check_path_like(wsi_in)
    wsi = Path(wsi_in).resolve()

    if not wsi.exists():
        raise FileNotFoundError(f"wsi path {wsi} does not exist.")
    if not wsi.is_file():
        raise IsADirectoryError(f"{wsi} is a directory. Should be file.")

    return wsi


def process_parent_dir_arg(parent_dir: Union[str, Path, None]) -> Path:
    """Process the parent_dir argument."""
    parent_dir = parent_dir if parent_dir is not None else ""
    check_path_like(parent_dir)

    parent_dir = str(parent_dir).replace(",", ";")
    parent_dir = parent_dir.replace("'", "")
    return Path(parent_dir)


def process_region_arg(region: Union[Dict[str, int], None]) -> Dict[str, int]:
    """Check the region dictionary is of the correct form.

    Parameters
    ----------
    region : dict[str, int], optional
        The region dictionary specifying where to look on the WSI.

    Raises
    ------
    TypeError
        If `region` is not a dict.
    TypeError
        If the keys in `region` are not all str.
    TypError
        If the region values are not all int.
    ValueError
        If the keys in `region` don't match `expected_keys`.
    ValueError
        If any of the values in `region` are less than zero.

    """
    expected_keys = sorted(["left", "top", "width", "height"])

    if region is None:
        return dict(zip(expected_keys, [0, 0, 0, 0]))

    if not isinstance(region, dict):
        raise TypeError(f"Region should be a dict, got '{type(region)}'.")

    if not all(map(lambda x: isinstance(x, str), region.keys())):
        msg = f"region keys should be str, got '{region.keys()}'."
        raise TypeError(msg)

    if not all(map(lambda x: isinstance(x, int), region.values())):
        msg = f"region values should be int. Got {region.values()}"
        raise TypeError(msg)

    if not sorted(expected_keys) == sorted(list(region.keys())):
        msg = f"Keys should be '{expected_keys}', got '{region.keys()}'."
        raise ValueError(msg)

    if any(map(lambda x: x < 0, region.values())):
        msg = f"region values cannot be negative. Got {region.values()}."
        raise ValueError(msg)

    if (width := region["width"]) == 0 and (height := region["height"]) != 0:
        msg = "Width and height should both be zero, or both non-zero. "
        msg += f"Got width {width} and height {height}."
        raise ValueError(msg)

    if (width := region["width"]) != 0 and (height := region["height"]) == 0:
        msg = "Width and height should both be zero, or both non-zero. "
        msg += f"Got width {width} and height {height}."
        raise ValueError(msg)

    return region


def process_user_supplied_mag(mag: Union[float, None]) -> Union[float, None]:
    """Process the user_supplied_mag variable.

    Parameters
    ----------
    mag : float or None
        WSI magnification supplied by the user.

    Raises
    ------
    TypeError
        If `mag` is not a float or None.

    """
    if not isinstance(mag, (float, type(None))):
        msg = "user_supplied_mag should be a float or None. "
        msg += f"Got {type(mag)}"
        raise TypeError(msg)

    if isinstance(mag, float) and isnan(mag):
        mag = None

    if isinstance(mag, float):
        process_mag_var(mag)
    return mag


def process_generate_patches_arg(generate_patches: bool) -> bool:
    """Process the `in_bool` arg.

    Parameters
    ----------
    generate_patches : bool
        Variable determining whether patches should be generated or not.

    Raises
    ------
    TypeError
        If `generate_patches` is not a bool.

    """
    if not isinstance(generate_patches, bool):
        msg = "generate_patches arg should be bool. Got type "
        msg += f"'{type(generate_patches)}'."
        raise TypeError(msg)
    return generate_patches
