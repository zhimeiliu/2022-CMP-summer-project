"""Tool for extracting patches from digital WSI scans."""
from typing import Union, Optional, Dict, List
from shutil import rmtree, make_archive
from pathlib import Path
from time import perf_counter
import multiprocessing as mp


from skimage import io
from skimage.filters import threshold_otsu

from PIL import Image

from lyzeum.patch_extraction import qupath
from lyzeum.patch_extraction.exceptions import MagnificationException
from lyzeum.patch_extraction.exceptions import QupathException
from lyzeum.patch_extraction import argument_processing as ap

PathLike = Union[str, Path]

Image.MAX_IMAGE_PIXELS = None

# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods


class PatchExtractor:
    """Callable object for extracting patches from Whole-Slide-Images (WSIs).

    #TODO: Update docs.

    In a given call we:

        — Attempt to determine the WSI's magnification using a QuPath
          subprocess.
        — Produce a thumbnail of the entire WSI (or requested region) by
          calling a QuPath subprocess.
        — Calculate Otsu's threshold using the thumbnail.
        — Generate patches by calling another QuPath subprocess.
        — Delete patches which contain too much background (determined using
          Otsu's threshold).
        — Optionally zip the directories containing patches to avoid huge
          numbers of files.

    Parameters
    ----------
    patch_size : int
        Size (in pixels) of the square patches to generate. Must be an integer greater than 2.
    stride : int, optional
        The stride (in pixels) of the sliding window used to generate patches.
        If None, the default stride is `int(0.75 * patch_size)`. If specified,
        `stride` must exceed 0.
    min_patch_magnification : float, optional
        Minum magnification to sample patches at. Must be greater than 1.
    max_patch_magnification : float, optional
        Maximum magnification to sample patches at. Must be greater than 1.
    overview_magnification : float, optional
        The level of magnification to produce a thumbnail overview of the
        slide / region at. Must be greater than 1.
    background_tolerance : float, optional
        The maximum fraction of a patch allowed to be background. We use Otsu's
        method to distinguish between foreground and background.
        Patches comprised of more than `background_tolerance` background are
        discarded. Must be on [0, 1].
    top_directory : PathLike, optional
        The top directory to save all output images in. If this argument is
        not supplied the output will appear in the current working directory.
        The patches will be placed in:
        "top_directory/patches_patchsize_stride/".
    cleanup_workers : int, optional
        The number of processes to use when cleaning up the patches that don't
        survive the Otsu thresholding. The processing time should scale
        near-linearly with `cleanup_workers`. If you pass an incompatible type
        or value, the number of workers available to the process will be used.
    qupath_binary : PathLike, optional
        The executable qupath binary file, by default "qupath".
        `qupath_binary` binary must be findable by
        `shutil.which(qupath_binary)`.
    zip_patches : bool, optional
        Determines whether the directories holding the patches are zipped or
        not.


    Examples
    --------
    #TODO: Add new examples.

    """

    def __init__(
        self,
        patch_size: int = 256,
        stride: int = 128,
        min_patch_mag=5.0,
        max_patch_mag=20.0,
        overview_mag: float = 2.5,
        background_tol: float = 0.75,
        top_dir: Optional[PathLike] = None,
        cleanup_workers: Optional[int] = None,
        qupath_binary: PathLike = "qupath",
        zip_patches: bool = True,
    ):
        """Construct PatchExtractor."""
        self._patch_size = ap.process_patch_size(patch_size)
        self._stride = ap.process_stride(stride)
        self._min_mag, self._max_mag = ap.process_min_max_mags(
            min_patch_mag,
            max_patch_mag,
        )
        self._overview_mag = ap.process_mag_var(overview_mag)
        self._background_tol = ap.process_background_tol(background_tol)
        self._top_directory = self._format_top_dir_path(top_dir)
        self._cleanup_workers = ap.process_cleanup_workers(cleanup_workers)
        self._qupath = ap.process_qupath_binary(qupath_binary)
        self._zip_patches = ap.process_zip_patches(zip_patches)

        self._output_format = ".png"

    _otsu_threshold = 0.0
    _magnification = 0.0
    _wsi = Path()

    def _format_top_dir_path(self, top_dir: Optional[PathLike] = None) -> Path:
        """Format the path to the top directory the patches are stored in.

        Paramerers
        ----------
        top_directory : PathLike
            Path to the folder you wish to save the patch directory in.

        Returns
        -------
        Path
            Path to the top directory the patch extraction output is stored in.

        """
        top_dir = top_dir if top_dir is not None else ""
        ap.check_path_like(top_dir)
        patch_dir = Path(f"patches_{self._patch_size}_{self._stride}")
        return (Path(top_dir or "") / patch_dir).resolve()

    def _save_wsi_overview(
        self,
        region: Dict[str, int],
        parent_dir: Optional[PathLike] = None,
    ) -> None:
        """Create a downscaled overview of the WSI or region.

        Parameters
        ----------
        region : dict[str, int]
            A dictonary of keys: `left`, `top`, `width`, `height`, which
            specify the region on the WSI we wish to focus on. If the entries
            are zero, the entire WSI is used.
        parent_dir : PathLike, optional
            Parent directory to append to the output path, or None.

        """
        target_path = self._build_output_dir(parent_dir or "")
        target_path = Path(target_path, "overview.png")

        command = qupath.create_qupath_command(
            self._wsi,
            target_path,
            self._calculate_overview_downscale(),
            self._patch_size,
            self._stride,
            self._qupath,
            self._output_format,
            "generate_overview",
            region,
        )
        qupath.run_command(command)

    def _calculate_overview_downscale(self) -> float:
        """Calculate the overview downscaling required.

        Returns
        -------
        float
            The downscaling factor reuired to achieve the requested overview
            magnification.

        """
        return self._magnification / self._overview_mag

    def _otsu_overview_image(self, parent_dir: PathLike) -> None:
        """Calculate Otsu threshold on the overview image.

        Parameters
        ----------
        parent_dir : PathLike
            Path appended to `self._top_directory` before the overview was
            saved.

        """
        overview_dir = self._build_output_dir(parent_dir)
        file_name = Path("overview.png")
        img = io.imread(overview_dir / file_name, as_gray=True)
        img[img == 0] = 1
        self._otsu_threshold = threshold_otsu(img)

    def _generate_patches(
        self,
        downscaling: float,
        region: Dict[str, int],
        parent_dir: Optional[PathLike] = None,
    ) -> None:
        """Generate patches from the WSI.

        Parameters
        ----------
        downscaling : int
            The level of downscaling to use when generating the patches.
        region : dict[str, int]
            Dictionary with the keys `left`, `top`, `width`, `height`
            specifying the region of interest on the WSI. If the entries are
            all zero, the entire WSI is used.
        parent_dir : PathLike
            Path to append to `self._top_directory` before saving the output.

        """
        parent_dir = Path(parent_dir or "")
        patch_dir = self._build_output_dir(parent_dir, downscaling)
        self._except_if_output_dir_exists(patch_dir)

        command = qupath.create_qupath_command(
            self._wsi,
            patch_dir,
            downscaling,
            self._patch_size,
            self._stride,
            self._qupath,
            self._output_format,
            "generate_patches",
            region,
        )

        qupath.run_command(command)
        self._apply_otsu_threshold_to_patches(patch_dir)
        self._except_if_empty_dir(patch_dir)
        self.zip_and_clean_patch_dir(patch_dir)

    def zip_and_clean_patch_dir(self, patch_dir: Path) -> None:
        """Zip the patches in `patch_dir`."""
        if self._zip_patches:
            make_archive(str(patch_dir), "zip", patch_dir)
            rmtree(patch_dir)

    def _build_output_dir(
        self,
        output_dir: PathLike,
        downscale: Optional[float] = None,
    ) -> Path:
        """Create the save directory.

        Parameters
        ----------
        output_dir : PathLike
            Output directory which will be appended to `self._top_directory`.
        downscale : int, optional
            Level of downscaling at which to extract the patches, or None to
            return the parent of the patch directory.

        Returns
        -------
        output_dir : Path
            The path to the parent directrory the patches will be saved in.

        """
        output_dir = Path(self._top_directory, output_dir)
        if downscale is not None:
            output_dir = output_dir / f"mag_{self._magnification / downscale}"
        return output_dir

    @staticmethod
    def _except_if_output_dir_exists(output_dir: PathLike) -> None:
        """Check if the output directory exists.

        Parameters
        ----------
        output_dir : PathLike
            Path we wish to check exists or not.

        Raises
        ------
        FileExistsError
            Ensure `output_dir` does not already exist.

        """
        dir_path = Path(output_dir).resolve()
        zip_path = dir_path.with_suffix(dir_path.suffix + ".zip")

        if dir_path.exists() or zip_path.exists():
            raise FileExistsError(f"Directory {output_dir} already exists.")

    @staticmethod
    def _except_if_empty_dir(directory: Union[str, Path]) -> None:
        """Raise an exception if `directory` is empty.

        Parameters
        ----------
        directory : str or Path
            The directory to be checked.

        Raises
        ------
        NotADirectoryError
            If `directory` is not a directory.
        RuntimeError
            If `directory` is empty.

        """
        test_dir = Path(directory)

        if not test_dir.is_dir():
            raise NotADirectoryError(f"'{test_dir}' is not a directory.")
        if len(list(test_dir.glob("*"))) == 0:
            raise RuntimeError(f"'{test_dir}' is empty.")

    def _apply_otsu_threshold_to_patches(self, patch_dir: PathLike) -> None:
        """Apply Otsu's threshold to the generated patches.

        Patches with too much background are deleted.

        Parameters
        ----------
        patch_dir : PathLike
            Directory containing the patches.

        """
        patch_paths = list(Path(patch_dir).glob("*.png"))
        with mp.Pool(processes=self._cleanup_workers) as pool:
            pool.map(
                self._threshold_single_patch,
                patch_paths,
                chunksize=len(patch_paths) // self._cleanup_workers,
            )
        pool.close()
        pool.join()

    def _threshold_single_patch(self, patch_path: PathLike) -> None:
        """Threshold a single patch.

        Parameters
        ----------
        patch_path : PathLike
            Path to the patch we wish to test for too much background.

        """
        patch = io.imread(patch_path, as_gray=True)
        patch[patch == 0] = 1.0
        background_pixels = (patch > self._otsu_threshold).sum()
        background_fraction = background_pixels / (self._patch_size ** 2)
        if background_fraction > self._background_tol:
            Path(patch_path).unlink()

    def _get_downscalings_from_magnification(self) -> List[float]:
        """Return iterator covering the requested patch configurations.

        Returns
        -------
        downscalings : List[float]
            List of downscalings to extract patches at.

        """
        mag = self._magnification
        counter = 0
        downscalings = []
        while mag >= self._min_mag:
            downscale = 2.0 ** counter
            mag = self._magnification / downscale
            counter += 1
            if self._min_mag <= mag <= self._max_mag:
                downscalings.append(downscale)
        return downscalings

    def _set_wsi_magnification(
        self,
        user_supplied_mag: Optional[float] = None,
    ):
        """Determine the magnification of the current wsi.

        Raises
        ------
        MagnificationException
            If qupath cannot determine the slide's magnification, and the user
            has not supplied an acceptable alternative value, raise.
        ValueError
            If the slide magnifcation is less than the minimum requested
            patch magnification, raise.

        """
        qupath_mag = qupath.get_wsi_mag(self._wsi, self._qupath)

        if not qupath_mag != qupath_mag:
            self._magnification = qupath_mag
        elif user_supplied_mag is not None:
            ap.process_mag_var(user_supplied_mag)
            self._magnification = user_supplied_mag
        else:
            msg = f"Unable to determine the magnification of wsi {self._wsi}"
            raise MagnificationException(msg)

        self._check_wsi_mag_is_compatible()

    def _check_wsi_mag_is_compatible(self) -> None:
        """Check WSI mag is compatible with requested mags.

        Raises
        ------
        ValueError
            If WSI mag is less than min requested user mag.
        ValueError
            If WSI mag is less than overview mag.

        """
        if self._magnification < self._min_mag:
            msg = f"WSI magnification is set to '{self._magnification}', "
            msg += "but the minimum requested magnification is "
            msg += f"'{self._min_mag}'. The WSI magnifcation cannot be less "
            msg += "than the minimum requested magnification."
            raise ValueError(msg)

        if self._magnification < self._overview_mag:
            msg = f"WSI magnification '{self._magnification}' is less than "
            msg += f"overview magnification '{self._overview_mag}'. The "
            msg += "overview magnification cannot be less than the WSI "
            msg += "magnification."
            raise ValueError(msg)

    def _qupath_calls(
        self,
        region: dict[str, int],
        parent_dir: PathLike,
        user_supplied_mag: Optional[float] = None,
        generate_patches: bool = True,
    ):
        """Call the qupath routines for region exporting and patch extraction.

        Parameters
        ----------
        region : dict[str, int]
            The region of the WSI to focus on. If each item is zero,
            use the entire WSI.
        parent_dir : path_like
            The folder to export image(s) to.
        user_supplied_mag : float
            Magnification supplied by user. Only used if qupath fails
            to read the magnification.
        generate_patches : bool
            Controls whether patches are generated or not.

        """
        self._set_wsi_magnification(user_supplied_mag=user_supplied_mag)
        self._save_wsi_overview(region=region, parent_dir=parent_dir)
        self._otsu_overview_image(parent_dir)


        if generate_patches is True:
            for downscaling in self._get_downscalings_from_magnification():
                self._generate_patches(
                    max(1.0, downscaling),
                    region=region,
                    parent_dir=parent_dir,
                )

    def __call__(
        self,
        wsi: PathLike,
        parent_dir: Optional[PathLike] = None,
        region: Optional[Dict[str, int]] = None,
        user_supplied_mag: Optional[float] = None,
        generate_patches: bool = True,
    ) -> None:
        """Generate patches from `wsi`.

        Parameters
        ----------
        wsi_path : PathLike
            Path to the WSI we want to extract patches from.
        parent_dir : PathLike, optional
            Parent directory to output the patches in.
        region : dict[str, int], optional
            A dictionary with the keys `left`, `top`, `width` and `height`
            which specifies the region of interest on the WSI. If None, the
            entire WSI is used. If all values are zero, the entire WSI is used.
        user_supplied_mag : float, optional
            For some images, Qupath may be unable to determine the
            magnification. If this happens, the user can supply the
            magnification directly here. If qupath can determine the slide's
            magnification, this value is ignored.
        generate_patches : bool, optional
            If True, an overview of the WSI / region of interest will be
            generated as well as patches. If False, only the overview image
            will be generated (and no patches).

        """
        self._wsi = ap.process_wsi_arg(wsi)
        parent_dir = ap.process_parent_dir_arg(parent_dir)
        region = ap.process_region_arg(region)
        user_supplied_mag = ap.process_user_supplied_mag(user_supplied_mag)
        generate_patches = ap.process_generate_patches_arg(generate_patches)

        start_time = perf_counter()

        # for _ in range(3):
        #     try:
        #         self._qupath_calls(
        #             region,
        #             parent_dir,
        #             user_supplied_mag=user_supplied_mag,
        #             generate_patches=generate_patches,
        #         )
        #         break
        #     except QupathException:
        #         if (folder := self._build_output_dir(parent_dir)).exists():
        #             rmtree(folder)
        self._qupath_calls(
                    region,
                    parent_dir,
                    user_supplied_mag=user_supplied_mag,
                    generate_patches=generate_patches,
                )

        stop_time = perf_counter()
        print(f"Scan processed in {stop_time - start_time:.6f} seconds")
