"""Tool for heat-mapping patch-wise predictions."""
from typing import Union, Tuple
from pathlib import Path
import regex

import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
from pandas import DataFrame

from skimage.io import imread


from .base_vis import BaseVis


class PredictionViewer(BaseVis):
    """Tool for reconstructing scans and overlaying the predictions.

    Parameters
    ----------
    pred_df : DataFrame
        DataFrame holding the tile metadata for all of the patients we wish to
        produce plots for.
    out_dir : str or Path
        Directory the predictions visuals will be generated in.
    pred_key : str
        `pred_df` key giving the predictions.
    save_dpi : int, optional
        DPI to save the figure with.

    """

    def __init__(
        self,
        pred_key: str = "pred_coeliac",
        save_dpi: int = 1500,
    ):
        """Construct PredictionViewer."""
        super().__init__(dpi=save_dpi)

        self.pred_key = pred_key
        self.colour_map = "Blues"

    @staticmethod
    def _pixel_info_from_patch_path(patch_name: Union[str, Path]) -> Tuple[int, ...]:
        """Extract the pixel metadata from `patch_filename` with regex.

        Parameters
        ----------
        patch_name : str or Path
            The filename of the patch in question.

        Returns
        -------
        downscaling : int
            The level of downscaling the patch was taken at.
        x_pixel : int
            The column pixel index from the WSI with in the level zero
            reference frame.
        y_pixel : int
            Same as x but row coordinate.
        width : int
            The width, in pixels, of the patch in the level zero reference
            frame.
        height : int
            Same as width but height.

        """
        pattern = r"\[d=(\d+),x=(\d+),y=(\d+),w=(\d+),h=(\d+)\]"
        match = regex.search(pattern, str(patch_name))
        downscaling = int(match.group(1))
        x_pixel = int(match.group(2))
        y_pixel = int(match.group(3))
        width = int(match.group(4))
        height = int(match.group(5))
        return downscaling, x_pixel, y_pixel, width, height

    def _set_downscale(self, patch_name: Union[Path, str]) -> None:
        """Work out and set the downscaling based on patch filename.

        The patch filename contains the level of downscaling.

        """
        scale, _, _, _, _ = self._pixel_info_from_patch_path(patch_name)
        setattr(self, "downscale", scale)

    def _set_patch_size(self, patch_name: Union[str, Path]):
        """Set the patch size for this instance of plotting.

        Parameters
        ----------
        patch_name : str or Path
            Path to a patch.

        """
        patch = imread(patch_name)
        setattr(self, "patch_width", patch.shape[1])
        setattr(self, "patch_height", patch.shape[0])

    def _add_coords_to_patient_df(self, patient_df: DataFrame):
        """Extract the tile coordinates from their file names.

        Parameters
        ----------
        patient_df : DataFrame
            DataFrame holding the tile metadata for a single patient.

        """
        coords = list(
            zip(*patient_df.patch_path.apply(self._pixel_info_from_patch_path))
        )[1:]

        downscale = getattr(self, "downscale")

        for key, val in zip(["col", "row", "width", "height"], coords):
            patient_df[key] = np.array(val, dtype=int) // downscale
        patient_df = patient_df.convert_dtypes()

    def _calculate_image_size(
        self, row_coords: pd.Series, col_coords: pd.Series
    ) -> Tuple[int, int]:
        """Calculate the dimensions of the output image in pixels."""
        patch_height = getattr(self, "patch_height")
        patch_width = getattr(self, "patch_width")

        return (
            int(row_coords.max() - row_coords.min() + patch_height),
            int(col_coords.max() - col_coords.min() + patch_width),
        )

    def fill_image_and_preds_from_tiles(
        self,
        patient_df: DataFrame,
        image_array: np.ndarray,
        pred_array: np.ndarray,
        divisor: np.ndarray,
    ):
        """Fill the image and prediction arrays using the tiles.

        patient_df : DataFrame
            DataFrame holding the tile metadata for the patient in question.
        image_array : np.ndarray
            Array we are using to reconstruct the image in.
        pred_array : np.ndarray
            Array we are using to hold the predictions we want overlay on
            `image_array`.
        divisor : np.ndarray
            Array storing the count of the number of times each pixel in
            `image_array` was added to. We `image_array` by the to deal with
            the overlapping tiles.

        """
        patch_height = getattr(self, "patch_height")
        patch_width = getattr(self, "patch_width")

        for row in patient_df.itertuples():

            patch = imread(row.patch_path).astype(image_array.dtype)
            prediction = getattr(row, self.pred_key)

            top, bottom = row.row, row.row + patch_height
            left, right = row.col, row.col + patch_width

            img_slice = image_array[top:bottom, left:right, :]
            pred_slice = pred_array[top:bottom, left:right]

            new_img_slice = np.maximum(img_slice, patch)
            new_pred_slice = np.maximum(pred_slice, prediction)

            image_array[top:bottom, left:right, :] = new_img_slice
            pred_array[top:bottom, left:right] = new_pred_slice

            # divisor[top:bottom, left:right] += 1

    def _get_img_and_prediction_arrays(
        self, patient_df: DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the patient's image and prediction as an np.ndarray.

        Parameters
        ----------
        patient_df : DataFrame
            DataFrame holding the tile metadata for the patient in question.

        """
        self._add_coords_to_patient_df(patient_df)
        self._rescale_and_sort_tile_coords(patient_df)
        img_size = self._calculate_image_size(patient_df.row, patient_df.col)

        image_array = np.zeros(img_size + (3,), dtype=int)
        divisor = np.zeros(img_size, dtype=int)
        pred_array = -np.ones(img_size, dtype=float)

        self.fill_image_and_preds_from_tiles(
            patient_df,
            image_array,
            pred_array,
            divisor,
        )

        image_array = image_array.astype(float)
        self.average_pixels(image_array, divisor)
        image_array = image_array.astype(int)

        self.average_pixels(pred_array, divisor)
        pred_array = np.round(pred_array, 2)

        return image_array.astype(int), pred_array

    @staticmethod
    def _purge_background_rows_and_cols(img: np.ndarray, preds: np.ndarray):
        """Remove the background rows and columns from `img` and `preds`.

        WSIs contain lots of rows and columns which are entirely background.
        We may as well cut these from the image and afford the tissue regions
        more space.

        Parameters
        ----------
        img : np.ndarray
            RGB image.
        preds : np.ndarray
            The predictions to overlay on the image.

        Returns
        -------
        img : np.ndarray
            The image with rows and cols which are entire background removed.
        preds : np.ndarray
            Same as `img` but preds.

        """
        keep_rows = (preds > -1).any(axis=1)
        keep_cols = (preds > -1).any(axis=0)

        img = img[:, keep_cols, :]
        preds = preds[:, keep_cols]

        img = img[keep_rows]
        preds = preds[keep_rows]
        return img, preds

    @staticmethod
    def _make_ndarray_landscape(img_array: np.ndarray):
        """Make `img_like` landscape if it isn't.

        Parameters
        ----------
        img_array : np.ndarray
            An image-like array

        Returns
        -------
        img_array
            A landscape version of the input (rotated if it wasn't already
            landscape).

        """
        if img_array.shape[0] > img_array.shape[1]:
            img_array = np.rot90(img_array, k=-1)
        return img_array

    def produce_plot(self, scan_df: DataFrame, file_name: Union[str, Path]):
        """Imshow the image and the overlay the predictions.

        Parameters
        ----------
        patient_df : DataFrame
            DataFrame holding the predictions and tile paths for the patient
            image.

        """
        self._set_downscale(scan_df.patch_path.iloc[0])
        self._set_patch_size(scan_df.patch_path.iloc[0])

        # patient_df[self.pred_key] = patient_df[self.pred_key].apply(
        #     lambda x: 0 if x < 0.5 else 1
        # )

        img, preds = self._get_img_and_prediction_arrays(scan_df)
        img, preds = self._purge_background_rows_and_cols(img, preds)

        img = self._make_ndarray_landscape(img)
        preds = self._make_ndarray_landscape(preds)

        preds[preds <= 0] = None

        figsize = (3.0, 3.0 * img.shape[0] / img.shape[1])
        fig, axes = plt.subplots(1, 1, figsize=figsize)

        axes.imshow(img)
        pred_map = axes.imshow(
            preds,
            alpha=0.75,
            vmin=0,
            vmax=1,
            cmap=self.colour_map,
        )
        axes.set_xticks([])
        axes.set_yticks([])

        # divider = make_axes_locatable(axes)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = fig.colorbar(pred_map, cax)
        # cbar.set_ticks(np.arange(0, 1.2, 0.2))

        fig.tight_layout(pad=0.05)

        self._save_fig(fig, Path(file_name))
        self._figure_cleanup(axes)

    @staticmethod
    def average_pixels(img_arr: np.ndarray, divisor: np.ndarray):
        """Average pixels of RGB or Greyscale-like image.

        Parameters
        ----------
        img_arr : ndarray
            RGB or greyscale like image
        divisor : ndarray
            Array of counts we with to divide each pixel by. Must have same
            height and width as image.

        """
        divisor[divisor == 0] = 1

        if len(img_arr.shape) == 3:
            img_arr /= np.stack([divisor, divisor, divisor], axis=2)
        elif len(img_arr.shape) == 2:
            img_arr /= divisor
        else:
            raise RuntimeError("Unacceptable image dimensionality.")

    @staticmethod
    def _rescale_and_sort_tile_coords(patient_df: DataFrame):
        """Rescale and sort the tile coordinates.

        We shift the tile coordinates so the origin goes to zero and then sort
        the patient's data frame based on the coordinates.
        """
        patient_df.col -= patient_df.col.min()
        patient_df.row -= patient_df.row.min()
        patient_df.sort_values(by=["row", "col"], inplace=True)
        patient_df.reset_index(drop=True, inplace=True)
