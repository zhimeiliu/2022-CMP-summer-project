"""Miscellaneous functions."""
from typing import Union, List
from pathlib import Path
from zipfile import ZipFile


def list_zipfile_contents(zip_path: Union[Path, str]) -> List[str]:
    """List the files in the zip archive `zip_path`.

    Parameters
    ----------
    zip_path : Path
        Path to the zip file containing the patches.

    Returns
    -------
    List[str]
        List of paths to the contents of `zip_path`.

    """
    zip_path = Path(zip_path)
    with ZipFile(zip_path) as zip_archive:
        file_names = zip_archive.namelist()
    return list(map(lambda x: str(zip_path / Path(x)), file_names))
