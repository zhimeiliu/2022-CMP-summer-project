#!/usr/bin/env python
"""Summarise the case counts we in `"../wsi_metadata/"`."""
from pathlib import Path

from pandas import DataFrame, read_csv, concat


def _load_scan_metadata() -> DataFrame:
    """Load the metadata for each scan source.

    Returns
    -------
    DataFrame
        The scan-level metadata for every source.

    """
    metadata_dir = Path(__file__).resolve().parent.parent / "wsi-metadata/"
    files = list(metadata_dir.glob("*.csv"))
    return concat(map(read_csv, files), axis=0, ignore_index=True)


df = _load_scan_metadata()

keys = ["source", "label", "test_or_train"]
source_label = df.groupby(keys).case_id.nunique()
print(source_label)
print(f"Total cases = {source_label.sum()}")


print("\n")

df["file_format"] = df.scan.apply(lambda x: x.split(".")[-1])

keys = ["source", "label", "file_format", "test_or_train"]
src_label_format = df.groupby(keys).scan.count()

print(src_label_format)
print(f"Total scans = {src_label_format.sum()}")
