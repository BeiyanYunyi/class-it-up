"""Shared data preparation for the CLI and training."""

from pathlib import Path
import sqlite3

import polars as pl


def export_dataset(database: Path, output: Path) -> pl.DataFrame:
    # Read-only mode also prevents a typo from creating an empty database.
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        frame = pl.read_database("SELECT comment, status FROM wl_Comment", connection)
    frame = frame.select(
        pl.col("comment"),
        pl.col("status").eq("waiting").cast(pl.UInt8).alias("label"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_ndjson(output)
    return frame


def load_dataframe(source: Path) -> pl.DataFrame:
    frame = pl.read_ndjson(source).select("comment", "label")
    if frame["comment"].null_count() or frame["label"].null_count():
        raise ValueError("comment and label must not contain null values")
    if frame["comment"].dtype != pl.String:
        raise ValueError("comment must contain strings")
    labels = frame["label"]
    if not (labels.dtype.is_numeric() or labels.dtype == pl.Boolean):
        raise ValueError("label must be 0 or 1")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("label must be 0 or 1")
    return frame.select(
        pl.col("comment").str.replace_all(r"<[^>]*>", ""),
        pl.col("label").cast(pl.UInt8),
    )
