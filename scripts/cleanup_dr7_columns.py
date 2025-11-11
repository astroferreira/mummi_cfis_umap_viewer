#!/usr/bin/env python3
"""
Normalize DR7 column names by removing *_x duplicates and stripping the *_y suffix.

This keeps one copy of each column (the *_y version) and, to preserve the
UMAP coordinates used in the web UI, writes explicit `UMAP_dim1` / `UMAP_dim2`
columns before the suffixes are removed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("data/UMAPS/DR7_with_UMAP.pk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the DR7 dataframe pickle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the updated dataframe (defaults to --input).",
    )
    parser.add_argument(
        "--keep-umap",
        action="store_true",
        help="Preserve UMAP_x/UMAP_y values by copying them to UMAP_dim1/UMAP_dim2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_pickle(args.input).reset_index(drop=True)

    if args.keep_umap:
        if "UMAP_x" in df.columns:
            df["UMAP_dim1"] = df["UMAP_x"].astype(float)
        if "UMAP_y" in df.columns:
            df["UMAP_dim2"] = df["UMAP_y"].astype(float)

    drop_cols = [col for col in df.columns if col.endswith("_x")]
    df = df.drop(columns=drop_cols, errors="ignore")

    rename_map = {col: col[:-2] for col in df.columns if col.endswith("_y")}
    df = df.rename(columns=rename_map)

    output_path = args.output or args.input
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output_path)

    print(
        f"Wrote {output_path} (dropped {len(drop_cols)} *_x columns, "
        f"renamed {len(rename_map)} *_y columns)."
    )


if __name__ == "__main__":
    main()
