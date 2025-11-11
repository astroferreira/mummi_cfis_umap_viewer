#!/usr/bin/env python3
"""
Compute MiniBatchKMeans clusters in the 192-D CFIS feature space and add the
resulting cluster id to the DR7 dataframe.

Usage:
    python scripts/compute_clusters.py \
        --features data/UMAPS/feature_explorer/cfis_reduce.npy \
        --dr7 data/dataframes/DR7_with_UMAP.pk \
        --output data/dataframes/DR7_with_UMAP.pk \
        --n-clusters 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

DEFAULT_FEATURES = Path("data/UMAPS/feature_explorer/cfis_reduce.npy")
DEFAULT_DR7 = Path("data/dataframes/DR7_with_UMAP.pk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster CFIS features and persist cluster IDs into the DR7 dataframe."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help="Path to the CFIS feature numpy array (.npy).",
    )
    parser.add_argument(
        "--dr7",
        type=Path,
        default=DEFAULT_DR7,
        help="Path to the DR7 dataframe pickle to read.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the updated dataframe. Defaults to --dr7.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Number of clusters to compute in the 192-D space.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="MiniBatchKMeans batch size.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for MiniBatchKMeans.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = np.load(args.features)
    dr7 = pd.read_pickle(args.dr7).reset_index(drop=True)

    if len(dr7) != len(features):
        raise ValueError(
            f"Feature matrix length {len(features)} does not match DR7 length {len(dr7)}"
        )

    model = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        batch_size=args.batch_size,
        random_state=args.random_state,
        n_init="auto",
    )
    model.fit(features)

    updated = dr7.copy()
    updated["cluster_id"] = model.labels_.astype(int)

    output_path = args.output or args.dr7
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_pickle(output_path)

    uniq = updated["cluster_id"].nunique()
    print(
        f"Wrote {output_path} with cluster_id column "
        f"(n_clusters requested={args.n_clusters}, unique={uniq})."
    )


if __name__ == "__main__":
    main()
