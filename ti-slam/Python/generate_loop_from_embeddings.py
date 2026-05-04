"""
Python reimplementation of the MATLAB script `generate_loop_from_embeddings.m`.
It loads per-frame embedding vectors, computes pairwise cosine distances,
thresholds probable loop closures, and saves candidate index pairs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
from scipy.spatial.distance import pdist, squareform

HANDHELD_FILES: List[str] = [
    "2020-01-28-11-39-07",
]


def compute_loop_pairs(
    embedding_matrix: np.ndarray,
    threshold: float,
    tril_val: int,
) -> np.ndarray:
    """Return candidate loop pairs as (row, col) indices."""
    if embedding_matrix.ndim != 2:
        raise ValueError("Embedding matrix must be 2-D.")

    distances = pdist(embedding_matrix, metric="cosine")
    square = squareform(distances)
    tril_mask = np.tril(square, k=tril_val)
    mask = (tril_mask < threshold) & (tril_mask > 0)
    rows, cols = np.where(mask)
    return np.stack([rows, cols], axis=1) if rows.size else np.empty((0, 2), dtype=int)


def process_sequence(
    sequence: str,
    base_filename: str,
    data_folder: Path,
    output_folder: Path,
    threshold: float,
    tril_val: int,
) -> Path:
    embedding_path = data_folder / f"{base_filename}{sequence}.csv"
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    embedding_matrix = np.loadtxt(embedding_path, delimiter=",")
    loop_pairs = compute_loop_pairs(embedding_matrix, threshold, tril_val)

    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{base_filename}{sequence}_{threshold}.csv"
    np.savetxt(output_path, loop_pairs, fmt="%d", delimiter=",")
    print(f"Generated {loop_pairs.shape[0]} loop pairs -> {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate loop pairs from embeddings.")
    parser.add_argument(
        "--model-name",
        default="neural_embedding",
        help="Embedding model name (matches filename prefix).",
    )
    parser.add_argument(
        "--embed-threshold",
        type=float,
        default=0.045,
        help="Cosine distance threshold for candidate loops.",
    )
    parser.add_argument(
        "--tril-val",
        type=int,
        default=-18,
        help="Diagonal offset when masking lower-triangular pairs.",
    )
    parser.add_argument(
        "--data-folder",
        default="Python/loop/results/",
        help="Directory containing embedding csv files.",
    )
    parser.add_argument(
        "--output-folder",
        default="Python/odometry/results/",
        help="Directory to write loop pair csv files.",
    )
    parser.add_argument(
        "--list-type",
        choices=["inhouse", "handheld"],
        default="inhouse",
        help="Preset list of sequences to process.",
    )
    parser.add_argument(
        "--list-files",
        nargs="+",
        help="Override list of sequence names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    list_files: Iterable[str]
    if args.list_files:
        list_files = args.list_files
    else:
        list_files = HANDHELD_FILES

    base_filename = f"{args.model_name}_"
    data_folder = Path(args.data_folder)
    output_folder = Path(args.output_folder)

    for sequence in list_files:
        try:
            process_sequence(
                sequence=sequence,
                base_filename=base_filename,
                data_folder=data_folder,
                output_folder=output_folder,
                threshold=args.embed_threshold,
                tril_val=args.tril_val,
            )
        except FileNotFoundError as exc:
            print(exc)


if __name__ == "__main__":
    main()

