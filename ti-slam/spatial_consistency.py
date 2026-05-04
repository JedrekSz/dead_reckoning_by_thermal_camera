"""
Python equivalent of `spatial_consistency.m`.
Filters loop-closure measurements based on spatial (cycle) consistency.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


@dataclass
class LoopMeasurement:
    src_idx: int
    dst_idx: int
    translation: np.ndarray  # shape (3,)
    euler_deg: np.ndarray  # XYZ order, degrees
    covariance: np.ndarray  # shape (6,)

    def to_array(self) -> np.ndarray:
        return np.concatenate(
            [
                [self.src_idx + 1, self.dst_idx + 1],
                self.translation,
                self.euler_deg,
                self.covariance,
            ]
        )


def row_to_tform(row: np.ndarray) -> np.ndarray:
    """Convert a flattened 3x4 matrix row into homogeneous transform."""
    mat = row.reshape(3, 4)
    last_row = np.array([[0.0, 0.0, 0.0, 1.0]])
    return np.vstack([mat, last_row])


def euler_deg_to_tform(translation: np.ndarray, euler_deg: np.ndarray) -> np.ndarray:
    rot = R.from_euler("XYZ", np.deg2rad(euler_deg))
    tform = np.eye(4)
    tform[:3, :3] = rot.as_matrix()
    tform[:3, 3] = translation
    return tform


def tform_to_pose_vectors(tform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    translation = tform[:3, 3]
    rot = R.from_matrix(tform[:3, :3]).as_euler("XYZ")
    return translation, rot


def load_loop_measurements(path: Path) -> List[LoopMeasurement]:
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 16:
        raise ValueError(f"Loop file {path} must have at least 16 columns.")

    measurements: List[LoopMeasurement] = []
    for _, row in df.iterrows():
        measurements.append(
            LoopMeasurement(
                src_idx=int(row[0]) - 1,
                dst_idx=int(row[2]) - 1,
                translation=row[[4, 5, 6]].to_numpy(dtype=float),
                euler_deg=row[[7, 8, 9]].to_numpy(dtype=float),
                covariance=row[[10, 11, 12, 13, 14, 15]].to_numpy(dtype=float),
            )
        )
    return measurements


def compute_consistent_loops(
    loop_measurements: List[LoopMeasurement],
    odom_array: np.ndarray,
    threshold: float,
    search_window: int = 50,
) -> List[LoopMeasurement]:
    consistent: List[LoopMeasurement] = []
    odom_len = odom_array.shape[0]

    for i, meas_i in enumerate(loop_measurements):
        max_j = min(len(loop_measurements), i + search_window + 1)
        abs_p1 = row_to_tform(odom_array[meas_i.dst_idx])
        abs_p1_trans, abs_p1_rot = tform_to_pose_vectors(abs_p1)

        for j in range(i + 1, max_j):
            meas_j = loop_measurements[j]
            if (
                meas_j.src_idx >= odom_len
                or meas_j.dst_idx >= odom_len
                or meas_i.src_idx >= odom_len
                or meas_i.dst_idx >= odom_len
            ):
                continue

            abs_p2 = row_to_tform(odom_array[meas_j.dst_idx])
            rel_pose_1 = np.linalg.inv(euler_deg_to_tform(meas_j.translation, meas_j.euler_deg))

            abs_p3 = row_to_tform(odom_array[meas_j.src_idx])
            abs_p4 = row_to_tform(odom_array[meas_i.src_idx])
            rel_pose_2 = np.linalg.inv(abs_p3) @ abs_p4

            rel_pose_3 = euler_deg_to_tform(meas_i.translation, meas_i.euler_deg)

            final_pose = abs_p2 @ rel_pose_1 @ rel_pose_2 @ rel_pose_3
            final_trans, final_rot = tform_to_pose_vectors(final_pose)

            diff = np.sqrt(
                (
                    (abs_p1_trans - final_trans) ** 2
                    + (abs_p1_rot - final_rot) ** 2
                ).sum()
                / 6.0
            )

            if diff < threshold:
                consistent.append(meas_i)
                consistent.append(
                    replace(
                        meas_j,
                        covariance=meas_i.covariance.copy(),
                    )
                )
                break
    return consistent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatial consistency filtering.")
    parser.add_argument("--loop-file", required=True, help="CSV of loop measurements.")
    parser.add_argument("--odom-file", required=True, help="CSV of odometry poses.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Consistency threshold.")
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional CSV to write consistent loops.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    odom_array = np.loadtxt(args.odom_file, delimiter=",")
    loops = load_loop_measurements(Path(args.loop_file))
    consistent = compute_consistent_loops(loops, odom_array, args.threshold)

    if args.output_file:
        matrix = np.stack([loop.to_array() for loop in consistent]) if consistent else np.empty((0, 14))
        np.savetxt(args.output_file, matrix, delimiter=",")
        print(f"Wrote {len(consistent)} consistent loops -> {args.output_file}")
    else:
        print(f"Found {len(consistent)} consistent loop entries.")


if __name__ == "__main__":
    main()

