"""
Python equivalent of `robust_pose_graph_optimization.m`.
Builds and optimizes a 3D pose graph using odometry and loop-closure constraints.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from spatial_consistency import (
    LoopMeasurement,
    compute_consistent_loops,
    load_loop_measurements,
    row_to_tform,
    euler_deg_to_tform,
)



HANDHELD_ODOM = ["37"]
HANDHELD_LOOP = [
    "2020-01-28-11-39-07",
]


def build_information_matrix(sigmas: np.ndarray, weight: float) -> np.ndarray:
    sigmas = np.asarray(sigmas).flatten()
    if sigmas.size < 6:
        raise ValueError("Sigma array must contain at least 6 values.")
    cov = np.diag(sigmas[:6])
    scaled = weight * cov
    return np.linalg.inv(scaled + 1e-12 * np.eye(6))


def row_to_position(row: np.ndarray) -> np.ndarray:
    return np.array([row[3], row[7], row[11]])


def load_absolute_poses(odom_array: np.ndarray) -> List[np.ndarray]:
    return [row_to_tform(row) for row in odom_array]


def add_odometry_edges(
    pose_graph: o3d.pipelines.registration.PoseGraph,
    poses: List[np.ndarray],
    sigma_array: np.ndarray,
    weight_odom: float,
) -> None:
    for idx in range(1, len(poses)):
        relative = np.linalg.inv(poses[idx - 1]) @ poses[idx]
        sigma_row = sigma_array[idx] if sigma_array.ndim > 1 else sigma_array
        info = build_information_matrix(sigma_row, weight_odom)
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                idx - 1,
                idx,
                relative,
                info,
                uncertain=False,
            )
        )


def add_loop_edges(
    pose_graph: o3d.pipelines.registration.PoseGraph,
    loops: Iterable[LoopMeasurement],
    weight_loop: float,
) -> None:
    for meas in loops:
        tform = euler_deg_to_tform(meas.translation, meas.euler_deg)
        info = build_information_matrix(meas.covariance, weight_loop)
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                meas.src_idx,
                meas.dst_idx,
                tform,
                info,
                uncertain=True,
            )
        )


def optimize_pose_graph(pose_graph: o3d.pipelines.registration.PoseGraph) -> None:
    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=1.0,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(pose_graph, method, criteria, option)


def plot_trajectories(
    gt: np.ndarray,
    odom_positions: np.ndarray,
    optimized_positions: np.ndarray,
    output_path: Path,
) -> Tuple[float, float, float, np.ndarray]:
    gt_origin = gt.copy()
    gt_origin[:, [3, 7, 11]] -= gt_origin[0, [3, 7, 11]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_origin[:, 3], gt_origin[:, 7], "--", color=(0.25, 0.25, 0.25), linewidth=3, label="Ground truth")
    ax.plot(odom_positions[:, 0], odom_positions[:, 1], color=(0.301, 0.745, 0.933), linewidth=3, label="Odometry")
    ax.plot(optimized_positions[:, 0], optimized_positions[:, 1], color=(0.635, 0.078, 0.184), linewidth=3, label="Optimized")
    ax.axis("equal")
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    min_len = min(len(gt_origin), len(optimized_positions))
    gt_xyz = gt_origin[:min_len, [3, 7, 11]]
    optimized_xyz = optimized_positions[:min_len]
    odom_xyz = odom_positions[:min_len]

    ate_slam = np.sqrt(((optimized_xyz - gt_xyz) ** 2).sum(axis=1) / 3.0)
    rmse_odom = np.mean(np.sqrt(((odom_xyz - gt_xyz) ** 2).sum(axis=1) / 3.0))
    improvement = (rmse_odom - np.mean(ate_slam)) / rmse_odom * 100.0
    return np.mean(ate_slam), np.var(ate_slam), rmse_odom, improvement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust pose graph optimization.")
    parser.add_argument("--list-type", choices=["inhouse", "handheld"], default="inhouse")
    parser.add_argument("--list-odom", nargs="+", help="Override odometry sequence list.")
    parser.add_argument("--list-loop", nargs="+", help="Override loop file list.")
    parser.add_argument("--odom-name", default="neural_odometry")
    parser.add_argument("--embedding-name", default="neural_embedding")
    parser.add_argument("--loop-pose-name", default="neural_loop_closure")
    parser.add_argument("--loop-threshold", default="0.045")
    parser.add_argument("--spatial-threshold", type=float, default=0.7)
    parser.add_argument("--weight-odom", type=float, default=0.01)
    parser.add_argument("--weight-loop", type=float, default=300.0)
    parser.add_argument("--main-dir", default="Python/")
    parser.add_argument("--results-dir", default="Python/odometry/results/")
    parser.add_argument("--gt-prefix", default="gt_seq")
    parser.add_argument("--output-dir", default="figures/optimized_odometry/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_odom and args.list_loop:
        list_odom = args.list_odom
        list_loop = args.list_loop
    elif args.list_type == "handheld":
        list_odom = HANDHELD_ODOM
        list_loop = HANDHELD_LOOP
    else:
        list_odom = IN_HOUSE_ODOM
        list_loop = IN_HOUSE_LOOP

    base_odom_filename = f"{args.odom_name}_epbest_seq"
    base_sigma_filename = f"sigmapose_{args.odom_name}_epbest_seq"
    base_loop_filename = f"pose_{args.loop_pose_name}_{args.embedding_name}_epbest_"

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) / f"{args.odom_name}_{args.loop_pose_name}"

    for odom_seq, loop_seq in zip(list_odom, list_loop):
        print(f"Processing sequence {odom_seq} / {loop_seq}")

        odom_path = results_dir / f"{base_odom_filename}{odom_seq}.txt"
        sigma_path = results_dir / f"{base_sigma_filename}{odom_seq}.txt"
        loop_path = results_dir / f"{base_loop_filename}{loop_seq}_{args.loop_threshold}.csv"
        gt_path = results_dir / f"{args.gt_prefix}{odom_seq}.txt"

        odom_array = np.loadtxt(odom_path, delimiter=",")
        sigma_array = np.loadtxt(sigma_path, delimiter=",")
        loop_measurements = load_loop_measurements(loop_path)
        consistent_loops = compute_consistent_loops(loop_measurements, odom_array, args.spatial_threshold)

        pose_graph = o3d.pipelines.registration.PoseGraph()
        absolute_poses = load_absolute_poses(odom_array)
        for pose in absolute_poses:
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose))

        add_odometry_edges(pose_graph, absolute_poses, sigma_array, args.weight_odom)
        add_loop_edges(pose_graph, consistent_loops, args.weight_loop)
        optimize_pose_graph(pose_graph)

        optimized_positions = np.array([node.pose[:3, 3] for node in pose_graph.nodes])
        odom_positions = np.array([row_to_position(row) for row in odom_array])
        gt_array = np.loadtxt(gt_path, delimiter=",")

        output_path = output_dir / f"optimized_traj_seq{odom_seq}_{loop_seq}.pdf"
        ate_mean, ate_var, rmse_odom, improvement = plot_trajectories(
            gt=gt_array,
            odom_positions=odom_positions,
            optimized_positions=optimized_positions,
            output_path=output_path,
        )

        print("RMSE ATE TI-SLAM (m):", ate_mean)
        print("Variance ATE TI-SLAM (m):", ate_var)
        print("RMSE odometry (m):", rmse_odom)
        print("Improvement (%):", improvement)


if __name__ == "__main__":
    main()

