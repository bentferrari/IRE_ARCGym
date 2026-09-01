#!/usr/bin/env python3
"""Extract uniformly spaced centerline keypoints from hollow colon OBJ meshes.

The colon assets are thin hollow-wall meshes. The wall is voxelized and closed
to reconstruct the solid outer volume, whose medial skeleton estimates the
lumen center. The retained route is smoothed and resampled at uniform arc
length.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


MESH_NAMES = (
    "outputconv_shell_hole_0000.obj",
    "conv_shell_hole_0000_IJK_1000_600_600_3mmthick_hole2.obj",
    "conv_shell_hole_0002_IJK_1000_600_600_3mmthick_hole2.obj",
    "conv_shell_hole_0003_IJK_1000_600_600_3mmthick_hole2.obj",
    "conv_shell_hole_0006_IJK_1000_600_600_3mmthick_hole2.obj",
)


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    triangles: list[tuple[int, int, int]] = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                face = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
                triangles.extend((face[0], face[i], face[i + 1]) for i in range(1, len(face) - 1))
    if not vertices or not triangles:
        raise ValueError(f"OBJ has no usable triangular surface: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int32)


def skeleton_graph(voxels: np.ndarray, pitch: float):
    labels, count = ndimage.label(voxels, structure=np.ones((3, 3, 3), dtype=bool))
    if count == 0:
        raise ValueError("Skeleton is empty")
    sizes = np.bincount(labels.ravel())
    largest_label = int(np.argmax(sizes[1:]) + 1)
    indices = np.argwhere(labels == largest_label)
    lookup = {tuple(index): number for number, index in enumerate(indices)}
    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    for number, index in enumerate(indices):
        for offset_tuple in itertools.product((-1, 0, 1), repeat=3):
            offset = np.asarray(offset_tuple)
            if not np.any(offset):
                continue
            neighbor = lookup.get(tuple(index + offset))
            if neighbor is not None and neighbor > number:
                weight = float(np.linalg.norm(offset) * pitch)
                rows.extend((number, neighbor))
                columns.extend((neighbor, number))
                weights.extend((weight, weight))
    graph = coo_matrix(
        (weights, (rows, columns)), shape=(len(indices), len(indices))
    ).tocsr()
    return indices, graph


def surface_longitudinal_coordinate(
    vertices: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Assign surface vertices an end-to-end coordinate along the hollow tube."""
    edges = np.vstack(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    columns = np.concatenate((edges[:, 1], edges[:, 0]))
    weights = np.concatenate((lengths, lengths))
    graph = coo_matrix(
        (weights, (rows, columns)), shape=(len(vertices), len(vertices))
    ).tocsr()

    _, labels = connected_components(graph, directed=False)
    component = np.flatnonzero(labels == np.argmax(np.bincount(labels)))
    component_graph = graph[component][:, component]
    first_distances = dijkstra(component_graph, directed=False, indices=0)
    first_end = int(np.argmax(first_distances))
    distances_a = dijkstra(component_graph, directed=False, indices=first_end)
    second_end = int(np.argmax(distances_a))
    distances_b = dijkstra(component_graph, directed=False, indices=second_end)
    diameter = float(distances_a[second_end])

    # The wall surface forms a longitudinal cycle: one half runs along one
    # side of the wall and the other half returns along the other side. Folding
    # the geodesic diameter aligns both sides into one opening-to-opening axis.
    unfolded = 0.5 * (distances_a - distances_b + diameter)
    longitudinal = np.minimum(unfolded, diameter - unfolded)
    return component, longitudinal


def predecessor_path(predecessors: np.ndarray, start: int, end: int) -> list[int]:
    result: list[int] = []
    current = end
    while current != -9999:
        result.append(int(current))
        if current == start:
            return result[::-1]
        current = int(predecessors[current])
    raise ValueError("No connected skeleton path between extrema")


def volumetric_centerline(
    vertices: np.ndarray,
    triangles: np.ndarray,
    pitch: float,
    closing_radius: float,
    scalar_penalty: float,
) -> np.ndarray:
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    voxel_grid = mesh.voxelized(pitch=pitch, method="subdivide")
    radius_voxels = max(1, int(round(closing_radius / pitch)))
    padding = radius_voxels + 2
    wall = np.pad(voxel_grid.matrix, padding)

    # Euclidean-distance closing fills the lumen enclosed by the thin wall and
    # yields the colon's solid outer volume without a large structuring element.
    dilated = ndimage.distance_transform_edt(~wall) <= radius_voxels
    closed = ndimage.distance_transform_edt(dilated) > radius_voxels
    volume = ndimage.binary_fill_holes(closed)
    skeleton = skeletonize(volume)
    voxel_indices, graph = skeleton_graph(skeleton, pitch)
    skeleton_points = voxel_grid.indices_to_points(voxel_indices - padding)

    surface_component, surface_scalar = surface_longitudinal_coordinate(vertices, triangles)
    neighbor_count = min(20, len(surface_component))
    nearest = cKDTree(vertices[surface_component]).query(
        skeleton_points, k=neighbor_count
    )[1]
    if neighbor_count == 1:
        skeleton_scalar = surface_scalar[nearest]
    else:
        skeleton_scalar = np.median(surface_scalar[nearest], axis=1)

    # Morphological closing can connect spatially adjacent colon folds. Such a
    # bridge is short in XYZ but represents a large jump along the tube. Penalize
    # that longitudinal jump while retaining normal local skeleton edges.
    weighted = graph.tocoo(copy=True)
    scalar_delta = skeleton_scalar[weighted.row] - skeleton_scalar[weighted.col]
    weighted.data = weighted.data + scalar_penalty * scalar_delta**2 / weighted.data
    weighted = weighted.tocsr()
    start = int(np.argmin(skeleton_scalar))
    end = int(np.argmax(skeleton_scalar))
    _, predecessors = dijkstra(
        weighted, directed=False, indices=start, return_predecessors=True
    )
    route = predecessor_path(predecessors, start, end)
    return skeleton_points[route]


def smooth_and_resample(points: np.ndarray, spacing: float) -> np.ndarray:
    # Remove high-frequency band jitter before fitting a gently smoothing spline.
    window = min(11, len(points) if len(points) % 2 else len(points) - 1)
    if window >= 5:
        points = savgol_filter(points, window_length=window, polyorder=2, axis=0, mode="interp")
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], segment > 1e-6))
    points = points[keep]
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    u = cumulative / cumulative[-1]
    spline, _ = splprep(points.T, u=u, s=len(points) * 0.25, k=min(3, len(points) - 1))
    dense_u = np.linspace(0.0, 1.0, max(2000, len(points) * 10))
    dense = np.column_stack(splev(dense_u, spline))
    dense_arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))))
    targets = np.arange(0.0, dense_arc[-1], spacing)
    if dense_arc[-1] - targets[-1] > spacing * 0.25:
        targets = np.append(targets, dense_arc[-1])
    return np.column_stack([np.interp(targets, dense_arc, dense[:, dim]) for dim in range(3)])


def write_csv(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("x", "y", "z"))
        writer.writerows((f"{x:.6f}", f"{y:.6f}", f"{z:.6f}") for x, y, z in points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", type=Path, default=Path("src/arcgym/assets/colons"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spacing-mm", type=float, default=5.0)
    parser.add_argument("--voxel-mm", type=float, default=2.5)
    parser.add_argument("--closing-radius-mm", type=float, default=15.0)
    parser.add_argument("--scalar-penalty", type=float, default=0.01)
    args = parser.parse_args()
    if (
        args.spacing_mm <= 0
        or args.voxel_mm <= 0
        or args.closing_radius_mm <= 0
        or args.scalar_penalty < 0
    ):
        parser.error("spacing, voxel size, and closing radius must be positive; penalty cannot be negative")

    for colon_number, mesh_name in enumerate(MESH_NAMES, start=1):
        mesh_path = args.mesh_dir / mesh_name
        vertices, triangles = read_obj(mesh_path)
        raw_centers = volumetric_centerline(
            vertices,
            triangles,
            args.voxel_mm,
            args.closing_radius_mm,
            args.scalar_penalty,
        )
        centerline = smooth_and_resample(raw_centers, args.spacing_mm)
        output_path = args.output_root / f"sampling_colon{colon_number}" / "centerline.csv"
        write_csv(output_path, centerline)
        length = np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum()
        print(f"colon{colon_number}: {len(centerline)} keypoints, {length:.1f} mm -> {output_path}")


if __name__ == "__main__":
    main()
