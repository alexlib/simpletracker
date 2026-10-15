"""A small particle tracker that can link detections across missing frames.

Points are supplied as one NumPy array per frame. Point indices in the public
return values are zero-based, following normal Python conventions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

LinkingMethod = Literal["hungarian", "nearestneighbor"]
Points = Sequence[np.ndarray]


def _distance_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return Euclidean distances between every source and target point."""
    differences = source[:, None, :] - target[None, :, :]
    return np.linalg.norm(differences, axis=2)


def _validate_points(points: Points) -> list[np.ndarray]:
    arrays = [np.asarray(frame, dtype=float) for frame in points]
    if not arrays:
        raise ValueError("points must contain at least one frame")
    if any(frame.ndim != 2 for frame in arrays):
        raise ValueError("each frame must be a two-dimensional array")
    dimensions = {frame.shape[1] for frame in arrays}
    if len(dimensions) != 1:
        raise ValueError("all frames must have the same point dimension")
    return arrays


def hungarian_linker(
    source: np.ndarray,
    target: np.ndarray,
    max_distance: float = np.inf,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match points globally while using each source and target at most once.

    Returns ``(target_indices, distances, unassigned_targets)``. Unmatched
    source points have index ``-1`` and distance ``nan``.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    distances = _distance_matrix(source, target)
    if not len(source):
        return (
            np.empty(0, dtype=int),
            np.empty(0, dtype=float),
            np.arange(len(target), dtype=int),
        )
    if not len(target):
        return (
            np.full(len(source), -1, dtype=int),
            np.full(len(source), np.nan, dtype=float),
            np.empty(0, dtype=int),
        )
    if np.isfinite(max_distance):
        penalty = max(
            float(max_distance) + 1,
            float(np.max(distances, initial=0)) + 1,
        ) * (len(source) + len(target) + 1)
        costs = np.full((len(source), len(target) + len(source)), penalty)
        costs[:, : len(target)] = np.where(
            distances <= max_distance, distances, penalty
        )
    else:
        costs = distances
    rows, columns = linear_sum_assignment(costs)
    target_indices = np.full(len(source), -1, dtype=int)
    target_distances = np.full(len(source), np.nan, dtype=float)
    for row, column in zip(rows, columns):
        if column < len(target) and distances[row, column] <= max_distance:
            target_indices[row] = column
            target_distances[row] = distances[row, column]
    assigned = target_indices[target_indices >= 0]
    unassigned = np.setdiff1d(np.arange(len(target)), assigned, assume_unique=True)
    return target_indices, target_distances, unassigned


def nearest_neighbor_linker(
    source: np.ndarray,
    target: np.ndarray,
    max_distance: float = np.inf,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedily match the globally closest remaining pair at each step."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    distances = _distance_matrix(source, target)
    target_indices = np.full(len(source), -1, dtype=int)
    target_distances = np.full(len(source), np.nan, dtype=float)
    available_sources = set(range(len(source)))
    available_targets = set(range(len(target)))

    while available_sources and available_targets:
        pairs = (
            (distances[row, column], row, column)
            for row in available_sources
            for column in available_targets
        )
        distance, row, column = min(pairs)
        if distance > max_distance:
            break
        target_indices[row] = column
        target_distances[row] = distance
        available_sources.remove(row)
        available_targets.remove(column)

    return (
        target_indices,
        target_distances,
        np.array(sorted(available_targets), dtype=int),
    )


def simpletracker(
    points: Points,
    *,
    max_linking_distance: float = np.inf,
    max_gap_closing: float = 3,
    method: LinkingMethod = "hungarian",
    debug: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Track detections through frames and optionally close short gaps.

    Returns ``(tracks, adjacency_tracks, adjacency)``:

    * ``tracks`` contains one array per track, with a point index per frame
      and ``-1`` where the track has a gap.
    * ``adjacency_tracks`` contains global indices into ``np.concatenate(points)``.
    * ``adjacency`` is a boolean forward-link matrix over those global points.
    """
    frames = _validate_points(points)
    n_frames = len(frames)
    counts = np.array([len(frame) for frame in frames], dtype=int)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    total_points = int(offsets[-1])
    links: dict[int, int] = {}
    unmatched_sources: list[np.ndarray] = [np.arange(count) for count in counts]
    unmatched_targets: list[np.ndarray] = [np.arange(count) for count in counts]

    for frame_index in range(n_frames - 1):
        if method == "hungarian":
            linker = hungarian_linker
        elif method == "nearestneighbor":
            linker = nearest_neighbor_linker
        else:
            raise ValueError("method must be 'hungarian' or 'nearestneighbor'")
        target_indices, _, unassigned = linker(
            frames[frame_index],
            frames[frame_index + 1],
            max_linking_distance,
        )
        unmatched_sources[frame_index] = np.flatnonzero(target_indices < 0)
        unmatched_targets[frame_index + 1] = unassigned
        for source_index, target_index in enumerate(target_indices):
            if target_index >= 0:
                links[offsets[frame_index] + source_index] = (
                    offsets[frame_index + 1] + target_index
                )

    gap_limit = n_frames - 1 if np.isinf(max_gap_closing) else int(max_gap_closing)
    for source_frame in range(n_frames - 2):
        source_indices = unmatched_sources[source_frame].copy()
        for target_frame in range(
            source_frame + 2, min(n_frames, source_frame + gap_limit + 1)
        ):
            if not len(source_indices) or not len(unmatched_targets[target_frame]):
                continue
            target_indices, _, _ = nearest_neighbor_linker(
                frames[source_frame][source_indices],
                frames[target_frame][unmatched_targets[target_frame]],
                max_linking_distance,
            )
            matched = target_indices >= 0
            for source_local, target_local in enumerate(target_indices):
                if not matched[source_local]:
                    continue
                source_index = source_indices[source_local]
                target_index = unmatched_targets[target_frame][target_local]
                links[offsets[source_frame] + source_index] = (
                    offsets[target_frame] + target_index
                )
            source_indices = source_indices[~matched]
            unmatched_targets[target_frame] = unmatched_targets[target_frame][
                ~np.isin(np.arange(len(unmatched_targets[target_frame])), target_indices[matched])
            ]

    adjacency = np.zeros((total_points, total_points), dtype=bool)
    for source, target in links.items():
        adjacency[source, target] = True

    incoming = set(links.values())
    starts = [point for point in range(total_points) if point not in incoming]
    adjacency_tracks: list[np.ndarray] = []
    tracks: list[np.ndarray] = []
    for start in starts:
        chain = []
        current = start
        while True:
            chain.append(current)
            if current not in links:
                break
            current = links[current]
        adjacency_tracks.append(np.array(chain, dtype=int))
        track = np.full(n_frames, -1, dtype=int)
        for global_index in chain:
            frame_index = int(np.searchsorted(offsets[1:], global_index, side="right"))
            track[frame_index] = global_index - offsets[frame_index]
        tracks.append(track)

    if debug:
        print(f"Created {len(links)} links over {total_points} points.")
        print(f"Built {len(tracks)} tracks.")
    return tracks, adjacency_tracks, adjacency


__all__ = [
    "hungarian_linker",
    "nearest_neighbor_linker",
    "simpletracker",
]
