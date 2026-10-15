"""Resolve ambiguous crossings by judging whole trajectories.

A frame-by-frame tracker commits at every crossing using only the past. When
two particles pass closer than the detector noise, that choice can be wrong,
and the evidence that would settle it only shows up *afterwards*: a wrong
pairing makes both paths turn sharply at the crossing.

This step revisits the finished tracks:

1. Find pairs of tracks that come close to each other.
2. For each such moment, imagine the two tracks **swap their later parts**.
3. Score a track by how well a constant-velocity motion model explains it from
   start to end (the Kalman filter's *negative log-likelihood*: smooth paths
   score low, kinked paths score high).
4. Keep the swap if the two new tracks explain the data better than the two
   old ones. Repeat until no swap helps.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from kalman_tracker import MotionModel, make_motion_model, predict, update


def track_cost(
    track: np.ndarray,
    frames: Sequence[np.ndarray],
    model: MotionModel,
    fresh_covariance: np.ndarray,
) -> float:
    """Negative log-likelihood (up to a constant) of a track under ``model``.

    Small means "smooth and well explained". Frames where the track has no
    detection (``-1``) are bridged by prediction.
    """
    seen = [(frame, int(index)) for frame, index in enumerate(track) if index >= 0]
    if len(seen) < 2:
        return 0.0
    dim = frames[0].shape[1]
    previous_frame, index = seen[0]
    state = np.concatenate([frames[previous_frame][index], np.zeros(dim)])
    covariance = fresh_covariance.copy()
    cost = 0.0
    for frame, index in seen[1:]:
        for _ in range(frame - previous_frame):
            state, covariance = predict(state, covariance, model)
        innovation_cov = (
            model.observation @ covariance @ model.observation.T
            + model.measurement_noise
        )
        error = frames[frame][index] - model.observation @ state
        cost += float(error @ np.linalg.solve(innovation_cov, error))
        cost += float(np.linalg.slogdet(innovation_cov)[1])
        state, covariance = update(state, covariance, frames[frame][index], model)
        previous_frame = frame
    return cost


def _position(track, frame, frames):
    index = track[frame]
    return None if index < 0 else frames[frame][index]


def _close_cuts(a, b, frames, radius):
    """Frames where swapping the tails of ``a`` and ``b`` is worth testing."""
    cuts = []
    for cut in range(1, len(a)):
        for frame_a, frame_b in ((cut - 1, cut), (cut, cut - 1), (cut, cut)):
            point_a = _position(a, frame_a, frames)
            point_b = _position(b, frame_b, frames)
            if (
                point_a is not None
                and point_b is not None
                and np.linalg.norm(point_a - point_b) <= radius
            ):
                cuts.append(cut)
                break
    return cuts


def resolve_crossings(
    tracks: Sequence[np.ndarray],
    points: Sequence[np.ndarray],
    *,
    measurement_sigma: float = 0.2,
    acceleration_sigma: float = 0.1,
    speed_sigma: float = 1.0,
    swap_radius: float = 2.0,
    margin: float = 0.0,
    max_segment: int = 4,
    max_swaps: int = 50,
) -> tuple[list[np.ndarray], int]:
    """Swap parts of two tracks while that makes the tracks more plausible.

    A swap exchanges what two tracks hold from a frame on (a *tail*), or only
    for up to ``max_segment`` frames (a brief flip that corrects itself).

    ``swap_radius``: only pairs of tracks passing this close are considered.
    ``margin``: a swap must lower the total cost by more than this.
    Returns ``(tracks, number_of_swaps_made)``.
    """
    frames = [np.asarray(frame, dtype=float) for frame in points]
    dim = frames[0].shape[1]
    model = make_motion_model(
        dim, measurement_sigma=measurement_sigma, acceleration_sigma=acceleration_sigma
    )
    fresh = np.diag([measurement_sigma**2] * dim + [speed_sigma**2] * dim)
    current = [np.array(track) for track in tracks]
    costs = [track_cost(t, frames, model, fresh) for t in current]

    swaps = 0
    while swaps < max_swaps:
        best_gain, best = margin, None
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                for cut in _close_cuts(current[i], current[j], frames, swap_radius):
                    ends = [len(current[i])] + list(range(cut + 1, cut + max_segment))
                    for end in ends:
                        if end > len(current[i]):
                            continue
                        new_i, new_j = current[i].copy(), current[j].copy()
                        new_i[cut:end] = current[j][cut:end]
                        new_j[cut:end] = current[i][cut:end]
                        gain = (
                            costs[i]
                            + costs[j]
                            - track_cost(new_i, frames, model, fresh)
                            - track_cost(new_j, frames, model, fresh)
                        )
                        if gain > best_gain:
                            best_gain, best = gain, (i, j, new_i, new_j)
        if best is None:
            break
        i, j, current[i], current[j] = best
        costs[i] = track_cost(current[i], frames, model, fresh)
        costs[j] = track_cost(current[j], frames, model, fresh)
        swaps += 1
    return current, swaps


__all__ = ["resolve_crossings", "track_cost"]
