"""Helpers for studying the tracker on 3D scenes with crossing trajectories.

Two particles A and B move in straight lines and cross at frame ``n_frames // 2``.
``separation`` is the distance between them in z at the crossing: 0 means they
touch, large values mean they pass far apart.
"""

from __future__ import annotations

import numpy as np

from simpletracker import simpletracker

PARTICLE_NAMES = "AB"


def make_crossing_frames(
    separation: float,
    noise: float = 0.0,
    seed: int = 0,
    n_frames: int = 11,
    drop: tuple[int, int] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build shuffled 3D detections of two crossing particles.

    Returns ``(frames, truth)``. ``truth[f][i]`` is 0 (A) or 1 (B) for row ``i``
    of ``frames[f]``, so results can be checked against the real identities.
    ``drop=(frame, particle)`` removes one detection to create a gap.
    """
    t = np.arange(n_frames, dtype=float)[:, None]
    last = n_frames - 1
    a = np.hstack([t, 0.3 * t, np.full_like(t, separation / 2)])
    b = np.hstack([last - t, 0.3 * last - 0.3 * t, np.full_like(t, -separation / 2)])
    rng = np.random.default_rng(seed)
    frames, truth = [], []
    for frame_index in range(n_frames):
        order = rng.permutation(2)
        points = np.vstack([a[frame_index], b[frame_index]])[order]
        points = points + rng.normal(0.0, noise, points.shape)
        if drop is not None and frame_index == drop[0]:
            keep = order != drop[1]
            points, order = points[keep], order[keep]
        frames.append(points)
        truth.append(order)
    return frames, truth


def track_identities(tracks: list[np.ndarray], truth: list[np.ndarray]) -> list[str]:
    """Describe each track as a string of true particle names ('-' for a gap)."""
    return [
        "".join(
            "-" if index < 0 else PARTICLE_NAMES[truth[frame][index]]
            for frame, index in enumerate(track)
        )
        for track in tracks
    ]


def has_identity_swap(tracks: list[np.ndarray], truth: list[np.ndarray]) -> bool:
    """True if any track follows more than one real particle."""
    return any(
        len({truth[frame][index] for frame, index in enumerate(track) if index >= 0}) > 1
        for track in tracks
    )


def swap_rate(
    separation: float,
    noise: float,
    method: str = "hungarian",
    n_trials: int = 100,
    max_linking_distance: float = 3.0,
) -> float:
    """Fraction of random trials in which the tracker mixes up the particles."""
    swaps = 0
    for seed in range(n_trials):
        frames, truth = make_crossing_frames(separation, noise, seed)
        tracks, _, _ = simpletracker(
            frames, method=method, max_linking_distance=max_linking_distance
        )
        swaps += has_identity_swap(tracks, truth)
    return swaps / n_trials
