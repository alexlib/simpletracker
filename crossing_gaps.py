"""A 3D scene with crossing trajectories and missing detections.

Three particles move in straight lines:

* **A** and **B** fly towards each other and cross in the middle of the movie.
  ``separation`` is how far apart in z they are when they cross.
* **C** flies on its own, far from the other two.

Detections go missing on purpose: **B** disappears for a few frames exactly
while it crosses A, and **C** blinks out for one frame. The real identity of
every detection is kept in ``truth`` so results can be checked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from simpletracker import simpletracker

PARTICLE_NAMES = "ABC"
N_FRAMES = 12
B_MISSING_START = 4
C_BLINK_FRAME = 3


@dataclass(frozen=True)
class Scene:
    frames: list[np.ndarray]
    truth: list[np.ndarray]
    paths: np.ndarray
    missing: list[tuple[int, int]]


def make_scene(
    separation: float = 2.0,
    noise: float = 0.1,
    seed: int = 1,
    b_missing: int = 2,
    c_blinks: bool = True,
) -> Scene:
    """Build the scene.

    ``truth[f][i]`` is the particle (0=A, 1=B, 2=C) behind row ``i`` of
    ``frames[f]``. ``paths[p, f]`` is the true position of particle ``p`` in
    frame ``f``. ``missing`` lists the ``(frame, particle)`` detections that
    were removed.
    """
    t = np.linspace(0.0, 1.0, N_FRAMES)[:, None]
    path_a = t * np.array([11.0, 3.0, 0.0])
    path_b = np.array([11.0, 0.0, separation]) + t * np.array([-11.0, 3.0, 0.0])
    path_c = np.array([2.0, 6.0, 4.0]) + t * np.array([7.0, -1.0, -1.0])
    paths = np.stack([path_a, path_b, path_c])

    missing = [(B_MISSING_START + k, 1) for k in range(b_missing)]
    if c_blinks:
        missing.append((C_BLINK_FRAME, 2))

    rng = np.random.default_rng(seed)
    frames, truth = [], []
    for frame_index in range(N_FRAMES):
        order = rng.permutation(len(paths))
        points = paths[order, frame_index] + rng.normal(0.0, noise, (len(paths), 3))
        hidden = {p for f, p in missing if f == frame_index}
        keep = np.array([p not in hidden for p in order])
        frames.append(points[keep])
        truth.append(order[keep])
    return Scene(frames, truth, paths, missing)


def identities(tracks: Sequence[np.ndarray], truth: Sequence[np.ndarray]) -> list[str]:
    """Describe each track by the true particle behind each frame ('-' = gap)."""
    return [
        "".join(
            "-" if index < 0 else PARTICLE_NAMES[truth[frame][index]]
            for frame, index in enumerate(track)
        )
        for track in tracks
    ]


def is_perfect(
    tracks: Sequence[np.ndarray], truth: Sequence[np.ndarray], n_particles: int = 3
) -> bool:
    """True if every track follows one particle and each particle has one track."""
    owners: dict[int, int] = {}
    for track in tracks:
        particles = {
            int(truth[frame][index]) for frame, index in enumerate(track) if index >= 0
        }
        if len(particles) != 1:
            return False
        particle = particles.pop()
        owners[particle] = owners.get(particle, 0) + 1
    return len(owners) == n_particles and all(count == 1 for count in owners.values())


def recovery_rate(
    *,
    separation: float = 2.0,
    noise: float = 0.1,
    b_missing: int = 2,
    max_gap_closing: float = 3,
    max_linking_distance: float = 4.0,
    method: str = "hungarian",
    n_trials: int = 30,
) -> float:
    """Fraction of random scenes where the tracker recovers the true tracks."""
    recovered = 0
    for seed in range(n_trials):
        scene = make_scene(separation, noise, seed, b_missing)
        tracks, _, _ = simpletracker(
            scene.frames,
            method=method,
            max_linking_distance=max_linking_distance,
            max_gap_closing=max_gap_closing,
        )
        recovered += is_perfect(tracks, scene.truth)
    return recovered / n_trials


def gap_candidates(
    frames: Sequence[np.ndarray],
    offsets: np.ndarray,
    unmatched_sources: Sequence[np.ndarray],
    unmatched_targets: Sequence[np.ndarray],
    max_gap_closing: float,
) -> list[tuple[int, int, int, int, float]]:
    """List every track end / later track start pair the gap closing may consider.

    Each item is ``(source_global, target_global, source_frame, target_frame,
    distance)``, sorted by distance. The distance limit is not applied here.
    """
    n_frames = len(frames)
    reach = n_frames - 1 if np.isinf(max_gap_closing) else int(max_gap_closing)
    candidates = []
    for source_frame in range(n_frames - 2):
        for source_index in unmatched_sources[source_frame]:
            for target_frame in range(
                source_frame + 2, min(n_frames, source_frame + reach + 1)
            ):
                for target_index in unmatched_targets[target_frame]:
                    distance = float(
                        np.linalg.norm(
                            frames[source_frame][source_index]
                            - frames[target_frame][target_index]
                        )
                    )
                    candidates.append(
                        (
                            int(offsets[source_frame] + source_index),
                            int(offsets[target_frame] + target_index),
                            source_frame,
                            target_frame,
                            distance,
                        )
                    )
    return sorted(candidates, key=lambda item: item[4])
