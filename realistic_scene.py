"""A noisy 3D scene with many particles, near-miss crossings and clutter.

Every particle moves with a nearly constant velocity that wiggles a little.
Some pairs are built to meet: they pass within ``miss_distance`` of each other
in the middle of the movie, flying in different directions. On top of that the
detector is imperfect:

* positions have Gaussian ``noise``,
* each particle is missed with probability ``1 - detection_prob``,
* ``clutter`` false detections per frame appear at random places.

The real identity of every detection is kept in ``truth`` (``-1`` for clutter)
so tracks can be scored with :func:`score_tracks`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RealisticScene:
    frames: list[np.ndarray]
    truth: list[np.ndarray]
    paths: np.ndarray
    n_particles: int


def _simulate(position, velocity, meet_frame, n_frames, wiggle, rng):
    """Integrate a path outwards from the frame where it is at ``position``."""
    path = np.zeros((n_frames, len(position)))
    path[meet_frame] = position
    speed = np.array(velocity, dtype=float)
    for frame in range(meet_frame, n_frames - 1):
        path[frame + 1] = path[frame] + speed
        speed = speed + rng.normal(0.0, wiggle, speed.shape)
    speed = np.array(velocity, dtype=float)
    for frame in range(meet_frame, 0, -1):
        path[frame - 1] = path[frame] - speed
        speed = speed + rng.normal(0.0, wiggle, speed.shape)
    return path


def _random_direction(rng, dim=3):
    direction = rng.normal(size=dim)
    return direction / np.linalg.norm(direction)


def make_realistic_scene(
    n_particles: int = 10,
    n_frames: int = 25,
    *,
    noise: float = 0.2,
    detection_prob: float = 0.9,
    clutter: float = 1.0,
    crossing_pairs: int = 3,
    miss_distance: float = 0.3,
    speed: float = 1.0,
    wiggle: float = 0.05,
    box: float = 12.0,
    seed: int = 0,
) -> RealisticScene:
    """Build a scene. ``crossing_pairs`` pairs meet at the middle frame.

    ``paths[p, f]`` is the true position of particle ``p`` in frame ``f``.
    """
    rng = np.random.default_rng(seed)
    middle = n_frames // 2
    paths = []
    for pair in range(min(crossing_pairs, n_particles // 2)):
        meeting_point = rng.uniform(0.35 * box, 0.65 * box, 3)
        first = _random_direction(rng)
        second = _random_direction(rng)
        while abs(first @ second) > 0.8:
            second = _random_direction(rng)
        offset = _random_direction(rng) * miss_distance
        for direction, shift in ((first, 0.0), (second, 1.0)):
            velocity = direction * speed * rng.uniform(0.8, 1.2)
            paths.append(
                _simulate(
                    meeting_point + shift * offset,
                    velocity,
                    middle,
                    n_frames,
                    wiggle,
                    rng,
                )
            )
    while len(paths) < n_particles:
        velocity = _random_direction(rng) * speed * rng.uniform(0.8, 1.2)
        paths.append(
            _simulate(rng.uniform(0, box, 3), velocity, 0, n_frames, wiggle, rng)
        )
    paths = np.stack(paths)

    low = paths.reshape(-1, 3).min(axis=0) - 1.0
    high = paths.reshape(-1, 3).max(axis=0) + 1.0
    frames, truth = [], []
    for frame in range(n_frames):
        seen = np.flatnonzero(rng.random(len(paths)) < detection_prob)
        points = paths[seen, frame] + rng.normal(0.0, noise, (len(seen), 3))
        ids = seen.copy()
        n_false = rng.poisson(clutter)
        if n_false:
            points = np.vstack([points, rng.uniform(low, high, (n_false, 3))])
            ids = np.concatenate([ids, np.full(n_false, -1)])
        order = rng.permutation(len(points))
        frames.append(points[order])
        truth.append(ids[order])
    return RealisticScene(frames, truth, paths, len(paths))


def _true_links(truth: Sequence[np.ndarray]) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    """Links between consecutive real detections of the same particle."""
    last: dict[int, tuple[int, int]] = {}
    links = set()
    for frame, ids in enumerate(truth):
        for index, particle in enumerate(ids):
            if particle < 0:
                continue
            if int(particle) in last:
                links.add((last[int(particle)], (frame, index)))
            last[int(particle)] = (frame, index)
    return links


def score_tracks(
    tracks: Sequence[np.ndarray], truth: Sequence[np.ndarray]
) -> dict[str, float]:
    """Score tracks by the links they contain.

    * ``precision``: fraction of the tracker's links that are real.
    * ``recall``: fraction of the real links the tracker found.
    * ``switches``: links joining two *different* real particles
      (an identity swap).
    * ``clutter_links``: links that touch a false detection.
    * ``f1``: harmonic mean of precision and recall.
    """
    wanted = _true_links(truth)
    found = set()
    for track in tracks:
        chain = [(frame, int(index)) for frame, index in enumerate(track) if index >= 0]
        found.update(zip(chain[:-1], chain[1:]))
    correct = len(found & wanted)
    switches = clutter_links = 0
    for (frame_a, index_a), (frame_b, index_b) in found:
        a, b = truth[frame_a][index_a], truth[frame_b][index_b]
        if a < 0 or b < 0:
            clutter_links += 1
        elif a != b:
            switches += 1
    precision = correct / len(found) if found else 1.0
    recall = correct / len(wanted) if wanted else 1.0
    f1 = 2 * precision * recall / (precision + recall) if correct else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "switches": float(switches),
        "clutter_links": float(clutter_links),
    }


__all__ = ["RealisticScene", "make_realistic_scene", "score_tracks"]
