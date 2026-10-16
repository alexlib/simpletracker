"""A motion-aware tracker: constant-velocity Kalman filter + Hungarian matching.

The distance-only tracker in :mod:`simpletracker` asks "which dot is *closest*
to where this particle *was*?". That breaks when particles cross, because the
wrong neighbour can be closer than the right one, and noise makes distances
unreliable.

This tracker asks a better question: "which dot is closest to where this
particle *should be now*, given how fast it was moving?" It keeps, for every
track, an estimate of position and velocity together with how uncertain that
estimate is:

1. **Predict** where every track will be in the new frame.
2. **Compare** every dot with every prediction using the *Mahalanobis*
   distance: how many standard deviations away the dot is, so uncertain
   predictions tolerate bigger jumps.
3. **Gate**: dots that are statistically too far are ignored.
4. **Match** with the Hungarian algorithm: veteran tracks (velocity known)
   choose first, young tracks get what is left.
5. **Update** the matched tracks with their new dot. Unmatched tracks keep
   *coasting* on their prediction for up to ``max_missed`` frames, which
   bridges gaps. Unmatched dots start new tracks.

The return values follow :func:`simpletracker.simpletracker`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from simpletracker import (
    Points,
    _validate_points,
    build_tracks,
    frame_offsets,
    links_to_adjacency,
)


@dataclass(frozen=True)
class MotionModel:
    """Matrices of a constant-velocity model in ``dim`` dimensions.

    The state is ``[position..., velocity...]``.
    """

    transition: np.ndarray
    process_noise: np.ndarray
    observation: np.ndarray
    measurement_noise: np.ndarray


def make_motion_model(
    dim: int = 3,
    *,
    dt: float = 1.0,
    measurement_sigma: float = 0.2,
    acceleration_sigma: float = 0.1,
) -> MotionModel:
    """Constant velocity with random accelerations (white-noise acceleration)."""
    eye, zero = np.eye(dim), np.zeros((dim, dim))
    transition = np.block([[eye, dt * eye], [zero, eye]])
    process_noise = acceleration_sigma**2 * np.block(
        [[dt**4 / 4 * eye, dt**3 / 2 * eye], [dt**3 / 2 * eye, dt**2 * eye]]
    )
    observation = np.hstack([eye, zero])
    measurement_noise = measurement_sigma**2 * eye
    return MotionModel(transition, process_noise, observation, measurement_noise)


def predict(state: np.ndarray, covariance: np.ndarray, model: MotionModel):
    """Move the estimate one step forward; uncertainty grows."""
    state = model.transition @ state
    covariance = model.transition @ covariance @ model.transition.T + model.process_noise
    return state, covariance


def update(state, covariance, measurement, model: MotionModel):
    """Correct a prediction with a measured position; uncertainty shrinks."""
    innovation = measurement - model.observation @ state
    innovation_cov = (
        model.observation @ covariance @ model.observation.T + model.measurement_noise
    )
    gain = covariance @ model.observation.T @ np.linalg.inv(innovation_cov)
    state = state + gain @ innovation
    covariance = (np.eye(len(state)) - gain @ model.observation) @ covariance
    return state, covariance


def mahalanobis_costs(
    states: np.ndarray, covariances: np.ndarray, detections: np.ndarray, model: MotionModel
) -> np.ndarray:
    """Squared Mahalanobis distance of every detection to every predicted track.

    ``costs[i, j]`` says how many "sigmas" (squared) detection ``j`` is from
    where track ``i`` is predicted to be.
    """
    costs = np.empty((len(states), len(detections)))
    for i, (state, covariance) in enumerate(zip(states, covariances)):
        innovation_cov = (
            model.observation @ covariance @ model.observation.T
            + model.measurement_noise
        )
        difference = detections - model.observation @ state
        solved = np.linalg.solve(innovation_cov, difference.T)
        costs[i] = np.einsum("ij,ji->i", difference, solved)
    return costs


def assign(costs: np.ndarray, gate: float) -> np.ndarray:
    """Hungarian matching that refuses pairs costing more than ``gate``.

    Returns, for every row, the matched column or ``-1``. Every row also owns a
    "stay unmatched" option that costs exactly ``gate``, so a pair beyond the
    gate can never win.
    """
    rows, cols = costs.shape
    matches = np.full(rows, -1, dtype=int)
    if rows == 0 or cols == 0:
        return matches
    padded = np.full((rows, cols + rows), gate)
    padded[:, :cols] = costs
    for row, column in zip(*linear_sum_assignment(padded)):
        if column < cols and costs[row, column] <= gate:
            matches[row] = column
    return matches


@dataclass
class _Track:
    state: np.ndarray
    covariance: np.ndarray
    last_point: int
    hits: int = 1
    missed: int = 0


def kalman_links(
    points: Points,
    *,
    measurement_sigma: float = 0.2,
    acceleration_sigma: float = 0.1,
    speed_sigma: float = 1.0,
    gate_probability: float = 0.99,
    max_missed: int = 3,
) -> dict[int, int]:
    """Link detections frame by frame; returns ``{source_global: target_global}``.

    ``measurement_sigma``: how noisy the detector is (same units as positions).
    ``acceleration_sigma``: how much a particle's velocity may change per frame.
    ``speed_sigma``: how fast a brand new particle may be moving.
    ``gate_probability``: a dot is only considered if it falls inside this
    probability region around the prediction.
    ``max_missed``: how many frames a track may coast without a dot.
    """
    frames = _validate_points(points)
    dim = frames[0].shape[1]
    offsets = frame_offsets(frames)
    model = make_motion_model(
        dim, measurement_sigma=measurement_sigma, acceleration_sigma=acceleration_sigma
    )
    gate = float(chi2.ppf(gate_probability, df=dim))
    fresh_covariance = np.diag(
        [measurement_sigma**2] * dim + [speed_sigma**2] * dim
    )

    links: dict[int, int] = {}
    active: list[_Track] = []
    for frame_index, detections in enumerate(frames):
        predicted = [predict(t.state, t.covariance, model) for t in active]
        for track, (state, covariance) in zip(active, predicted):
            track.state, track.covariance = state, covariance

        matches = np.full(len(active), -1, dtype=int)
        free = np.arange(len(detections))
        for veterans in (True, False):
            rows = np.array(
                [i for i, t in enumerate(active) if (t.hits >= 2) == veterans],
                dtype=int,
            )
            if not len(rows) or not len(free):
                continue
            costs = mahalanobis_costs(
                np.array([active[i].state for i in rows]),
                np.array([active[i].covariance for i in rows]),
                detections[free],
                model,
            )
            chosen = assign(costs, gate)
            for row, column in zip(rows, chosen):
                if column >= 0:
                    matches[row] = free[column]
            free = np.setdiff1d(free, matches[matches >= 0])

        survivors = []
        for track, match in zip(active, matches):
            if match >= 0:
                track.state, track.covariance = update(
                    track.state, track.covariance, detections[match], model
                )
                target = int(offsets[frame_index] + match)
                links[track.last_point] = target
                track.last_point = target
                track.hits += 1
                track.missed = 0
            else:
                track.missed += 1
            if track.missed <= max_missed:
                survivors.append(track)
        active = survivors
        for index in free:
            state = np.concatenate([detections[index], np.zeros(dim)])
            active.append(
                _Track(state, fresh_covariance.copy(), int(offsets[frame_index] + index))
            )
    return links


def kalman_tracker(
    points: Points,
    *,
    min_length: int = 1,
    **kwargs,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Track detections with motion prediction.

    Accepts the options of :func:`kalman_links`. Tracks with fewer than
    ``min_length`` detections are dropped (they are usually clutter).
    Returns ``(tracks, adjacency_tracks, adjacency)`` like ``simpletracker``.
    """
    frames = _validate_points(points)
    links = kalman_links(frames, **kwargs)
    tracks, adjacency_tracks = build_tracks(links, frames)
    keep = [i for i, t in enumerate(tracks) if np.count_nonzero(t >= 0) >= min_length]
    adjacency = links_to_adjacency(links, int(frame_offsets(frames)[-1]))
    return [tracks[i] for i in keep], [adjacency_tracks[i] for i in keep], adjacency


__all__ = [
    "MotionModel",
    "assign",
    "kalman_links",
    "kalman_tracker",
    "mahalanobis_costs",
    "make_motion_model",
    "predict",
    "update",
]
