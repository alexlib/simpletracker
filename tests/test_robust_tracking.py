import numpy as np
import pytest

from crossing_resolver import resolve_crossings, track_cost
from kalman_tracker import (
    assign,
    kalman_tracker,
    mahalanobis_costs,
    make_motion_model,
    predict,
    update,
)
from realistic_scene import make_realistic_scene, score_tracks
from simpletracker import simpletracker

OPTIONS = dict(measurement_sigma=0.05, acceleration_sigma=0.05)


def steep_crossing(n_frames=9):
    """Two particles crossing at a steep angle; index 0 is A, index 1 is B.

    Between frames 4 and 5 the *wrong* pairing is shorter, so a tracker that
    only looks at distances swaps them.
    """
    t = np.arange(n_frames, dtype=float)
    a = np.stack([t, 0.8 * t], axis=1)
    b = np.stack([t, 6.6 - 0.8 * t], axis=1)
    frames = [np.stack([a[f], b[f]]) for f in range(n_frames)]
    truth = [np.array([0, 1]) for _ in range(n_frames)]
    return frames, truth


def test_distance_only_swaps_but_motion_prediction_does_not():
    frames, truth = steep_crossing()
    swapped, _, _ = simpletracker(frames, max_linking_distance=3.0, max_gap_closing=1)
    assert score_tracks(swapped, truth)["switches"] > 0

    tracks, _, _ = kalman_tracker(frames, **OPTIONS)
    assert score_tracks(tracks, truth)["switches"] == 0
    assert sorted(t.tolist() for t in tracks) == [[0] * 9, [1] * 9]


def test_coasting_bridges_a_missing_detection():
    frames, truth = steep_crossing()
    frames[6] = frames[6][:1]
    truth[6] = truth[6][:1]
    tracks, _, _ = kalman_tracker(frames, max_missed=2, **OPTIONS)
    assert len(tracks) == 2
    gapped = next(t for t in tracks if t[6] == -1)
    assert gapped[5] >= 0 and gapped[7] >= 0
    assert score_tracks(tracks, truth)["switches"] == 0


def test_too_long_a_gap_splits_the_track():
    frames, _ = steep_crossing()
    for frame in (4, 5, 6):
        frames[frame] = frames[frame][:1]
    tracks, _, _ = kalman_tracker(frames, max_missed=1, **OPTIONS)
    assert len(tracks) == 3


def test_a_far_false_detection_does_not_disturb_real_tracks():
    frames, truth = steep_crossing()
    frames[3] = np.vstack([frames[3], [[50.0, 50.0]]])
    truth[3] = np.append(truth[3], -1)
    tracks, _, _ = kalman_tracker(frames, **OPTIONS)
    real = [t for t in tracks if np.count_nonzero(t >= 0) > 1]
    assert len(real) == 2
    assert score_tracks(tracks, truth)["clutter_links"] == 0
    assert len(kalman_tracker(frames, min_length=2, **OPTIONS)[0]) == 2


def test_filter_learns_velocity_and_uncertainty_shrinks():
    model = make_motion_model(2, measurement_sigma=0.05, acceleration_sigma=0.01)
    state = np.array([0.0, 0.0, 0.0, 0.0])
    covariance = np.diag([0.0025, 0.0025, 1.0, 1.0])
    for step in range(1, 8):
        state, covariance = predict(state, covariance, model)
        state, covariance = update(state, covariance, np.array([step * 1.0, step * 0.5]), model)
    assert state[2:] == pytest.approx([1.0, 0.5], abs=0.05)
    assert covariance[2, 2] < 0.1


def test_uncertainty_grows_while_coasting():
    model = make_motion_model(2)
    covariance = np.eye(4) * 0.1
    _, grown = predict(np.zeros(4), covariance, model)
    assert np.trace(grown) > np.trace(covariance)


def test_mahalanobis_costs_scale_with_uncertainty():
    model = make_motion_model(2, measurement_sigma=1.0)
    states = np.zeros((2, 4))
    covariances = np.stack([np.eye(4) * 0.0, np.eye(4) * 3.0])
    costs = mahalanobis_costs(states, covariances, np.array([[2.0, 0.0]]), model)
    assert costs[0, 0] == pytest.approx(4.0)
    assert costs[1, 0] == pytest.approx(1.0)


def test_assign_respects_gate_and_total_cost():
    costs = np.array([[1.0, 2.0], [1.5, 10.0]])
    assert assign(costs, gate=5.0).tolist() == [1, 0]
    assert assign(costs, gate=1.2).tolist() == [0, -1]
    assert assign(np.empty((0, 3)), gate=1.0).size == 0


def test_smooth_track_costs_less_than_kinked_track():
    frames, _ = steep_crossing()
    model = make_motion_model(2, **OPTIONS)
    fresh = np.diag([0.0025, 0.0025, 1.0, 1.0])
    smooth = np.zeros(9, dtype=int)
    kinked = smooth.copy()
    kinked[5:] = 1
    assert track_cost(smooth, frames, model, fresh) < track_cost(kinked, frames, model, fresh)


def test_resolver_repairs_a_brief_flip_and_a_permanent_swap():
    frames, truth = steep_crossing()
    perfect = [np.zeros(9, dtype=int), np.ones(9, dtype=int)]
    for start, end in ((5, 6), (5, 9)):
        broken = [t.copy() for t in perfect]
        broken[0][start:end], broken[1][start:end] = 1, 0
        assert score_tracks(broken, truth)["switches"] > 0
        fixed, swaps = resolve_crossings(broken, frames, swap_radius=2.0, **OPTIONS)
        assert swaps >= 1
        assert score_tracks(fixed, truth)["switches"] == 0


def test_resolver_leaves_correct_tracks_alone():
    frames, _ = steep_crossing()
    perfect = [np.zeros(9, dtype=int), np.ones(9, dtype=int)]
    fixed, swaps = resolve_crossings(perfect, frames, **OPTIONS)
    assert swaps == 0
    assert all(np.array_equal(a, b) for a, b in zip(fixed, perfect))


def test_pipeline_beats_distance_only_on_noisy_crossings():
    totals = {"distance": 0.0, "kalman": 0.0, "resolved": 0.0}
    for seed in range(8):
        scene = make_realistic_scene(noise=0.2, seed=seed)
        options = dict(measurement_sigma=0.2, acceleration_sigma=0.1)
        base, _, _ = simpletracker(scene.frames, max_linking_distance=3.0, max_gap_closing=3)
        moving, _, _ = kalman_tracker(scene.frames, speed_sigma=0.7, **options)
        resolved, _ = resolve_crossings(moving, scene.frames, **options)
        totals["distance"] += score_tracks(base, scene.truth)["switches"]
        totals["kalman"] += score_tracks(moving, scene.truth)["switches"]
        totals["resolved"] += score_tracks(resolved, scene.truth)["switches"]
    assert totals["kalman"] < totals["distance"] / 2
    assert totals["resolved"] <= totals["kalman"]
