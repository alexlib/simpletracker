import numpy as np
import pytest

from crossing_demo import (
    has_identity_swap,
    make_crossing_frames,
    swap_rate,
    track_identities,
)
from simpletracker import simpletracker


def test_frames_are_3d_and_truth_matches_shape():
    frames, truth = make_crossing_frames(separation=2.0)

    assert len(frames) == len(truth) == 11
    assert all(frame.shape == (2, 3) for frame in frames)
    assert all(sorted(order) == [0, 1] for order in truth)


@pytest.mark.parametrize("method", ["hungarian", "nearestneighbor"])
def test_well_separated_crossing_keeps_identities(method):
    frames, truth = make_crossing_frames(separation=4.0)

    tracks, _, _ = simpletracker(frames, method=method, max_linking_distance=3.0)

    assert not has_identity_swap(tracks, truth)
    assert sorted(track_identities(tracks, truth)) == ["A" * 11, "B" * 11]


@pytest.mark.parametrize("method", ["hungarian", "nearestneighbor"])
def test_touching_crossing_swaps_identities(method):
    frames, truth = make_crossing_frames(separation=0.0)

    tracks, _, _ = simpletracker(frames, method=method, max_linking_distance=3.0)

    assert has_identity_swap(tracks, truth)


def test_gap_closing_reconnects_a_dropped_detection_at_the_crossing():
    frames, truth = make_crossing_frames(separation=3.0, drop=(5, 1))

    without_closing, _, _ = simpletracker(
        frames, max_linking_distance=2.5, max_gap_closing=1
    )
    with_closing, _, _ = simpletracker(
        frames, max_linking_distance=2.5, max_gap_closing=2
    )

    assert len(without_closing) == 3
    assert sorted(track_identities(with_closing, truth)) == ["A" * 11, "BBBBB-BBBBB"]


def test_swap_rate_falls_as_particles_pass_further_apart():
    close = swap_rate(0.25, noise=0.1, n_trials=40)
    far = swap_rate(4.0, noise=0.1, n_trials=40)

    assert close > 0.5
    assert far == 0.0
    assert np.isfinite(close)
