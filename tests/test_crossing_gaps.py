import numpy as np
import pytest

from crossing_gaps import (
    N_FRAMES,
    gap_candidates,
    identities,
    is_perfect,
    make_scene,
    recovery_rate,
)
from simpletracker import (
    build_tracks,
    close_gaps,
    frame_offsets,
    link_frames,
    simpletracker,
)


def test_scene_is_3d_with_the_requested_missing_detections():
    scene = make_scene(b_missing=2)

    assert len(scene.frames) == N_FRAMES
    assert scene.paths.shape == (3, N_FRAMES, 3)
    assert all(frame.shape[1] == 3 for frame in scene.frames)
    assert scene.missing == [(4, 1), (5, 1), (3, 2)]
    assert [len(frame) for frame in scene.frames][3:7] == [2, 2, 2, 3]


def test_step_one_alone_leaves_fragments_where_detections_are_missing():
    scene = make_scene()

    links, _, _ = link_frames(scene.frames, max_linking_distance=4.0)
    tracks, _ = build_tracks(links, scene.frames)

    assert len(tracks) == 5
    assert not is_perfect(tracks, scene.truth)


def test_gap_closing_joins_the_fragments_into_one_track_per_particle():
    scene = make_scene()
    links, unmatched_sources, unmatched_targets = link_frames(
        scene.frames, max_linking_distance=4.0
    )

    gap_links = close_gaps(
        scene.frames,
        unmatched_sources,
        unmatched_targets,
        max_linking_distance=4.0,
        max_gap_closing=3,
    )
    tracks, _ = build_tracks({**links, **gap_links}, scene.frames)

    assert len(gap_links) == 2
    assert is_perfect(tracks, scene.truth)
    assert sorted(identities(tracks, scene.truth)) == [
        "AAAAAAAAAAAA",
        "BBBB--BBBBBB",
        "CCC-CCCCCCCC",
    ]


def test_steps_compose_into_simpletracker():
    scene = make_scene(separation=0.5, noise=0.2, seed=7)
    kwargs = dict(max_linking_distance=4.0, max_gap_closing=3)

    expected_tracks, expected_chains, expected_adjacency = simpletracker(
        scene.frames, **kwargs
    )
    links, unmatched_sources, unmatched_targets = link_frames(
        scene.frames, max_linking_distance=4.0
    )
    links.update(
        close_gaps(scene.frames, unmatched_sources, unmatched_targets, **kwargs)
    )
    tracks, chains = build_tracks(links, scene.frames)

    for got, want in zip(tracks, expected_tracks):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(chains, expected_chains):
        np.testing.assert_array_equal(got, want)
    assert len(tracks) == len(expected_tracks)
    assert int(frame_offsets(scene.frames)[-1]) == expected_adjacency.shape[0]


@pytest.mark.parametrize(
    "missing, max_gap_closing, expected",
    [(2, 2, 0.0), (2, 3, 1.0), (1, 1, 0.0), (1, 2, 1.0)],
)
def test_a_gap_of_k_frames_needs_max_gap_closing_of_k_plus_one(
    missing, max_gap_closing, expected
):
    rate = recovery_rate(
        b_missing=missing,
        max_gap_closing=max_gap_closing,
        max_linking_distance=4.0,
        n_trials=10,
    )

    assert rate == expected


def test_distance_limit_applies_to_the_whole_jump_across_a_gap():
    too_small = recovery_rate(b_missing=2, max_linking_distance=2.0, n_trials=10)
    big_enough = recovery_rate(b_missing=2, max_linking_distance=4.0, n_trials=10)

    assert too_small == 0.0
    assert big_enough == 1.0


def test_touching_particles_cannot_be_told_apart():
    touching = recovery_rate(separation=0.0, n_trials=20)
    apart = recovery_rate(separation=3.0, n_trials=20)

    assert touching < 0.2
    assert apart > 0.9


def test_gap_candidates_include_the_true_pair_and_respect_the_reach():
    scene = make_scene()
    offsets = frame_offsets(scene.frames)
    _, unmatched_sources, unmatched_targets = link_frames(
        scene.frames, max_linking_distance=4.0
    )

    reach_three = gap_candidates(
        scene.frames, offsets, unmatched_sources, unmatched_targets, 3
    )
    reach_two = gap_candidates(
        scene.frames, offsets, unmatched_sources, unmatched_targets, 2
    )

    def same_particle(item):
        source, target, source_frame, target_frame, _ = item
        local_source = source - offsets[source_frame]
        local_target = target - offsets[target_frame]
        return (
            scene.truth[source_frame][local_source]
            == scene.truth[target_frame][local_target]
        )

    assert any(same_particle(c) and c[3] - c[2] == 3 for c in reach_three)
    assert all(c[3] - c[2] <= 2 for c in reach_two)
    assert [c[4] for c in reach_three] == sorted(c[4] for c in reach_three)
