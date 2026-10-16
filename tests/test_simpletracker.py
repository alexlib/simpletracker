import numpy as np

from simpletracker import (
    hungarian_linker,
    nearest_neighbor_linker,
    simpletracker,
)


def test_hungarian_linker_respects_distance_and_exclusivity():
    source = np.array([[0.0], [10.0]])
    target = np.array([[0.5], [20.0]])

    indices, distances, unassigned = hungarian_linker(source, target, max_distance=1)

    np.testing.assert_array_equal(indices, [0, -1])
    np.testing.assert_allclose(distances[0], 0.5)
    np.testing.assert_array_equal(unassigned, [1])


def test_nearest_neighbor_linker_matches_closest_pairs():
    source = np.array([[0.0], [3.0]])
    target = np.array([[2.0], [10.0]])

    indices, _, _ = nearest_neighbor_linker(source, target)

    np.testing.assert_array_equal(indices, [1, 0])


def test_simpletracker_closes_a_missing_frame():
    points = [
        np.array([[0.0, 0.0], [10.0, 0.0]]),
        np.array([[10.5, 0.0]]),
        np.array([[2.0, 0.0], [11.0, 0.0]]),
    ]

    tracks, adjacency_tracks, adjacency = simpletracker(
        points, max_linking_distance=2, max_gap_closing=2
    )

    assert len(tracks) == 2
    np.testing.assert_array_equal(tracks[0], [0, -1, 0])
    np.testing.assert_array_equal(tracks[1], [1, 0, 1])
    assert adjacency.shape == (5, 5)
    np.testing.assert_array_equal(adjacency_tracks[0], [0, 3])


def test_empty_frames_are_supported():
    tracks, adjacency_tracks, adjacency = simpletracker(
        [np.empty((0, 2)), np.array([[1.0, 2.0]])]
    )

    assert len(tracks) == 1
    np.testing.assert_array_equal(tracks[0], [-1, 0])
    np.testing.assert_array_equal(adjacency_tracks[0], [0])
    assert adjacency.shape == (1, 1)
