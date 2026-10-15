import numpy as np
import pytest

from realistic_scene import make_realistic_scene, score_tracks


def truth_tracks(scene):
    """The perfect tracks: one per particle, from the hidden identities."""
    tracks = []
    for particle in range(scene.n_particles):
        track = np.full(len(scene.frames), -1)
        for frame, ids in enumerate(scene.truth):
            hits = np.flatnonzero(ids == particle)
            if len(hits):
                track[frame] = hits[0]
        tracks.append(track)
    return tracks


def test_scene_is_reproducible_and_shaped():
    a = make_realistic_scene(seed=4)
    b = make_realistic_scene(seed=4)
    assert a.paths.shape == (10, 25, 3)
    assert len(a.frames) == len(a.truth) == 25
    for left, right in zip(a.frames, b.frames):
        assert np.array_equal(left, right)


def test_truth_labels_clutter_and_missed_detections():
    scene = make_realistic_scene(detection_prob=0.5, clutter=4.0, seed=1)
    ids = np.concatenate(scene.truth)
    assert (ids == -1).any()
    assert (ids >= 0).sum() < scene.n_particles * 25
    for frame, labels in zip(scene.frames, scene.truth):
        assert len(frame) == len(labels)


def test_crossing_pairs_really_meet():
    scene = make_realistic_scene(
        n_particles=4, crossing_pairs=2, miss_distance=0.4, wiggle=0.0, seed=2
    )
    middle = 25 // 2
    for first in (0, 2):
        gap = np.linalg.norm(scene.paths[first, middle] - scene.paths[first + 1, middle])
        assert gap == pytest.approx(0.4)


def test_perfect_tracks_score_perfectly():
    scene = make_realistic_scene(seed=3)
    scores = score_tracks(truth_tracks(scene), scene.truth)
    assert scores["precision"] == scores["recall"] == scores["f1"] == 1.0
    assert scores["switches"] == 0 and scores["clutter_links"] == 0


def test_swapped_tracks_are_counted_as_switches():
    scene = make_realistic_scene(clutter=0.0, detection_prob=1.0, seed=3)
    tracks = truth_tracks(scene)
    first, second = tracks[0].copy(), tracks[1].copy()
    first[10:], second[10:] = tracks[1][10:], tracks[0][10:]
    scores = score_tracks([first, second, *tracks[2:]], scene.truth)
    assert scores["switches"] == 2
    assert scores["precision"] < 1.0 and scores["recall"] < 1.0


def test_linking_clutter_is_counted():
    scene = make_realistic_scene(clutter=3.0, seed=5)
    tracks = truth_tracks(scene)
    frame = next(f for f, ids in enumerate(scene.truth) if (ids < 0).any() and f > 0)
    clutter_index = int(np.flatnonzero(scene.truth[frame] < 0)[0])
    tracks[0] = tracks[0].copy()
    tracks[0][frame] = clutter_index
    assert score_tracks(tracks, scene.truth)["clutter_links"] >= 1
