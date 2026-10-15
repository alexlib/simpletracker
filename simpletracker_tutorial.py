"""A beginner-friendly, step-by-step Marimo notebook for SimpleTracker.

Run it with:

    marimo edit simpletracker_tutorial.py
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from simpletracker import (
        hungarian_linker,
        nearest_neighbor_linker,
        simpletracker,
    )

    return (
        hungarian_linker,
        nearest_neighbor_linker,
        np,
        mo,
        plt,
        simpletracker,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # How particle tracking works

        Imagine taking a photo of moving fireflies every second. Each photo
        gives us positions, but not names. The tracker answers: **which point
        in the next frame is the same firefly?**

        We will learn this in small steps:
        1. represent points in frames,
        2. measure distances,
        3. match points,
        4. handle a missing observation (a gap),
        5. rebuild complete tracks.
        """
    )

    return


@app.cell
def _(np):
    frames = [
        np.array([[1.0, 1.0], [8.0, 2.0], [4.0, 8.0]]),
        np.array([[2.0, 1.5], [7.0, 2.5], [4.2, 7.0]]),
        np.array([[3.0, 2.0], [6.0, 3.0], [4.4, 6.0]]),
    ]
    return (frames,)


@app.cell
def _(frames, mo):
    mo.md(
        f"""
        ## 1. Frames are lists of points

        A frame is a table: one row per detected particle, with columns
        `(x, y)`. This example has `{len(frames)}` frames.

        The order of rows can change from one frame to the next. That is why
        we need a tracking algorithm instead of simply joining row 1 to row 1.
        """
    )
    return


@app.cell
def _(frames, np, plt):
    fig_points, ax_points = plt.subplots()
    for frame_number, frame in enumerate(frames):
        ax_points.scatter(
            frame[:, 0], frame[:, 1], label=f"frame {frame_number}"
        )
        for point_number, (x, y) in enumerate(frame):
            ax_points.text(x + 0.08, y + 0.08, f"{frame_number}:{point_number}")
    ax_points.set(xlabel="x", ylabel="y", title="Detections, before tracking")
    ax_points.legend()
    return (fig_points,)


@app.cell
def _(frames, mo, np):
    source, target = frames[0], frames[1]
    distance_table = np.linalg.norm(source[:, None, :] - target[None, :, :], axis=2)
    mo.md(
        f"""
        ## 2. Distance is our clue

        For every point in one frame, we calculate its distance to every point
        in the next frame. The resulting distance table has shape
        `{distance_table.shape}`. Small numbers mean "these points are close".
        """
    )
    return (distance_table,)


@app.cell
def _(distance_table, mo):
    mo.md("```text\n" + str(distance_table.round(2)) + "\n```")
    return


@app.cell
def _(frames, hungarian_linker, mo):
    matches, distances, unmatched = hungarian_linker(frames[0], frames[1])
    mo.md(
        f"""
        ## 3. Matching points

        The **Hungarian algorithm** finds an overall best assignment. It never
        uses one point twice. Here, source points match target points at
        indices `{matches.tolist()}`, with distances `{distances.round(2).tolist()}`.

        An alternative is `nearest_neighbor_linker`, which greedily chooses
        the closest remaining pair. It is simpler and faster, but not always
        globally best.
        """
    )
    return


@app.cell
def _(np, simpletracker):
    frames_with_gap = [
        np.array([[0.0, 0.0], [10.0, 0.0]]),
        np.array([[10.5, 0.0]]),  # the first particle was not detected
        np.array([[2.0, 0.0], [11.0, 0.0]]),
    ]
    tracks, adjacency_tracks, adjacency = simpletracker(
        frames_with_gap,
        max_linking_distance=2,
        max_gap_closing=2,
    )
    return adjacency, adjacency_tracks, frames_with_gap, tracks


@app.cell
def _(adjacency, adjacency_tracks, mo, tracks):
    mo.md(
        f"""
        ## 4. Closing a gap

        The first particle disappears in frame 1 and returns in frame 2. The
        tracker first makes ordinary neighboring-frame links. Then it looks
        for an unmatched track ending near an unmatched point in a later
        frame. This creates a gap-closing link.

        Track arrays use `-1` for a missing observation:

        ```text
        {[track.tolist() for track in tracks]}
        ```

        The corresponding global point chains are:
        `{[chain.tolist() for chain in adjacency_tracks]}`.
        The adjacency matrix has shape `{adjacency.shape}` and stores the
        forward links.
        """
    )
    return


@app.cell
def _(adjacency_tracks, frames_with_gap, np, plt):
    all_points = np.concatenate(frames_with_gap)
    fig_tracks, ax_tracks = plt.subplots()
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for track_number, chain in enumerate(adjacency_tracks):
        selected = all_points[chain]
        ax_tracks.plot(
            selected[:, 0],
            selected[:, 1],
            "o-",
            color=colors[track_number % len(colors)],
            label=f"track {track_number}",
        )
    ax_tracks.set(xlabel="x", ylabel="y", title="The reconstructed trajectories")
    ax_tracks.legend()
    return (fig_tracks,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Key ideas to remember

        * **Frame:** one snapshot containing detected points.
        * **Link:** a decision that two points belong to the same moving object.
        * **Assignment:** links are exclusive; one point cannot have two next
          points.
        * **Gap closing:** a short disappearance does not have to end a track.
        * **Track:** the sequence of observations belonging to one object.

        Try changing the distances and deleting points above. That is the best
        way to see when a tracker is confident and when two particles are too
        close to tell apart.
        """
    )
    return


if __name__ == "__main__":
    app.run()
