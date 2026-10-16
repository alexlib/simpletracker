"""A beginner-friendly, step-by-step Marimo notebook for SimpleTracker.

Run it with:

    uv run marimo edit simpletracker_tutorial.py
"""

import marimo

__generated_with = "0.25.1"
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

    return hungarian_linker, mo, np, plt, simpletracker


@app.cell
def _(mo):
    mo.md("""
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
    """)
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
    mo.md(f"""
    ## 1. Frames are lists of points

    A frame is a table: one row per detected particle, with columns
    `(x, y)`. This example has `{len(frames)}` frames.

    The order of rows can change from one frame to the next. That is why
    we need a tracking algorithm instead of simply joining row 1 to row 1.
    """)
    return


@app.cell
def _(frames, plt):
    fig_points, ax_points = plt.subplots()
    for frame_number, frame in enumerate(frames):
        ax_points.scatter(
            frame[:, 0], frame[:, 1], label=f"frame {frame_number}"
        )
        for point_number, (x, y) in enumerate(frame):
            ax_points.text(x + 0.08, y + 0.08, f"{frame_number}:{point_number}")
    ax_points.set(xlabel="x", ylabel="y", title="Detections, before tracking")
    ax_points.legend()
    return


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
    mo.md(f"""
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
    """)
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
    return


@app.cell
def _(np, simpletracker):
    comparison_points = [
        np.array([[0.0, 0.0], [10.0, 0.0]]),
        np.array([[10.5, 0.0]]),
        np.array([[2.0, 0.0], [11.0, 0.0]]),
    ]
    comparison_tracks, comparison_adjacency_tracks, comparison_adjacency = (
        simpletracker(
            comparison_points,
            method="nearestneighbor",
            max_linking_distance=2,
            max_gap_closing=2,
        )
    )
    return comparison_adjacency, comparison_adjacency_tracks, comparison_tracks


@app.cell
def _(
    comparison_adjacency,
    comparison_adjacency_tracks,
    comparison_tracks,
    mo,
    np,
):
    # Expected output captured by running src/simpletracker.m in GNU Octave
    # with the same points and options, using MATLAB's one-based indices.
    octave_tracks = [
        np.array([1.0, np.nan, 1.0]),
        np.array([2.0, 1.0, 2.0]),
    ]
    octave_adjacency_tracks = [
        np.array([1, 4]),
        np.array([2, 3, 5]),
    ]
    octave_links = {(1, 4), (2, 3), (3, 5)}

    python_tracks_as_octave = [
        np.where(track < 0, np.nan, track + 1).astype(float)
        for track in comparison_tracks
    ]
    python_adjacency_tracks_as_octave = [
        chain + 1 for chain in comparison_adjacency_tracks
    ]
    python_links_as_octave = {
        (int(source + 1), int(target + 1))
        for source, target in zip(*np.nonzero(comparison_adjacency))
    }

    tracks_match = len(python_tracks_as_octave) == len(octave_tracks) and all(
        np.array_equal(python, octave, equal_nan=True)
        for python, octave in zip(python_tracks_as_octave, octave_tracks)
    )
    adjacency_tracks_match = (
        len(python_adjacency_tracks_as_octave) == len(octave_adjacency_tracks)
        and all(
            np.array_equal(python, octave)
            for python, octave in zip(
                python_adjacency_tracks_as_octave, octave_adjacency_tracks
            )
        )
    )
    links_match = python_links_as_octave == octave_links

    mo.md(
        f"""
        ## 5. Comparing with Octave

        This is the **same data and settings** used in GNU Octave 11.3:
        `NearestNeighbor`, maximum link distance `2`, and maximum gap
        closing `2`. Octave numbers points starting at 1; Python starts at 0,
        so the Python results below are converted to Octave numbering to make
        a fair comparison.

        | Result | Octave | Python (converted) | Same? |
        | --- | --- | --- | --- |
        | Tracks | `{[track.tolist() for track in octave_tracks]}` | `{[track.tolist() for track in python_tracks_as_octave]}` | **{tracks_match}** |
        | Adjacency tracks | `{[chain.tolist() for chain in octave_adjacency_tracks]}` | `{[chain.tolist() for chain in python_adjacency_tracks_as_octave]}` | **{adjacency_tracks_match}** |
        | Forward links | `{sorted(octave_links)}` | `{sorted(python_links_as_octave)}` | **{links_match}** |

        **All results match: {tracks_match and adjacency_tracks_match and links_match}.**
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Study: crossing trajectories in 3D

    Real particles do not move in a flat world, and their paths can cross.
    The tracker works in any number of dimensions: the only thing it
    needs is a distance between points, so 3D costs nothing extra.

    Here two particles, **A** and **B**, fly in straight lines and cross
    in the middle of the movie (frame 5). The tracker never sees their
    names or their speed, only unlabelled `(x, y, z)` points. We know the
    truth, so we can check whether it kept the identities right.

    The key control is the **separation**: how far apart in `z` they are
    when they cross. Try `0` (they touch) and then slide it up.
    """)
    return


@app.cell
def _(mo):
    separation_slider = mo.ui.slider(
        0.0, 4.0, step=0.25, value=0.0, label="separation at crossing", show_value=True
    )
    noise_slider = mo.ui.slider(
        0.0, 0.5, step=0.05, value=0.1, label="detection noise", show_value=True
    )
    method_dropdown = mo.ui.dropdown(
        ["hungarian", "nearestneighbor"], value="hungarian", label="method"
    )
    mo.hstack([separation_slider, noise_slider, method_dropdown])
    return method_dropdown, noise_slider, separation_slider


@app.cell
def _(method_dropdown, mo, noise_slider, separation_slider, simpletracker):
    from crossing_demo import (
        has_identity_swap,
        make_crossing_frames,
        swap_rate,
        track_identities,
    )

    crossing_frames, crossing_truth = make_crossing_frames(
        separation_slider.value, noise_slider.value, seed=1
    )
    crossing_tracks, crossing_chains, _ = simpletracker(
        crossing_frames,
        method=method_dropdown.value,
        max_linking_distance=3.0,
    )
    crossing_swapped = has_identity_swap(crossing_tracks, crossing_truth)
    crossing_verdict = (
        "**Identity swap!** A track jumped from one particle to the other "
        "at the crossing."
        if crossing_swapped
        else "**No swap.** Each track follows a single particle."
    )
    crossing_lines = "\n".join(track_identities(crossing_tracks, crossing_truth))
    mo.md(f"""
    The tracker found **{len(crossing_tracks)} tracks**. Reading each track
    as the *true* particle behind every point (A or B, one letter per frame):

    ```text
    {crossing_lines}
    ```

    {crossing_verdict}
    """)
    return (
        crossing_chains,
        crossing_frames,
        make_crossing_frames,
        swap_rate,
        track_identities,
    )


@app.cell
def _(crossing_chains, crossing_frames, np, plt):
    cross_points = np.concatenate(crossing_frames)
    fig_cross = plt.figure(figsize=(7, 6))
    ax_cross = fig_cross.add_subplot(projection="3d")
    cross_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for cross_number, cross_chain in enumerate(crossing_chains):
        cross_selected = cross_points[cross_chain]
        ax_cross.plot(
            cross_selected[:, 0],
            cross_selected[:, 1],
            cross_selected[:, 2],
            "o-",
            color=cross_colors[cross_number % len(cross_colors)],
            label=f"track {cross_number}",
        )
    ax_cross.set(
        xlabel="x", ylabel="y", zlabel="z", title="Tracks found by the tracker"
    )
    ax_cross.legend()
    ax_cross
    return


@app.cell
def _(mo):
    mo.md("""
    ### How often does it get confused?

    One run can be lucky. So we repeat the experiment with 100 different
    random noise patterns for each separation and count the fraction of
    runs where the tracks got mixed up. Distance is the only clue the
    tracker has: when two particles pass closer than about one step of
    their movement, the "wrong" neighbour looks as near as the "right"
    one, and nothing can tell them apart.
    """)
    return


@app.cell
def _(plt, swap_rate):
    sweep_separations = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    fig_sweep, ax_sweep = plt.subplots()
    for sweep_noise in (0.1, 0.3):
        for sweep_method in ("hungarian", "nearestneighbor"):
            ax_sweep.plot(
                sweep_separations,
                [
                    swap_rate(separation, sweep_noise, sweep_method, n_trials=100)
                    for separation in sweep_separations
                ],
                "o-" if sweep_method == "hungarian" else "s--",
                label=f"{sweep_method}, noise {sweep_noise}",
            )
    ax_sweep.set(
        xlabel="separation at crossing",
        ylabel="fraction of runs with a swap",
        title="Identity swaps vs. how close the particles pass",
    )
    ax_sweep.legend()
    ax_sweep
    return


@app.cell
def _(mo):
    mo.md("""
    What to notice:

    * **Far apart: no problem.** From about 2 units of separation, tracks
      almost never mix up.
    * **Touching: a swap is almost certain.** At separation 0 the tracker
      has no information to choose, so it mixes the particles in nearly
      every run.
    * **Noise makes it worse.** Shaky measurements widen the danger zone
      around the crossing.
    * **Hungarian vs nearest neighbor:** with only two particles they
      behave almost the same. The Hungarian method pays off when many
      particles compete for the same neighbours.
    * **The fix is more information, not more tracking.** This tracker
      only uses position. Predicting where a particle *should* be from its
      speed would solve crossings, but needs a more advanced tracker.
    """)
    return


@app.cell
def _(make_crossing_frames, mo, simpletracker, track_identities):
    # B is not detected in frame 5, exactly when the paths cross.
    gap_frames, gap_truth = make_crossing_frames(separation=3.0, drop=(5, 1))
    gap_open, _, _ = simpletracker(
        gap_frames, max_linking_distance=2.5, max_gap_closing=1
    )
    gap_closed, _, _ = simpletracker(
        gap_frames, max_linking_distance=2.5, max_gap_closing=2
    )
    gap_open_lines = "\n".join(track_identities(gap_open, gap_truth))
    gap_closed_lines = "\n".join(track_identities(gap_closed, gap_truth))
    mo.md(f"""
    ### A missing detection right at the crossing

    Gaps and crossings can happen together: particle **B** is not detected
    in frame 5, exactly when the paths cross.

    With `max_gap_closing=1` (no gap closing) B's track is cut in two, so
    we get **{len(gap_open)} tracks**:

    ```text
    {gap_open_lines}
    ```

    With `max_gap_closing=2` the tracker bridges the hole and we get
    **{len(gap_closed)} tracks**. The `-` marks the missing frame:

    ```text
    {gap_closed_lines}
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Key ideas to remember

    * **Frame:** one snapshot containing detected points.
    * **Link:** a decision that two points belong to the same moving object.
    * **Assignment:** links are exclusive; one point cannot have two next
      points.
    * **Gap closing:** a short disappearance does not have to end a track.
    * **Track:** the sequence of observations belonging to one object.
    * **Crossing:** when two paths pass closer than one step of movement,
      distance alone cannot tell the particles apart and identities can swap.

    Try changing the distances and deleting points above. That is the best
    way to see when a tracker is confident and when two particles are too
    close to tell apart.
    """)
    return


if __name__ == "__main__":
    app.run()
