"""Tutorial: crossing trajectories with gaps, in 3D.

How does the tracker follow fireflies that cross paths, and what does it do
when the camera misses them? We open the tracker and watch each step.

Run it with:

    uv run --no-project --python .venv/bin/python marimo edit crossing_gaps_tutorial.py
"""

import marimo

__generated_with = "0.17.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from crossing_gaps import (
        PARTICLE_NAMES,
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
        hungarian_linker,
        link_frames,
        simpletracker,
    )

    DEFAULT_LIMIT = 4.0
    TRACK_COLORS = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:olive",
        "tab:cyan",
    ]
    return (
        DEFAULT_LIMIT,
        PARTICLE_NAMES,
        TRACK_COLORS,
        build_tracks,
        close_gaps,
        frame_offsets,
        gap_candidates,
        hungarian_linker,
        identities,
        is_perfect,
        link_frames,
        make_scene,
        mo,
        np,
        plt,
        recovery_rate,
        simpletracker,
    )


@app.cell
def _(PARTICLE_NAMES, TRACK_COLORS, np, plt):
    def axes3d(title, size=(6.5, 5.5)):
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(projection="3d")
        ax.set(
            xlabel="x",
            ylabel="y",
            zlabel="z",
            title=title,
            xlim=(-1, 12),
            ylim=(-1, 8),
            zlim=(-1, 5),
        )
        ax.view_init(elev=22, azim=-62)
        return fig, ax

    def draw_chains(ax, chains, points):
        for number, chain in enumerate(chains):
            p = points[chain]
            ax.plot(
                p[:, 0],
                p[:, 1],
                p[:, 2],
                "o-",
                color=TRACK_COLORS[number % len(TRACK_COLORS)],
                label=f"piece {number}",
            )

    def draw_missing(ax, scene):
        for frame, particle in scene.missing:
            x, y, z = scene.paths[particle, frame]
            ax.scatter([x], [y], [z], marker="x", color="red", s=70, linewidths=2)
        ax.scatter([], [], [], marker="x", color="red", label="missed detection")

    def particle_of(scene, offsets, global_index):
        frame = int(np.searchsorted(offsets[1:], global_index, side="right"))
        local = int(global_index - offsets[frame])
        return PARTICLE_NAMES[scene.truth[frame][local]]

    def draw_heatmap(values, xlabels, ylabels, xlabel, ylabel, title):
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ax.imshow(values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set(
            xticks=range(len(xlabels)),
            yticks=range(len(ylabels)),
            xticklabels=xlabels,
            yticklabels=ylabels,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center")
        return fig

    return axes3d, draw_chains, draw_heatmap, draw_missing, particle_of


@app.cell
def _(mo):
    mo.md("""
    # Fireflies that cross, and blink

    Three fireflies fly in 3D. A camera takes a photo every second, and in
    each photo we get the **positions** of the fireflies, but not their
    **names**. Our tracker has to work out which dot in the next photo is the
    same firefly.

    It is hard for two reasons:

    1. **Crossing.** Two fireflies fly towards each other and pass close by.
       Afterwards, which one is which?
    2. **Gaps.** Sometimes the camera misses a firefly for a few photos. Then
       the track would be cut into pieces, unless we can join them back.

    We will open the tracker and watch it work, step by step:

    | Step | Question | Tool |
    | --- | --- | --- |
    | 1 | Which dot in the next photo matches this dot? | `hungarian_linker` |
    | 2 | What do we get if we do that for all photos? | `link_frames` |
    | 3 | How do we reconnect the broken pieces? | `close_gaps` |
    | 4 | How do pieces become tracks? | `build_tracks` |
    | 5 | When does it go wrong? | experiments |
    """)
    return


@app.cell
def _(mo):
    separation_slider = mo.ui.slider(
        0.0,
        4.0,
        step=0.25,
        value=2.0,
        label="how far apart A and B are when they cross",
        show_value=True,
    )
    noise_slider = mo.ui.slider(
        0.0,
        0.5,
        step=0.05,
        value=0.1,
        label="measurement noise",
        show_value=True,
    )
    missing_slider = mo.ui.slider(
        0,
        4,
        step=1,
        value=2,
        label="photos where B is invisible",
        show_value=True,
    )
    mo.vstack(
        [
            mo.md("## 0. The scene\n\nPlay with the scene. Everything below updates."),
            mo.hstack(
                [separation_slider, noise_slider, missing_slider], justify="start"
            ),
        ]
    )
    return missing_slider, noise_slider, separation_slider


@app.cell
def _(frame_offsets, make_scene, missing_slider, noise_slider, np, separation_slider):
    scene = make_scene(
        separation_slider.value,
        noise_slider.value,
        seed=1,
        b_missing=missing_slider.value,
    )
    offsets = frame_offsets(scene.frames)
    all_points = np.concatenate(scene.frames)
    n_frames = len(scene.frames)
    return all_points, n_frames, offsets, scene


@app.cell
def _(
    PARTICLE_NAMES,
    all_points,
    axes3d,
    draw_missing,
    mo,
    n_frames,
    np,
    offsets,
    scene,
):
    _fig_seen, _ax_seen = axes3d("What the tracker sees: unlabelled dots")
    _frame_of_point = np.repeat(np.arange(n_frames), np.diff(offsets))
    _dots = _ax_seen.scatter(
        all_points[:, 0],
        all_points[:, 1],
        all_points[:, 2],
        c=_frame_of_point,
        cmap="viridis",
        s=36,
    )
    _fig_seen.colorbar(_dots, ax=_ax_seen, shrink=0.55, label="photo number")

    _fig_truth, _ax_truth = axes3d("The hidden truth (we know it, the tracker does not)")
    for _p, _name in enumerate(PARTICLE_NAMES):
        _path = scene.paths[_p]
        _ax_truth.plot(
            _path[:, 0], _path[:, 1], _path[:, 2], "o-", label=f"firefly {_name}"
        )
    draw_missing(_ax_truth, scene)
    _ax_truth.legend(loc="upper left")

    _counts = ", ".join(str(len(frame)) for frame in scene.frames)
    mo.vstack(
        [
            mo.md(
                f"""
                ## 1. The data

                Left: all dots from all {n_frames} photos. Colour shows *when*
                a dot was seen, but nothing says *who* it is. Right: the truth.
                A and B cross near the middle. A red **×** marks a firefly the
                camera missed.

                Dots per photo: `{_counts}`. Three fireflies, but sometimes
                only two dots. Those are the dark moments.
                """
            ),
            mo.hstack([mo.as_html(_fig_seen), mo.as_html(_fig_truth)]),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Step 1: match the dots of two photos

    Take two photos in a row. For every dot in photo *f* we measure its
    distance to **every** dot in photo *f + 1*. That gives a table of
    distances. Small number means "probably the same firefly".

    The **Hungarian algorithm** picks pairs so that:

    * each dot is used **at most once** (one firefly cannot become two), and
    * the **total** distance of all pairs is as small as possible.

    A pair longer than the **distance limit** (here 4 units) is not allowed.
    Grey cells in the table are forbidden. A dot left without a partner means
    a track ends (or begins) there.

    Pick a pair of photos. The default pair is the moment **B vanishes**.
    """)
    return


@app.cell
def _(mo, n_frames):
    pair_slider = mo.ui.slider(
        0, n_frames - 2, step=1, value=3, label="photo f (matched with f + 1)", show_value=True
    )
    pair_slider
    return (pair_slider,)


@app.cell
def _(
    DEFAULT_LIMIT,
    PARTICLE_NAMES,
    axes3d,
    hungarian_linker,
    mo,
    np,
    pair_slider,
    plt,
    scene,
):
    _f = pair_slider.value
    _src, _tgt = scene.frames[_f], scene.frames[_f + 1]
    _dist = np.linalg.norm(_src[:, None, :] - _tgt[None, :, :], axis=2)
    _match, _, _free = hungarian_linker(_src, _tgt, DEFAULT_LIMIT)

    _cmap = plt.get_cmap("Blues_r").copy()
    _cmap.set_bad("lightgray")
    _fig_m, _ax_m = plt.subplots(figsize=(4.8, 3.8))
    _ax_m.imshow(np.where(_dist <= DEFAULT_LIMIT, _dist, np.nan), cmap=_cmap)
    _ax_m.set(
        xticks=range(len(_tgt)),
        yticks=range(len(_src)),
        xlabel=f"dots of photo {_f + 1}",
        ylabel=f"dots of photo {_f}",
        title="Distances (grey = beyond the limit)",
    )
    for _i in range(len(_src)):
        for _j in range(len(_tgt)):
            _ax_m.text(_j, _i, f"{_dist[_i, _j]:.1f}", ha="center", va="center")
            if _match[_i] == _j:
                _ax_m.add_patch(
                    plt.Rectangle(
                        (_j - 0.5, _i - 0.5), 1, 1, fill=False, edgecolor="red", lw=3
                    )
                )

    _fig_p, _ax_p = axes3d(f"Photos {_f} (circles) and {_f + 1} (squares)", size=(5.5, 4.6))
    _ax_p.scatter(_src[:, 0], _src[:, 1], _src[:, 2], marker="o", s=60, color="tab:blue")
    _ax_p.scatter(_tgt[:, 0], _tgt[:, 1], _tgt[:, 2], marker="s", s=60, color="tab:orange")
    for _i, _j in enumerate(_match):
        if _j >= 0:
            _a, _b = _src[_i], _tgt[_j]
            _ax_p.plot(*zip(_a, _b), color="red", lw=2)

    _lines = []
    for _i, _j in enumerate(_match):
        if _j >= 0:
            _same = scene.truth[_f][_i] == scene.truth[_f + 1][_j]
            _verdict = (
                f"✔ same firefly ({PARTICLE_NAMES[scene.truth[_f][_i]]})"
                if _same
                else "✘ **different fireflies!**"
            )
            _lines.append(
                f"* dot {_i} of photo {_f} → dot {_j} of photo {_f + 1}, "
                f"distance {_dist[_i, _j]:.2f}: {_verdict}"
            )
        else:
            _lines.append(
                f"* dot {_i} of photo {_f} → **no partner**: a track ends here"
            )
    for _j in _free:
        _lines.append(
            f"* dot {_j} of photo {_f + 1} has **no partner**: a track starts here"
        )

    mo.vstack(
        [
            mo.hstack([mo.as_html(_fig_m), mo.as_html(_fig_p)]),
            mo.md("\n".join(_lines)),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Step 2: do it for every pair of photos

    `link_frames` repeats step 1 for photo 0→1, 1→2, 2→3, and so on. Each link
    joins a dot to the dot that follows it. Following the links from the
    start gives **pieces** of track.

    * 🔻 **red triangle**: a piece **ends** here (no partner in the next photo)
    * 🔺 **green triangle**: a piece **starts** here (no partner in the previous one)

    There are only three fireflies, so if you see more than three pieces, some
    pieces belong together.
    """)
    return


@app.cell
def _(DEFAULT_LIMIT, build_tracks, link_frames, scene):
    links1, unmatched_sources1, unmatched_targets1 = link_frames(
        scene.frames, max_linking_distance=DEFAULT_LIMIT
    )
    pieces1, piece_chains1 = build_tracks(links1, scene.frames)
    return piece_chains1, pieces1, unmatched_sources1, unmatched_targets1


@app.cell
def _(
    all_points,
    axes3d,
    draw_chains,
    identities,
    mo,
    n_frames,
    offsets,
    piece_chains1,
    pieces1,
    scene,
    unmatched_sources1,
    unmatched_targets1,
):
    _fig, _ax = axes3d("Pieces after frame-to-frame linking")
    draw_chains(_ax, piece_chains1, all_points)
    _ends = [
        all_points[offsets[_f] + _i]
        for _f in range(n_frames - 1)
        for _i in unmatched_sources1[_f]
    ]
    _starts = [
        all_points[offsets[_g] + _j]
        for _g in range(1, n_frames)
        for _j in unmatched_targets1[_g]
    ]
    for _pts, _marker, _color, _label in (
        (_ends, "v", "red", "end of a piece"),
        (_starts, "^", "limegreen", "start of a piece"),
    ):
        if _pts:
            _arr = [list(_p) for _p in _pts]
            _ax.scatter(
                [_p[0] for _p in _arr],
                [_p[1] for _p in _arr],
                [_p[2] for _p in _arr],
                marker=_marker,
                color=_color,
                s=140,
                edgecolor="black",
                label=_label,
            )
    _ax.legend(loc="upper left", fontsize=8)

    _rows = "\n".join(identities(pieces1, scene.truth))
    mo.vstack(
        [
            mo.md(
                f"""
                We get **{len(pieces1)} pieces** from **3 fireflies**. Here is the
                *true* firefly behind each dot of each piece (`-` = no dot in that
                photo):

                ```text
                {_rows}
                ```

                Every piece follows one firefly only, so frame-to-frame matching
                did its job. But a firefly that blinks is split in two pieces.
                """
            ),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Step 3: close the gaps

    Now `close_gaps` looks at the **loose ends**. For each piece that ended
    early, it looks at the pieces that **start later**, and asks which is the
    closest. It skips at least one photo, so this can jump over a dark moment.

    Two settings control it:

    * **`max_gap_closing`**: how many photos ahead we are allowed to look.
      If a firefly is invisible for *k* photos, its last and next dots are
      *k + 1* photos apart, so we need `max_gap_closing ≥ k + 1`.
    * **distance limit** (`max_linking_distance`): how far a jump may be. The
      limit applies to the **whole jump**, so a longer dark moment means a
      longer distance to cover.

    Change them and watch the table, the picture and the verdict.
    """)
    return


@app.cell
def _(mo):
    reach_slider = mo.ui.slider(
        1, 6, step=1, value=3, label="max_gap_closing", show_value=True
    )
    limit_slider = mo.ui.slider(
        1.0, 8.0, step=0.5, value=4.0, label="distance limit", show_value=True
    )
    mo.hstack([reach_slider, limit_slider], justify="start")
    return limit_slider, reach_slider


@app.cell
def _(
    build_tracks,
    close_gaps,
    gap_candidates,
    limit_slider,
    link_frames,
    offsets,
    reach_slider,
    scene,
):
    links3, unmatched_sources3, unmatched_targets3 = link_frames(
        scene.frames, max_linking_distance=limit_slider.value
    )
    candidates3 = gap_candidates(
        scene.frames,
        offsets,
        unmatched_sources3,
        unmatched_targets3,
        reach_slider.value,
    )
    gap_links3 = close_gaps(
        scene.frames,
        unmatched_sources3,
        unmatched_targets3,
        max_linking_distance=limit_slider.value,
        max_gap_closing=reach_slider.value,
    )
    pieces3, piece_chains3 = build_tracks(links3, scene.frames)
    return candidates3, gap_links3, links3, piece_chains3, pieces3


@app.cell
def _(
    all_points,
    axes3d,
    candidates3,
    draw_chains,
    gap_links3,
    limit_slider,
    mo,
    missing_slider,
    np,
    offsets,
    particle_of,
    piece_chains3,
    pieces3,
    reach_slider,
    scene,
):
    _fig, _ax = axes3d("Pieces (solid) and gap links (dashed black)")
    draw_chains(_ax, piece_chains3, all_points)
    for _s, _t in gap_links3.items():
        _a, _b = all_points[_s], all_points[_t]
        _ax.plot(*zip(_a, _b), "k--", lw=3)
    _ax.legend(loc="upper left", fontsize=8)

    def _name(global_index, frame):
        return (
            f"dot {int(global_index - offsets[frame])} of photo {frame}"
        )

    _rows = []
    for _s, _t, _sf, _tf, _d in candidates3:
        _allowed = _d <= limit_slider.value
        _chosen = gap_links3.get(_s) == _t
        _same = particle_of(scene, offsets, _s) == particle_of(scene, offsets, _t)
        _rows.append(
            f"| {_name(_s, _sf)} → {_name(_t, _tf)} | {_tf - _sf} | {_d:.2f} "
            f"| {'yes' if _allowed else 'no'} | {'✔' if _same else '✘'} "
            f"| {'**linked**' if _chosen else ''} |"
        )
    _table = (
        "| loose end → later start | photos apart | distance | within limit? "
        "| same firefly? | result |\n| --- | --- | --- | --- | --- | --- |\n"
        + "\n".join(_rows)
        if _rows
        else "_There is no loose end and later start to compare. "
        "Try a larger `max_gap_closing`._"
    )

    _k = missing_slider.value
    _step = float(np.linalg.norm(scene.paths[1, 1] - scene.paths[1, 0]))
    _need = (
        f"B is invisible for **{_k}** photo(s). Its last dot and its next dot are "
        f"**{_k + 1}** photos apart, about **{_step * (_k + 1):.1f}** units "
        f"(B moves about {_step:.2f} per photo). So we need "
        f"`max_gap_closing ≥ {_k + 1}` and a distance limit above "
        f"**{_step * (_k + 1):.1f}**."
        if _k > 0
        else "B is never invisible here, only C blinks (1 photo)."
    )
    mo.vstack(
        [
            mo.md(
                f"""
                {_need}

                Pieces before closing gaps: **{len(pieces3)}**. Gap links
                created: **{len(gap_links3)}**.

                Candidates are sorted by distance. The closest allowed pair
                wins, and each dot can be used only once:

                {_table}
                """
            ),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Step 4: pieces become tracks

    `build_tracks` walks along the links. A track **starts** at every dot that
    nothing links to, and follows links until they run out. The result is one
    list per track with, for each photo, the index of the dot (or `-1` if
    the track has no dot there, which is a gap).

    `simpletracker` is exactly these three steps in a row. Below we run it, and
    we also compare it with doing the steps one by one.
    """)
    return


@app.cell
def _(
    all_points,
    axes3d,
    build_tracks,
    draw_missing,
    gap_links3,
    identities,
    is_perfect,
    limit_slider,
    links3,
    mo,
    np,
    reach_slider,
    scene,
    simpletracker,
    TRACK_COLORS,
):
    _links = {**links3, **gap_links3}
    _steps_tracks, _ = build_tracks(_links, scene.frames)
    _tracks, _chains, _ = simpletracker(
        scene.frames,
        max_linking_distance=limit_slider.value,
        max_gap_closing=reach_slider.value,
    )
    _same_as_steps = len(_tracks) == len(_steps_tracks) and all(
        np.array_equal(_a, _b) for _a, _b in zip(_tracks, _steps_tracks)
    )

    _fig, _ax = axes3d("Final tracks")
    for _n, _chain in enumerate(_chains):
        _p = all_points[_chain]
        _ax.plot(
            _p[:, 0],
            _p[:, 1],
            _p[:, 2],
            "o-",
            color=TRACK_COLORS[_n % len(TRACK_COLORS)],
            label=f"track {_n}",
        )
    draw_missing(_ax, scene)
    _ax.legend(loc="upper left", fontsize=8)

    _mixed = any(
        len(set(_row.replace("-", ""))) > 1 for _row in identities(_tracks, scene.truth)
    )
    if is_perfect(_tracks, scene.truth):
        _verdict = (
            "✅ **Perfect.** One track per firefly, and no track mixes two "
            "fireflies."
        )
    elif _mixed:
        _verdict = (
            "❌ **A track mixes two fireflies.** The tracker joined dots of "
            "different fireflies. Either the distance limit is too generous, "
            "or the fireflies passed too close to each other."
        )
    else:
        _verdict = (
            f"⚠️ **Too many tracks ({len(_tracks)} instead of 3).** Some pieces were "
            "not joined: `max_gap_closing` or the distance limit is too small "
            "for the gap."
        )

    _arrays = "\n".join(f"track {_n}: {_t.tolist()}" for _n, _t in enumerate(_tracks))
    _who = "\n".join(
        f"track {_n}: {_s}" for _n, _s in enumerate(identities(_tracks, scene.truth))
    )
    mo.vstack(
        [
            mo.md(
                f"""
                {_verdict}

                `simpletracker` gives the same tracks as the three steps
                separately: **{_same_as_steps}**.

                What the tracker returns (index of the dot in each photo,
                `-1` = gap):

                ```text
                {_arrays}
                ```

                What it really followed (we know the truth):

                ```text
                {_who}
                ```
                """
            ),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. When does it go wrong?

    Single examples can be lucky, so we run each experiment on 30 random
    versions of the scene (different noise) and record how often the tracker
    gets **perfect** tracks. Green is good, red is bad.

    ### Experiment A: how long a gap can we bridge?

    Rows: how many photos B is invisible. Columns: `max_gap_closing`. The
    distance limit is 5, large enough for all these gaps.
    """)
    return


@app.cell
def _(draw_heatmap, mo, np, recovery_rate):
    _missing_values = [0, 1, 2, 3]
    _reach_values = [1, 2, 3, 4, 5]
    _grid_a = np.array(
        [
            [
                recovery_rate(
                    b_missing=_k,
                    max_gap_closing=_r,
                    max_linking_distance=5.0,
                    n_trials=30,
                )
                for _r in _reach_values
            ]
            for _k in _missing_values
        ]
    )
    _fig = draw_heatmap(
        _grid_a,
        _reach_values,
        _missing_values,
        "max_gap_closing",
        "photos B is invisible",
        "Fraction of perfect results",
    )
    mo.vstack(
        [
            mo.as_html(_fig),
            mo.md(
                """
                **Reading it:** the green region is a staircase. A gap of *k*
                photos needs `max_gap_closing ≥ k + 1`, because the tracker
                looks from the last photo where the firefly was seen. Also
                note that C blinks for one photo in every row, so even the
                top row needs `max_gap_closing ≥ 2`.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Experiment B: how big may the distance limit be?

    Now `max_gap_closing` is large (6) and we change the **distance limit**.
    """)
    return


@app.cell
def _(draw_heatmap, mo, np, recovery_rate):
    _missing_values = [0, 1, 2, 3, 4]
    _limit_values = [1, 2, 3, 4, 5, 6, 7, 8]
    _grid_b = np.array(
        [
            [
                recovery_rate(
                    b_missing=_k,
                    max_gap_closing=6,
                    max_linking_distance=float(_l),
                    n_trials=30,
                )
                for _l in _limit_values
            ]
            for _k in _missing_values
        ]
    )
    _fig = draw_heatmap(
        _grid_b,
        _limit_values,
        _missing_values,
        "distance limit",
        "photos B is invisible",
        "Fraction of perfect results",
    )
    mo.vstack(
        [
            mo.as_html(_fig),
            mo.md(
                """
                **Reading it:** there is a *window* of good limits, and it
                gets narrower for longer gaps.

                * **Too small:** the jump over the gap is longer than the
                  limit, so the pieces are never joined. A longer gap needs a
                  larger limit.
                * **Too big:** the tracker joins things that should not be
                  joined. Why? When B vanishes, B's last dot and C's first dot
                  are *both left over*. They are about 6 units apart. With a
                  big limit the tracker pairs the two leftovers, even though
                  they are two different fireflies. When B is visible
                  (top row), it is matched with its own dot, so nothing is left
                  over to pair wrongly, and big limits are harmless.

                A good limit is a bit more than the longest jump you want to
                bridge, and clearly less than the distance between different
                fireflies.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Experiment C: how close can two paths cross?

    Now the crossing itself. We slide the **separation** of A and B at the
    crossing, with B always visible (solid) or invisible for 2 photos
    (dashed), at two noise levels.
    """)
    return


@app.cell
def _(mo, plt, recovery_rate):
    _separations = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    _fig, _ax = plt.subplots(figsize=(6.2, 3.8))
    for _noise, _color in ((0.1, "tab:blue"), (0.3, "tab:orange")):
        for _k, _style in ((0, "o-"), (2, "s--")):
            _ax.plot(
                _separations,
                [
                    recovery_rate(
                        separation=_s,
                        noise=_noise,
                        b_missing=_k,
                        max_gap_closing=3,
                        max_linking_distance=4.0,
                        n_trials=30,
                    )
                    for _s in _separations
                ],
                _style,
                color=_color,
                label=f"noise {_noise}, B invisible {_k} photos",
            )
    _ax.axvline(1.04, color="gray", ls=":")
    _ax.text(1.08, 0.05, "one step of movement", color="gray", fontsize=8)
    _ax.set(
        xlabel="separation of A and B at the crossing",
        ylabel="fraction of perfect results",
        title="Crossing too close",
    )
    _ax.legend(fontsize=7)
    mo.vstack(
        [
            mo.as_html(_fig),
            mo.md(
                """
                **Reading it:** below about one step of movement (the dotted
                line, about 1 unit) the tracker fails, with or without a gap.
                When A and B pass closer than they move in one photo, the
                *wrong* neighbour is as close as the *right* one. Distance is
                the only clue this tracker has, so it cannot tell them apart.
                Above about 1.5 units, it is almost always right.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Recap

    **The recipe**

    1. **Link photo to photo** (`link_frames`): the Hungarian algorithm pairs
       each dot with at most one dot of the next photo, minimising the total
       distance, and never beyond the distance limit.
    2. **Close gaps** (`close_gaps`): every loose end looks for the nearest
       later start, up to `max_gap_closing` photos ahead and within the limit.
    3. **Build tracks** (`build_tracks`): follow the links; a gap is shown as `-1`.

    **The three things to remember**

    * **Reach:** a gap of *k* missing photos needs `max_gap_closing ≥ k + 1`.
    * **Limit:** it must cover the longest jump you want to bridge, but stay
      well below the distance between *different* fireflies.
    * **Crossings:** with only positions to go on, fireflies that pass closer
      than one step of movement can be swapped. Predicting where a firefly
      *should* go, using its speed, would fix that, but needs a smarter tracker.
    """)
    return


if __name__ == "__main__":
    app.run()
