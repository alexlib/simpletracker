"""Tutorial: tracking when the world is noisy and paths cross.

The earlier demos had perfect dots and gentle crossings. Here the camera is
noisy, misses dots, invents false dots, and particles pass very close to each
other. We add motion prediction (a Kalman filter) and whole-track crossing
repair, and measure honestly whether they help.

Run it with:

    uv run marimo edit robust_tracking_tutorial.py
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from crossing_resolver import resolve_crossings
    from kalman_tracker import (
        kalman_tracker,
        make_motion_model,
        mahalanobis_costs,
        predict,
        update,
    )
    from realistic_scene import make_realistic_scene, score_tracks
    from simpletracker import simpletracker

    return (
        kalman_tracker,
        mahalanobis_costs,
        make_motion_model,
        make_realistic_scene,
        mo,
        np,
        plt,
        predict,
        resolve_crossings,
        score_tracks,
        simpletracker,
        update,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Tracking in a messy world

    Our first tracker said: *"link each dot to the nearest dot in the next
    frame."* That works when dots are clean and far apart. Real cameras are
    not like that:

    - **noise**: every position is slightly wrong,
    - **missed dots**: sometimes a particle is not detected,
    - **false dots** (clutter): the camera sees things that are not particles,
    - **near misses**: two particles pass so close that nearest-neighbour
      cannot tell them apart.

    We will build the messy world, watch the simple tracker struggle, then
    add two ideas: **predict where a particle should be**, and **check whole
    tracks for identity swaps**.
    """)
    return


@app.cell
def _(mo):
    noise = mo.ui.slider(0.0, 0.8, step=0.05, value=0.2, label="noise (detector error)")
    detection = mo.ui.slider(0.5, 1.0, step=0.05, value=0.9, label="chance a dot is seen")
    clutter = mo.ui.slider(0.0, 6.0, step=0.5, value=1.0, label="false dots per frame")
    miss = mo.ui.slider(0.1, 2.0, step=0.1, value=0.3, label="closest approach at crossings")
    seed = mo.ui.slider(0, 30, step=1, value=0, label="random seed")
    mo.vstack([noise, detection, clutter, miss, seed])
    return clutter, detection, miss, noise, seed


@app.cell
def _(clutter, detection, make_realistic_scene, miss, noise, seed):
    scene = make_realistic_scene(
        n_particles=10,
        n_frames=25,
        noise=noise.value,
        detection_prob=detection.value,
        clutter=clutter.value,
        miss_distance=miss.value,
        seed=seed.value,
    )
    return (scene,)


@app.cell
def _(mo, np, plt, scene):
    def _axes():
        fig = plt.figure(figsize=(5.5, 5))
        return fig, fig.add_subplot(projection="3d")

    _fig1, _ax1 = _axes()
    for _p in range(scene.n_particles):
        _ax1.plot(*scene.paths[_p].T, lw=1.5)
    _ax1.set_title("The truth: 10 particles, 3 pairs cross")

    _fig2, _ax2 = _axes()
    _all = np.vstack(scene.frames)
    _ax2.scatter(*_all.T, s=6, c="k")
    _ax2.set_title(f"What the camera sees: {len(_all)} dots")
    mo.hstack([mo.as_html(_fig1), mo.as_html(_fig2)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Scoring: how do we know a tracker is good?

    We know the truth, so we can count **links** (dot in frame *t* → dot in
    frame *t+1*):

    - **precision**: how many of the tracker's links are real,
    - **recall**: how many real links it found,
    - **F1**: one number combining both (1.0 is perfect),
    - **switches**: links that join two *different* particles. This is the
      dreaded identity swap.
    """)
    return


@app.cell
def _(mo, scene, score_tracks, simpletracker):
    base_tracks, _, _ = simpletracker(
        scene.frames, max_linking_distance=3.0, max_gap_closing=3
    )
    base_score = score_tracks(base_tracks, scene.truth)
    mo.md(
        f"**Distance-only tracker**: F1 = {base_score['f1']:.3f}, "
        f"identity switches = {base_score['switches']}, "
        f"links on false dots = {base_score['clutter_links']}"
    )
    return (base_score,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Idea 1: predict where the particle should be

    A firefly does not teleport. If it moved by (+1, 0) last time, we expect
    another (+1, 0). So instead of asking *"which dot is nearest to where it
    was?"* we ask *"which dot is nearest to where it should be?"*

    We also track **how sure** we are. A **Kalman filter** keeps a guess of
    position and velocity plus an uncertainty. We measure distance in units of
    that uncertainty (the *Mahalanobis distance*): a dot 1 unit away is
    suspicious if we are very sure, but fine if we are unsure.

    Below, two particles cross at a steep angle. Move the slider and compare
    which dot each method would pick for particle **A**.
    """)
    return


@app.cell
def _(mo):
    step = mo.ui.slider(2, 8, step=1, value=5, label="frame")
    step
    return (step,)


@app.cell
def _(
    mahalanobis_costs,
    make_motion_model,
    mo,
    np,
    plt,
    predict,
    step,
    update,
):
    _model = make_motion_model(2, dt=1.0, measurement_sigma=0.1, acceleration_sigma=0.05)
    _t = np.arange(10)
    _a = np.stack([_t * 1.0, _t * 0.5], axis=1)
    _b = np.stack([_t * 1.0, 4.5 - _t * 0.5], axis=1)

    _states, _covs = [], []
    for _path in (_a, _b):
        _x = np.array([_path[0, 0], _path[0, 1], 1.0, _path[1, 1] - _path[0, 1]])
        _p = np.diag([0.01, 0.01, 0.5, 0.5])
        _x, _p = predict(_x, _p, _model)
        for _k in range(1, step.value):
            _x, _p = update(_x, _p, _path[_k], _model)
            _x, _p = predict(_x, _p, _model)
        _states.append(_x)
        _covs.append(_p)

    _dots = np.array([_a[step.value], _b[step.value]])
    _preds = np.array([_s[:2] for _s in _states])
    _dist = np.linalg.norm(_preds[:, None] - _dots[None], axis=2)
    _maha = mahalanobis_costs(np.array(_states), np.array(_covs), _dots, _model)

    _fig, _ax = plt.subplots(figsize=(5.5, 4))
    _ax.plot(_a[:, 0], _a[:, 1], ":", c="tab:blue")
    _ax.plot(_b[:, 0], _b[:, 1], ":", c="tab:orange")
    _ax.scatter(*_dots.T, c=["tab:blue", "tab:orange"], s=60, label="dots now")
    _ax.scatter(*_preds.T, marker="x", c="k", s=60, label="predictions")
    _ax.set_aspect("equal")
    _ax.legend()
    _ax.set_title(f"frame {step.value}: dot 0 is truly A, dot 1 is truly B")

    _fmt = lambda m: "<br>".join(  # noqa: E731
        f"track {'AB'[i]}: " + ", ".join(f"{v:.1f}" for v in row) for i, row in enumerate(m)
    )
    mo.hstack(
        [
            mo.as_html(_fig),
            mo.md(
                f"**Euclidean distance** from each prediction to dot 0, dot 1:<br>{_fmt(_dist)}"
                f"<br><br>**Mahalanobis cost** (squared):<br>{_fmt(_maha)}"
                "<br><br>Each track should pick the smaller number in its row."
            ),
        ],
        align="center",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Idea 2: when a dot is missing, keep going

    If no dot falls near the prediction, the track does not die. It **coasts**
    on its prediction, and its uncertainty **grows** each frame, so the search
    area widens. If a dot appears again inside it, the track continues. After
    `max_missed` empty frames the track is dropped. This replaces the old
    separate gap-closing step.

    One more trick: tracks that have been confirmed by several dots choose
    first. A brand-new track knows nothing about its velocity, so its
    uncertainty is huge and it would otherwise steal dots from veterans.
    """)
    return


@app.cell
def _(base_score, kalman_tracker, mo, noise, scene, score_tracks):
    kal_tracks, _, _ = kalman_tracker(
        scene.frames, measurement_sigma=max(noise.value, 0.05), acceleration_sigma=0.1,
        speed_sigma=0.7,
    )
    kal_score = score_tracks(kal_tracks, scene.truth)
    mo.md(
        f"**Kalman tracker**: F1 = {kal_score['f1']:.3f} "
        f"(distance-only {base_score['f1']:.3f}), "
        f"switches = {kal_score['switches']} (was {base_score['switches']}), "
        f"links on false dots = {kal_score['clutter_links']} "
        f"(was {base_score['clutter_links']})"
    )
    return kal_score, kal_tracks


@app.cell
def _(mo):
    mo.md(r"""
    ## Idea 3: look at whole tracks, not single steps

    Even with prediction, two particles at a crossing can briefly trade
    identities for one frame and then each recovers. A single step cannot
    see this, but a whole track can: a track that jumps sideways and back is
    **unlikely** as physical motion.

    So we give each track a **plausibility score** (its Kalman negative
    log-likelihood: lower is smoother). For every pair of tracks that pass
    close to each other, we try swapping what they hold for a few frames (or
    from there to the end), and keep a swap whenever it lowers the total.
    """)
    return


@app.cell
def _(
    kal_score,
    kal_tracks,
    mo,
    noise,
    resolve_crossings,
    scene,
    score_tracks,
):
    fixed_tracks, n_swaps = resolve_crossings(
        kal_tracks,
        scene.frames,
        measurement_sigma=max(noise.value, 0.05),
        acceleration_sigma=0.1,
        speed_sigma=0.7,
        swap_radius=2.0,
    )
    fixed_score = score_tracks(fixed_tracks, scene.truth)
    mo.md(
        f"**Kalman + crossing repair** made {n_swaps} swap(s): "
        f"F1 = {fixed_score['f1']:.3f} (before {kal_score['f1']:.3f}), "
        f"switches = {fixed_score['switches']} (before {kal_score['switches']})"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Benchmark: does it really help?

    One scene proves nothing, so we average over **20 random scenes** per
    setting. The detector's noise level is given to the filters (an
    *oracle*; in practice it must be estimated).
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Run the benchmark (about a minute)")
    run
    return (run,)


@app.cell
def _(
    kalman_tracker,
    make_realistic_scene,
    mo,
    np,
    resolve_crossings,
    run,
    score_tracks,
    simpletracker,
):
    mo.stop(not run.value, mo.md("*Press the button to run.*"))

    def evaluate(n_scenes=20, **scene_args):
        rows = {"distance": [], "kalman": [], "kalman + repair": []}
        sigma = max(scene_args.get("noise", 0.2), 0.05)
        options = dict(measurement_sigma=sigma, acceleration_sigma=0.1, speed_sigma=0.7)
        for s in range(n_scenes):
            sc = make_realistic_scene(seed=100 + s, **scene_args)
            base, _, _ = simpletracker(sc.frames, max_linking_distance=3.0, max_gap_closing=3)
            kal, _, _ = kalman_tracker(sc.frames, **options)
            fix, _ = resolve_crossings(kal, sc.frames, swap_radius=2.0, **options)
            for name, tr in zip(rows, (base, kal, fix)):
                r = score_tracks(tr, sc.truth)
                rows[name].append((r["f1"], r["switches"]))
        return {k: np.mean(v, axis=0) for k, v in rows.items()}

    def table(title, values, key):
        head = f"| {title} | " + " | ".join(["distance", "kalman", "kalman + repair"]) + " |"
        lines = [head, "|---|---|---|---|"]
        for v, res in zip(values, key):
            lines.append(
                f"| {v} | "
                + " | ".join(f"{res[n][0]:.3f} / {res[n][1]:.1f}" for n in res)
                + " |"
            )
        return "\n".join(lines)

    noises = [0.1, 0.2, 0.4, 0.6]
    by_noise = [evaluate(noise=n) for n in noises]
    clutters = [0.0, 1.0, 3.0, 6.0]
    by_clutter = [evaluate(noise=0.2, clutter=c) for c in clutters]
    mo.md(
        "Each cell is **F1 / identity switches per scene**.\n\n"
        + table("noise", noises, by_noise)
        + "\n\n"
        + table("false dots per frame", clutters, by_clutter)
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What we learned (and what is still hard)

    From the benchmark (20 scenes per setting; the README lists the numbers we measured):

    - With little noise, prediction plus repair cuts identity switches
      roughly **from 9 to under 1** per scene.
    - As noise grows the advantage shrinks. At noise 0.4, most remaining
      switches happen where two particles are closer than the noise itself.
      There the dots are **truly indistinguishable**; no algorithm can do
      better without extra information (colour, size, a better camera).
    - **Weakness:** with many false dots (3 to 6 per frame) the Kalman
      tracker's F1 is *lower* than the plain distance tracker's (0.90 versus
      0.94 at 6), even though it makes fewer identity switches. Wide gates of
      coasting tracks invite false dots to join. Clutter needs its own
      handling (e.g. requiring confirmation before coasting).
    - The filters were told the true noise. Estimating it from data is
      another step.
    - A tempting idea that **did not work**: track forwards and backwards,
      cut where they disagree, and stitch the pieces. Short pieces have
      unknown velocity, so wrong stitches looked cheap.

    **Take-away for students:** a better algorithm comes from using more
    knowledge (particles move smoothly), and a good scientist measures
    whether it helped, including where it did not.
    """)
    return


if __name__ == "__main__":
    app.run()
