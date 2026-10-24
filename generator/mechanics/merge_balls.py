"""Mécanique 3 : fusion de balles dans un cercle avec gravité.

Deux balles de même niveau qui se touchent fusionnent en une balle du niveau suivant.
Objectif : atteindre la plus grosse. Une rangée de niveaux à découvrir (« ? ») en haut.
Le niveau objectif est choisi après la simulation (le plus haut atteint dans la fenêtre de durée).
Fin alternative « FULL! » si le cercle déborde.
"""
import numpy as np

from .. import objects
from ..palettes import bgr, pick_theme
from ..physics import Balls, collide_balls, collide_ring
from ..scene import RING_R, Timeline
from .common import (CENTER, MAX_FINALE, MIN_FINALE, SIM_FPS, TAIL, Event, duration_score, snapshot,
                     strip_events_after)

NAME = "merge_balls"
SUB = 10

TITLES = {
    "planets": ["Can it make {name}?", "Moon to {name}... can it get there?", "Merge planets until {name}?",
                "Will it reach {name}?"],
    "cats": ["Can it make {name}?", "Kitten to {name}... can it get there?", "Merge cats until {name}?",
             "Will it reach {name}?"],
}


def _with_article(oset, name):
    if oset == "cats" or name in ("MOON", "SUN"):
        return f"the {name.title()}"
    return name.title()


DESCRIPTIONS = {
    "planets": ["Two identical planets merge into the next one, all the way up to the Sun?",
                "From the Moon to the Sun, one merge at a time.", "Same planets fuse together under gravity."],
    "cats": ["Two identical cats merge into a bigger cat. Meow!", "From tiny kitten to King Cat, one merge at a time.",
             "Same cats fuse together. Each one meows a little deeper."],
}
HASHTAGS = {"planets": ["#shorts", "#satisfying", "#simulation", "#planets", "#merge"],
            "cats": ["#shorts", "#cat", "#satisfying", "#simulation", "#funny"]}
OVERLAYS = ["", "", "{name}?", ""]


def sample(rng):
    return dict(
        theme=pick_theme(rng),
        objects=str(rng.choice(objects.SETS)),
        root=int(rng.integers(55, 66)),
        r0=float(rng.uniform(22, 30)),
        ratio=float(rng.uniform(1.22, 1.3)),
        gravity=float(rng.uniform(650, 1100)),     # chute lente (demande du 2026-10-24)
        spawn0=float(rng.uniform(0.22, 0.34)),
        spawn1=float(rng.uniform(0.10, 0.16)),
        p_levels=[0.5, 0.3, 0.2] if rng.random() < 0.5 else [0.6, 0.3, 0.1],
        title_idx=int(rng.integers(4)),
        overlay_idx=int(rng.integers(len(OVERLAYS))),
        trails=0,
    )


def simulate(seed, p):
    rng = np.random.default_rng(seed)
    R = RING_R
    ramp = list(objects.colors(p["objects"]))
    names = objects.names(p["objects"])
    lmax = min(len(ramp) - 1, int(np.log(0.5 * R / p["r0"]) / np.log(p["ratio"])))
    radius = [p["r0"] * p["ratio"] ** L for L in range(lmax + 1)]
    ring_col = bgr(p["theme"]["ring"])
    b = Balls()
    dt = 1 / SIM_FPS / SUB
    first_reach = {0: 0}
    score = 0
    frames, events, score_hist = [], [], []
    spawn_y = CENTER[1] - R * 0.72
    next_spawn, blocked_since, full_frame = 0.0, None, None
    # Accroche : quelques balles déjà en chute dès la première image.
    for _ in range(6):
        L = int(rng.choice(3, p=p["p_levels"]))
        b.add([CENTER[0] + rng.uniform(-200, 200), CENTER[1] + rng.uniform(-250, 50)], rng.uniform(-200, 200, 2), radius[L], L)
    max_frames = int((MAX_FINALE + TAIL + 0.5) * SIM_FPS)
    for f in range(max_frames):
        t = f / SIM_FPS
        if t >= next_spawn and full_frame is None:
            L = int(rng.choice(3, p=p["p_levels"]))
            x = CENTER[0] + rng.uniform(-0.38, 0.38) * R
            free = len(b) == 0 or np.all(np.hypot(*(b.pos - [x, spawn_y]).T) > b.rad + radius[L] + 2)
            if free:
                b.add([x, spawn_y], [rng.uniform(-60, 60), 80], radius[L], L)
                events.append(Event(f, "spawn", x, spawn_y, ramp[L], 0.4, L))
                frac = min(1, t / 45)
                next_spawn = t + p["spawn0"] * (1 - frac) + p["spawn1"] * frac
                blocked_since = None
            elif blocked_since is None:
                blocked_since = f
            elif f - blocked_since > 1.5 * SIM_FPS:
                full_frame = f
        for _ in range(SUB):
            b.vel[:, 1] += p["gravity"] * dt
            b.vel *= 0.9995
            b.pos += b.vel * dt
            i, j, rv = collide_balls(b, 0.15, 0.9, 400)
            for a_, v in zip(i[:1], rv[:1]):
                events.append(Event(f, "hit", *b.pos[a_], ramp[b.kind[a_]], min(1, v / 1500), int(b.kind[a_])))
            collide_ring(b, CENTER, R, 0.3)
            # Fusions : paires en contact de même niveau.
            if len(b) > 1:
                d = b.pos[:, None] - b.pos[None]
                dist = np.sqrt((d ** 2).sum(-1))
                same = (b.kind[:, None] == b.kind[None]) & (dist < b.rad[:, None] + b.rad[None] + 1.0)
                same = np.triu(same, 1)
                ii, jj = np.nonzero(same)
                used, rm = set(), []
                for a_, c_ in zip(ii, jj):
                    if a_ in used or c_ in used or b.kind[a_] >= lmax:
                        continue
                    used.update((a_, c_))
                    L = int(b.kind[a_]) + 1
                    pos = (b.pos[a_] + b.pos[c_]) / 2
                    vel = (b.vel[a_] + b.vel[c_]) / 2
                    rm += [a_, c_]
                    b.add(pos, vel, radius[L], L, spin=(b.spin[a_] + b.spin[c_]) / 2, ang=b.ang[a_])
                    score += 2 ** L
                    if L not in first_reach:
                        first_reach[L] = f
                        events.append(Event(f, "chime", *pos, ramp[L], 1.0, L + 2))
                    else:
                        events.append(Event(f, "pop", *pos, ramp[L], 0.8, L))
                if rm:
                    b.remove(rm)
        score_hist.append(score)
        frames.append(snapshot(b, [ramp[c] for c in b.kind], [(R, None, 0, ring_col)]))
        if full_frame is not None and f > full_frame + TAIL * SIM_FPS:
            break

    # Choix du niveau objectif : le plus haut atteint dans la fenêtre.
    goal = None
    for L in sorted(first_reach, reverse=True):
        tL = first_reach[L] / SIM_FPS
        if tL > MAX_FINALE:
            continue
        if full_frame is not None and first_reach[L] > full_frame:
            continue
        if tL >= MIN_FINALE and L >= 5:
            goal = L
        break
    outcome = "max"
    if goal is not None:
        fin = first_reach[goal]
    elif full_frame is not None and MIN_FINALE <= full_frame / SIM_FPS <= MAX_FINALE:
        # Débordement : l'objectif affiché est le niveau suivant celui atteint, non atteint.
        goal = max(L for L in first_reach if first_reach[L] <= full_frame) + 1
        if goal > lmax or goal < 6:
            return None
        fin, outcome = full_frame, "full"
    else:
        return None
    frames = frames[: fin + int(TAIL * SIM_FPS)]
    events = strip_events_after(events, fin)
    fx, fy = CENTER
    if outcome == "max":
        k = int(np.argmax(frames[fin].balls[:, 2]))
        fx, fy = frames[fin].balls[k, :2]
    events.append(Event(fin, "boom", fx, fy, ramp[min(goal, lmax)], 1.0, goal))
    if outcome == "max":
        events.append(Event(fin, "win", fx, fy, ramp[goal], 1.0, 0))

    accent = bgr(p["theme"]["accent"])
    n_icons = goal + 1
    rs = np.linspace(20, 36, n_icons)
    gaps = 10
    total = rs.sum() * 2 + gaps * (n_icons - 1)
    if total > 880:
        rs *= 880 / total
        total = 880
    xs, x = [], 520 - total / 2
    for r in rs:
        xs.append(x + r)
        x += 2 * r + gaps
    for fi, fr in enumerate(frames):
        fz = min(fi, fin)
        sc = score_hist[fz]
        hud = []
        for L in range(n_icons):
            if L in first_reach and first_reach[L] <= fz:
                hud.append(("obj", p["objects"], L, xs[L], 470, rs[L]))
            else:
                hud.append(("qmark", xs[L], 470, rs[L]))
        fr.hud = hud
    title = TITLES[p["objects"]][p["title_idx"]].format(name=_with_article(p["objects"], names[goal]))
    title = title[0].upper() + title[1:]
    txt, col = (f"{names[goal]}!", ramp[goal]) if outcome == "max" else ("FULL!", (70, 70, 255))
    return Timeline(
        mechanic=NAME, seed=seed, params=p, title=title, frames=frames, events=events,
        finale_frame=fin, finale_text=txt, finale_color=col, end_sub="", overlay=OVERLAYS[p["overlay_idx"]].format(name=names[goal].title()),
        palette=p["theme"], trails=0,
        metrics=dict(outcome=outcome, goal_level=goal, lmax=lmax, finale_s=round(fin / SIM_FPS, 2),
                     score=int(score_hist[fin])),
    )


def score(tl):
    m = tl.metrics
    s = 0.45 * duration_score(m["finale_s"]) + 0.35 * m["goal_level"] / max(1, m["lmax"])
    return s + (0.2 if m["outcome"] == "max" else 0.12)
