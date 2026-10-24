"""Mécanique 4 : chaque rebond de la balle « mère » ajoute une balle.

Une balle blanche lumineuse rebondit ; à chaque contact avec l'anneau, une nouvelle balle colorée
apparaît. Objectif : atteindre un nombre rond de balles (choisi après la simulation pour une fin
dans la fenêtre de durée).
"""
import numpy as np

from ..palettes import LEVEL_RAMPS, bgr, pick_theme
from ..physics import Balls, collide_balls, collide_ring, rotate
from ..scene import RING_R, Timeline
from .common import (CENTER, MAX_FINALE, MIN_FINALE, SIM_FPS, TAIL, Event, duration_score, snapshot,
                     strip_events_after, wall_point)

NAME = "bounce_spawn"
SUB = 6
TARGETS = [50, 64, 75, 100, 120, 128, 150, 200]

TITLES = ["Every bounce adds a ball. Can it reach {N}?", "Will it hit {N} balls?", "1 bounce = 1 new ball. {N} incoming?",
          "How fast can it make {N} balls?"]

DESCRIPTIONS = ["Every time the white ball bounces, a new ball appears.", "One bounce, one new ball. Watch it fill up!",
                "A single spawner ball and a lot of bounces."]
HASHTAGS = ["#shorts", "#satisfying", "#simulation", "#physics", "#bouncingball"]
OVERLAYS = ["", "", "{N}?", "1 bounce = 1 ball"]


def sample(rng):
    return dict(
        theme=pick_theme(rng),
        ramp=int(rng.integers(len(LEVEL_RAMPS))),
        root=int(rng.integers(55, 66)),
        mother_r=float(rng.uniform(26, 38)),
        rad=float(rng.uniform(11, 17)),
        gravity=float(rng.choice([0.0, 700.0, 1400.0, 2000.0])),
        speed=float(rng.uniform(700, 1200)),
        jitter=float(np.radians(rng.uniform(3, 10))),
        trails=int(rng.integers(0, 3)),   # peu de traînées : beaucoup de balles = fichiers lourds
        title_idx=int(rng.integers(len(TITLES))),
        overlay_idx=int(rng.integers(len(OVERLAYS))),
    )


def simulate(seed, p):
    rng = np.random.default_rng(seed)
    R = RING_R
    ramp = [bgr(c) for c in LEVEL_RAMPS[p["ramp"]]][:10]
    ring_col = bgr(p["theme"]["ring"])
    white = (255, 255, 255)
    b = Balls()
    a = rng.uniform(0, 2 * np.pi)
    b.add(CENTER, [np.cos(a) * p["speed"], np.sin(a) * p["speed"]], p["mother_r"], -1)
    dt = 1 / SIM_FPS / SUB
    frames, events, counts = [], [], []
    max_frames = int((MAX_FINALE + TAIL + 0.3) * SIM_FPS)
    for f in range(max_frames):
        for _ in range(SUB):
            b.vel[:, 1] += p["gravity"] * dt
            b.pos += b.vel * dt
            i, j, rv = collide_balls(b, 1.0, 0.8, 200)
            for a_, v in zip(i[:1], rv[:1]):
                events.append(Event(f, "hit", *b.pos[a_], white, min(1, v / 1000), 6))
            hit, _, _ = collide_ring(b, CENTER, R, 1.0)
            if 0 in set(hit.tolist()):
                b.vel[0] = rotate(b.vel[0], rng.uniform(-p["jitter"], p["jitter"]))
                n = len(b)
                c = ramp[(n - 1) % len(ramp)]
                ang = rng.uniform(0, 2 * np.pi)
                pos = b.pos[0] + (CENTER - b.pos[0]) * 0.15
                b.add(pos, [np.cos(ang) * p["speed"] * 0.8, np.sin(ang) * p["speed"] * 0.8], p["rad"], n)
                x, y = wall_point(b.pos[0], b.rad[0])
                events.append(Event(f, "bounce", x, y, c, 0.9, (n // 5) % 12 + [0, 2, 4, 1, 3][n % 5] // 2))
        if len(b):
            sp = np.linalg.norm(b.vel, axis=1) + 1e-9
            low = sp < p["speed"] * 0.5
            b.vel[low] *= (p["speed"] * 0.5 / sp[low])[:, None]
            if p["gravity"] == 0:
                b.vel *= (1 + 0.05 * (p["speed"] / sp - 1))[:, None]
            m = np.linalg.norm(b.vel[0])            # la mère garde toujours de l'élan
            if m < p["speed"]:
                b.vel[0] *= p["speed"] / m
        counts.append(len(b) - 1)
        cols = [white] + [ramp[(int(k) - 1) % len(ramp)] for k in b.kind[1:]]
        frames.append(snapshot(b, cols, [(R, None, 0, ring_col)]))
        if len(b) > max(TARGETS) + 5:
            break
    counts = np.array(counts)
    best = None
    for N in TARGETS:
        hit = np.nonzero(counts >= N)[0]
        if not len(hit):
            continue
        t = hit[0] / SIM_FPS
        if MIN_FINALE <= t <= MAX_FINALE:
            q = duration_score(t)
            if best is None or q > best[0]:
                best = (q, N, int(hit[0]))
    if best is None:
        return None
    _, N, fin = best
    frames = frames[: fin + int(TAIL * SIM_FPS)]
    events = strip_events_after(events, fin)
    events.append(Event(fin, "boom", *CENTER, ramp[0], 1.0, 4))
    events.append(Event(fin, "win", *CENTER, ramp[0], 1.0, 0))
    accent = bgr(p["theme"]["accent"])
    for fi, fr in enumerate(frames):
        c = int(counts[min(fi, fin)])
        fr.hud = [("text", f"{c}/{N}", 520, 470, 96, accent, "c")]
    return Timeline(
        mechanic=NAME, seed=seed, params=p, title=TITLES[p["title_idx"]].format(N=N), frames=frames, events=events,
        finale_frame=fin, finale_text=f"{N}!", finale_color=accent, end_sub="", overlay=OVERLAYS[p["overlay_idx"]].format(N=N),
        palette=p["theme"], trails=p["trails"],
        metrics=dict(N=N, finale_s=round(fin / SIM_FPS, 2), gravity=p["gravity"]),
    )


def score(tl):
    return 0.7 * duration_score(tl.metrics["finale_s"]) + 0.3


def comment(tl, rng):
    n = tl.metrics["N"]
    opts = [f"How long did you think it would take to hit {n}? ⏱️👇", f"Guess the time for {n * 2} next time! 👇",
            "Watch it again and count the bounces 👀👇"]
    return str(rng.choice(opts))
