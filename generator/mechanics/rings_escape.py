"""Mécanique 6 : une balle doit s'échapper de N anneaux concentriques qui tournent (style « Will the ball escape? »).

Chaque anneau a une ouverture. Quand la balle passe entièrement dehors, l'anneau éclate en morceaux
et la balle rebondit désormais contre le suivant. Fin : le dernier anneau tombe (« FREE! »).
Durée réglée en élargissant ou en resserrant les ouvertures entre deux essais.
"""
import colorsys

import numpy as np

from ..palettes import NAMED, bgr, pick_theme
from ..physics import Balls, collide_ring, rotate
from ..scene import RING_R, Timeline
from .common import CENTER, MAX_FINALE, MIN_FINALE, SIM_FPS, TAIL, Event, duration_score, snapshot, wall_point

NAME = "rings_escape"
SUB = 8

TITLES = ["Can it escape {n} rings?", "Will the ball escape?", "{n} rings. One ball. Can it get out?",
          "Escape all {n} rings?"]
DESCRIPTIONS = ["One ball, {n} spinning rings, one gap each.", "Every ring it escapes shatters. Can it get out?",
                "Spinning rings, tiny gaps, and a very determined ball."]
HASHTAGS = ["#shorts", "#satisfying", "#simulation", "#willtheballescape", "#oddlysatisfying"]
OVERLAYS = ["", "", "Escape?", ""]


def sample(rng):
    n = int(rng.integers(7, 14))
    p = dict(
        theme=pick_theme(rng),
        root=int(rng.integers(52, 64)),
        n=n,
        r_in=float(rng.uniform(70, 95)),
        color=list(NAMED)[rng.integers(len(NAMED))],
        gravity=float(rng.uniform(900, 1500)),
        speed=float(rng.uniform(650, 950)),
        jitter=float(np.radians(rng.uniform(2, 7))),
        gap=float(rng.uniform(0.28, 0.42)),       # demi-ouverture de base (rad)
        omega=float(rng.uniform(0.7, 1.5)),
        hue0=float(rng.random()),
        alt=bool(rng.random() < 0.6),             # sens de rotation alterné d'un anneau à l'autre
        title_idx=int(rng.integers(len(TITLES))),
        overlay_idx=int(rng.integers(len(OVERLAYS))),
        trails=int(rng.integers(6, 11)),
    )
    sp = (RING_R - p["r_in"]) / (n - 1)
    p["r0"] = float(min(18.0, sp * 0.38))
    if sp > 44 and rng.random() < 0.5:            # anneaux espacés : parfois un chat à la place de la balle
        p["objects"] = "cats"
        p["cat_level"] = int(rng.integers(10))
        p["r0"] = float(min(20.0, sp * 0.4))
    return p


def _rings(p):
    n = p["n"]
    radii = np.linspace(p["r_in"], RING_R, n)
    th = float(np.clip((RING_R - p["r_in"]) / (n - 1) * 0.22, 4, 9))
    cols = [bgr(tuple(int(255 * c) for c in colorsys.hsv_to_rgb((p["hue0"] + k / n * 0.8) % 1, 0.75, 1.0)))
            for k in range(n)]
    return radii, th, cols


def _run(seed, p, gap_scale):
    rng = np.random.default_rng(seed)
    radii, th, rcols = _rings(p)
    n = p["n"]
    # Ouverture : au moins 1,5 diamètre de balle de corde, sinon `gap` x échelle.
    gaps = np.maximum(p["gap"] * gap_scale, 1.5 * p["r0"] / radii)
    gaps = np.minimum(gaps, 1.2)
    phis = rng.uniform(0, 2 * np.pi, n)
    oms = p["omega"] * rng.uniform(0.75, 1.25, n) * (np.where(np.arange(n) % 2, -1, 1) if p["alt"] else rng.choice([-1, 1]))
    b = Balls()
    a = rng.uniform(0, 2 * np.pi)
    b.add(CENTER.copy(), [np.cos(a) * p["speed"], np.sin(a) * p["speed"]], p["r0"], p.get("cat_level", 0))
    col = bgr(NAMED[p["color"]])
    if p.get("objects") == "cats":
        from ..objects import CAT_COLORS
        col = CAT_COLORS[p["cat_level"]]
    dt = 1 / SIM_FPS / SUB
    k = 0                          # anneau courant (le plus intérieur encore debout)
    frames, events, breaks = [], [], []
    fin = None
    bounces = 0
    max_frames = int((MAX_FINALE + 2) * SIM_FPS)
    for f in range(max_frames):
        if fin is None:
            for _ in range(SUB):
                b.vel[:, 1] += p["gravity"] * dt
                b.pos += b.vel * dt
                hit, vn, _ = collide_ring(b, CENTER, radii[k], 1.0, phis[k], gaps[k])
                if len(hit):
                    bounces += 1
                    b.vel = rotate(b.vel, rng.uniform(-p["jitter"], p["jitter"]))
                    s = np.linalg.norm(b.vel[0])
                    if s < p["speed"] * 0.75:
                        b.vel *= p["speed"] * 0.75 / (s + 1e-9)
                    x, y = wall_point(b.pos[0], b.rad[0])
                    events.append(Event(f, "bounce", x, y, rcols[k], 0.5 + 0.5 * k / n, k + bounces % 3))
                if np.linalg.norm(b.pos[0] - CENTER) - b.rad[0] > radii[k] + th:
                    # L'anneau éclate : éclats tout autour, carillon de plus en plus aigu.
                    for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                        if abs((ang - phis[k] + np.pi) % (2 * np.pi) - np.pi) > gaps[k]:
                            events.append(Event(f, "shard", *(CENTER + radii[k] * np.array([np.cos(ang), np.sin(ang)])), rcols[k], 1, 0))
                    events.append(Event(f, "chime", *b.pos[0], rcols[k], 0.9, k + 2))
                    breaks.append(f)
                    k += 1
                    if k == n:
                        fin = f
                        events.append(Event(f, "boom", *b.pos[0], col, 1.0, n + 2))
                        events.append(Event(f, "win", *CENTER, col, 1.0, 0))
                        break
            phis += oms / SIM_FPS
        rings = [(radii[j], phis[j], gaps[j], rcols[j], th) for j in range(k, n)]
        frames.append(snapshot(b, [col] * len(b), rings))
        if fin is not None and f >= fin + int(TAIL * SIM_FPS):
            break
    return frames, events, fin, breaks, bounces


def simulate(seed, p):
    scale = 1.0
    for tr in range(14):
        frames, events, fin, breaks, bounces = _run(seed + 7919 * tr, p, scale)
        t = fin / SIM_FPS if fin is not None else None
        if t is not None and MIN_FINALE <= t <= MAX_FINALE:
            break
        # Trop rapide : ouvertures plus étroites ; trop lent : plus larges (selon l'avancement).
        prog = len(breaks) / p["n"]
        scale *= 0.8 if t is not None else float(np.clip(1.0 + 0.6 * (1 - prog), 1.05, 1.6))
    else:
        return None
    p["gap_scale"] = scale
    p["sub_seed"] = seed + 7919 * tr
    gaps_t = np.diff([0] + breaks) / SIM_FPS
    col = bgr(NAMED[p["color"]]) if p.get("objects") != "cats" else (255, 255, 255)
    return Timeline(
        mechanic=NAME, seed=seed, params=p, title=TITLES[p["title_idx"]].format(n=p["n"]), frames=frames, events=events,
        finale_frame=fin, finale_text="FREE!", finale_color=col, end_sub="", overlay=OVERLAYS[p["overlay_idx"]],
        palette=p["theme"], trails=p["trails"],
        metrics=dict(finale_s=round(fin / SIM_FPS, 2), n=p["n"], bounces=bounces,
                     last_wait=round(float(gaps_t[-1]), 2), max_wait=round(float(gaps_t.max()), 2)),
    )


def score(tl):
    m = tl.metrics
    # Durée, puis suspense : le dernier anneau doit résister un peu, sans qu'une attente écrase tout le reste.
    s = 0.6 * duration_score(m["finale_s"])
    s += 0.25 * min(1.0, m["last_wait"] / 4.0)
    s += 0.15 * (1.0 - min(1.0, m["max_wait"] / 20.0))
    return s


def comment(tl, rng):
    n = tl.metrics["n"]
    opts = [f"How many of the {n} rings did you think it would break? 👇", "Which ring was the hardest? 🤔👇",
            f"Should the next one have {n * 2} rings? 👇", "Be honest, did you think it would get out? 👀"]
    return str(rng.choice(opts))
