"""Mécanique 1 : une balle grossit à chaque rebond jusqu'à remplir le cercle.

Variante « escape » : une ouverture tourne ; la balle doit sortir avant d'être trop grosse.
"""
import numpy as np

from ..palettes import NAMED, bgr, pick_theme
from ..physics import Balls, collide_ring, escaped, rotate
from ..scene import RING_R, Timeline
from .common import (CENTER, MAX_FINALE, MIN_FINALE, SIM_FPS, TAIL, Event, duration_score, snapshot,
                     wall_point)

NAME = "grow_ball"
SUB = 8

TITLES = {
    "fill": ["Can it fill the circle?", "Every bounce makes it bigger...", "How many bounces to fill the ring?",
             "Will it fill the whole circle?"],
    "escape": ["Can it escape before it's too big?", "It grows on every bounce. Can it get out?",
               "Escape or get trapped?"],
}


DESCRIPTIONS = ["A ball that grows bigger with every single bounce.", "One ball, one circle, and it grows on every hit.",
                "Every bounce adds a little size. Watch till the end!"]
HASHTAGS = ["#shorts", "#satisfying", "#simulation", "#physics", "#oddlysatisfying"]


def sample(rng):
    variant = "escape" if rng.random() < 0.45 else "fill"
    color = list(NAMED)[rng.integers(len(NAMED))]
    p = dict(
        variant=variant,
        theme=pick_theme(rng),
        color=color,
        root=int(rng.integers(55, 66)),
        r0=float(rng.uniform(16, 30)),
        growth=float(rng.uniform(2.0, 5.0)),
        gravity=float(rng.uniform(1100, 2600)),
        speed=float(rng.uniform(900, 1500)),
        jitter=float(np.radians(rng.uniform(2, 8))),
        gap_half=float(np.radians(rng.uniform(17, 26))),
        omega=float(rng.uniform(0.5, 1.4) * rng.choice([-1, 1])),
        trails=int(rng.integers(5, 10)),
    )
    if variant == "escape":
        # Balle déjà assez grosse : chaque passage devant l'ouverture devient une vraie chance, pas une formalité.
        p["r0"] = float(RING_R * np.sin(p["gap_half"]) * rng.uniform(0.55, 0.7))
        p["growth"] = float(rng.uniform(0.6, 1.2))
    p["title"] = TITLES[variant][rng.integers(len(TITLES[variant]))]
    return p


def _run(seed, p):
    rng = np.random.default_rng(seed)
    R = RING_R
    b = Balls()
    ang = rng.uniform(0, 2 * np.pi)
    b.add(CENTER + rng.uniform(-60, 60, 2), [np.cos(ang) * p["speed"], np.sin(ang) * p["speed"]], p["r0"], 0)
    col = bgr(NAMED[p["color"]])
    ring_col = bgr(p["theme"]["ring"])
    accent = bgr(p["theme"]["accent"])
    esc = p["variant"] == "escape"
    phi = rng.uniform(0, 2 * np.pi)
    dt = 1 / SIM_FPS / SUB
    frames, events = [], []
    bounces, finale, outcome = 0, None, None
    too_big_frame = None
    esc_margin = None
    chord_r = R * np.sin(p["gap_half"])          # rayon max qui passe par l'ouverture
    max_frames = int((MAX_FINALE + 2) * SIM_FPS)
    f = 0
    while f < max_frames:
        if finale is None:
            for _ in range(SUB):
                b.vel[:, 1] += p["gravity"] * dt
                b.pos += b.vel * dt
                hit, vn, _ = collide_ring(b, CENTER, R, 1.0, phi if esc else None, p["gap_half"])
                if len(hit):
                    bounces += 1
                    b.vel = rotate(b.vel, rng.uniform(-p["jitter"], p["jitter"]))
                    sp = np.linalg.norm(b.vel[0])
                    if sp < p["speed"] * 0.7:
                        b.vel *= p["speed"] * 0.7 / (sp + 1e-9)
                    b.rad[0] = min(R, b.rad[0] + p["growth"])
                    frac = b.rad[0] / R
                    x, y = wall_point(b.pos[0], b.rad[0])
                    note = int(frac * 9) + [0, 2, 1, 3][bounces % 4]
                    events.append(Event(f, "bounce", x, y, col, 0.6 + 0.4 * frac, note))
                    # Si la balle a grossi contre la paroi, la recentrer pour rester dans le cercle.
                    d = b.pos[0] - CENTER
                    dist = np.linalg.norm(d)
                    if dist + b.rad[0] > R:
                        b.pos[0] = CENTER + d / (dist + 1e-9) * (R - b.rad[0])
                if esc and len(escaped(b, CENTER, R)):
                    finale, outcome = f, "escaped"
                    esc_margin = float(1 - b.rad[0] / chord_r)
                    events.append(Event(f, "boom", *b.pos[0], col, 1.0, 7))
                    b.remove(0)
                    break
                if b.rad[0] >= R * 0.965:
                    finale, outcome = f, ("trapped" if esc else "full")
                    b.rad[0] = R
                    b.pos[0] = CENTER.copy()
                    b.vel[:] = 0
                    events.append(Event(f, "boom", *CENTER, col, 1.0, 5))
                    break
            if esc:
                phi += p["omega"] / SIM_FPS
                if finale is None and b.rad[0] > chord_r:
                    too_big_frame = finale = f
                    outcome = "trapped"
                    events.append(Event(f, "boom", *b.pos[0], col, 1.0, 2))
        size = (b.rad[0] / R) if len(b) else None
        hud = []
        if size is not None:
            hud.append(("text", f"SIZE {int(round(size * 100))}%", 520, 430, 64, accent, "c"))
            hud.append(("text", f"{bounces} BOUNCE" + ("S" if bounces != 1 else ""), 520, 505, 40, (230, 230, 230), "c"))
        else:
            hud.append(("text", f"{bounces} BOUNCE" + ("S" if bounces != 1 else ""), 520, 470, 52, accent, "c"))
        if esc and size is not None:
            fit = max(0.0, 1 - (b.rad[0] - p["r0"]) / (chord_r - p["r0"]))
            hud.append(("bar", 320, 560, 400, 16, (80, 220, 120) if fit > 0.3 else (80, 80, 255), fit))
        rings = [(R, phi if esc else None, p["gap_half"], ring_col)]
        frames.append(snapshot(b, [col] * len(b), rings, hud))
        f += 1
        if finale is not None and f >= finale + int(TAIL * SIM_FPS):
            break
    return frames, events, finale, outcome, bounces, too_big_frame, esc_margin


def simulate(seed, p):
    # Ajuste la croissance pour que la fin tombe dans la fenêtre de durée (plusieurs essais).
    # Variante escape : une évasion trop précoce relance avec une autre sous-graine.
    tries = 30 if p["variant"] == "escape" else 6
    for k in range(tries):
        sub = seed + 7919 * k
        frames, events, finale, outcome, bounces, too_big, margin = _run(sub, p)
        t = finale / SIM_FPS if finale is not None else 99
        if MIN_FINALE <= t <= MAX_FINALE:
            p["sub_seed"] = sub
            break
        if outcome != "escaped":
            p["growth"] *= float(np.clip(t / 42.0, 0.5, 2.0))
    else:
        return None
    texts = {"full": ("FULL!", "accent"), "escaped": ("ESCAPED!", "ball"), "trapped": ("TOO BIG!", "red")}
    txt, ck = texts[outcome]
    fcol = {"accent": bgr(p["theme"]["accent"]), "ball": bgr(NAMED[p["color"]]), "red": (70, 70, 255)}[ck]
    return Timeline(
        mechanic=NAME, seed=seed, params=p, title=p["title"], frames=frames, events=events,
        finale_frame=finale, finale_text=txt, finale_color=fcol,
        end_sub=f"{bounces} bounces", palette=p["theme"], trails=p["trails"],
        metrics=dict(outcome=outcome, bounces=bounces, finale_s=round(finale / SIM_FPS, 2),
                     too_big_s=round(too_big / SIM_FPS, 2) if too_big else None,
                     escape_margin=margin),
    )


def score(tl):
    m = tl.metrics
    s = 0.6 * duration_score(m["finale_s"]) + 0.2
    if m["outcome"] == "escaped":
        s += 0.15 + 0.1 * max(0.0, 1 - m["escape_margin"] / 0.5)   # évasion de justesse = mieux
    elif m["outcome"] == "trapped":
        s += 0.2
    else:
        s += 0.2
    return min(1.0, s)
