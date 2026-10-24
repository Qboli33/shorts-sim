"""Mécanique 2 : course d'évasion par couleurs.

Un cercle tourne avec une ouverture. Chaque évasion rapporte 1 point à la couleur et fait apparaître
2 nouvelles balles de cette couleur au centre. La première couleur à N évasions gagne.
N est choisi après la simulation pour que la victoire tombe dans la fenêtre de durée.
"""
import numpy as np

from ..palettes import NAMED, bgr, pick_theme
from ..physics import Balls, collide_balls, collide_ring, escaped
from ..scene import RING_R, Timeline
from .common import (CENTER, MAX_FINALE, MIN_FINALE, SIM_FPS, TAIL, Event, duration_score, snapshot,
                     strip_events_after, wall_point)

NAME = "color_escape"
SUB = 6

TITLES = ["Which color escapes first?", "First to {N} escapes wins!", "Every escape spawns 2 more. Who wins?",
          "Pick your color before it starts!", "Which color reaches {N} first?"]


DESCRIPTIONS = ["Every escape spawns two more balls of the same color.", "A rotating gap, a few colors, one winner.",
                "Escape the ring, score a point, multiply. Which color did you pick?"]
HASHTAGS = ["#shorts", "#satisfying", "#simulation", "#colorrace", "#marblerace"]


def sample(rng):
    k = int(rng.choice([2, 3, 3, 4, 4, 5]))
    names = list(rng.choice([n for n in NAMED if n not in ("WHITE", "LIME")], k, replace=False))
    return dict(
        theme=pick_theme(rng),
        colors=names,
        root=int(rng.integers(55, 66)),
        start=int(rng.integers(1, 3)),
        rad=float(rng.uniform(14, 22)),
        gravity=float(rng.choice([0.0, 0.0, 500.0, 1200.0])),
        speed=float(rng.uniform(450, 850)),
        gap_half=float(np.radians(rng.uniform(12, 24))),
        omega=float(rng.uniform(0.5, 1.5) * rng.choice([-1, 1])),
        trails=int(rng.integers(3, 7)),
        cap=int(rng.integers(24, 44)),
        title_idx=int(rng.integers(len(TITLES))),
    )


def simulate(seed, p):
    rng = np.random.default_rng(seed)
    R = RING_R
    k = len(p["colors"])
    cols = [bgr(NAMED[c]) for c in p["colors"]]
    ring_col = bgr(p["theme"]["ring"])
    b = Balls()

    def spawn(kind, n):
        for _ in range(n):
            a = rng.uniform(0, 2 * np.pi)
            off = rng.uniform(0, 70, 2) * [np.cos(a), np.sin(a)]
            b.add(CENTER + off, [np.cos(a) * p["speed"], np.sin(a) * p["speed"]], p["rad"], kind)

    per_cap = max(4, p["cap"] // k)
    for c in range(k):
        spawn(c, p["start"])
    phi = rng.uniform(0, 2 * np.pi)
    dt = 1 / SIM_FPS / SUB
    scores = np.zeros(k, int)
    frames, events, score_hist, phis = [], [], [], []
    max_frames = int((MAX_FINALE + TAIL + 0.5) * SIM_FPS)
    for f in range(max_frames):
        for _ in range(SUB):
            b.vel[:, 1] += p["gravity"] * dt
            b.pos += b.vel * dt
            i, j, rv = collide_balls(b, 1.0, 0.8, 150)
            for a_, b_, v in zip(i[:2], j[:2], rv[:2]):
                events.append(Event(f, "hit", *b.pos[a_], cols[b.kind[a_]], min(1, v / 800), int(b.kind[a_]) + 5))
            hit, vn, _ = collide_ring(b, CENTER, R, 1.0, phi, p["gap_half"])
            for h in hit[:2]:
                x, y = wall_point(b.pos[h], b.rad[h])
                events.append(Event(f, "bounce", x, y, cols[b.kind[h]], 0.5, int(b.kind[h]) * 2 + (f // 20) % 3))
            out = escaped(b, CENTER, R)
            if len(out):
                for o in out:
                    c = int(b.kind[o])
                    scores[c] += 1
                    events.append(Event(f, "pop", *b.pos[o], cols[c], 1.0, c * 2 + scores[c] % 5))
                kinds = [int(b.kind[o]) for o in out]
                b.remove(out)
                # Plafond par couleur : au-delà, l'évasion est seulement remplacée (évite l'emballement du leader).
                for c in kinds:
                    spawn(c, 2 if (b.kind == c).sum() < per_cap else 1)
        # Garder une vitesse vivante (sans gravité, l'énergie se conserve déjà ; on corrige la dérive).
        if len(b):
            sp = np.linalg.norm(b.vel, axis=1) + 1e-9
            if p["gravity"] == 0:
                b.vel *= (1 + 0.05 * (p["speed"] / sp - 1))[:, None]
            else:
                low = sp < p["speed"] * 0.4
                b.vel[low] *= (p["speed"] * 0.4 / sp[low])[:, None]
        phi += p["omega"] / SIM_FPS
        score_hist.append(scores.copy())
        phis.append(phi)
        frames.append(snapshot(b, [cols[c] for c in b.kind], [(R, phi, p["gap_half"], ring_col)]))

    hist = np.array(score_hist)
    best = None
    for N in range(6, 41):
        reached = np.nonzero(hist.max(1) >= N)[0]
        if not len(reached):
            break
        fN = int(reached[0])
        t = fN / SIM_FPS
        winners = np.nonzero(hist[fN] >= N)[0]
        if MIN_FINALE <= t <= MAX_FINALE and len(winners) == 1:
            second = np.sort(hist[fN])[-2]
            lead = np.argmax(hist[: fN + 1], 1)
            changes = int((np.diff(lead) != 0).sum())
            cand = dict(N=N, frame=fN, winner=int(winners[0]), margin=int(N - second), changes=changes, t=t)
            q = _quality(cand, N)
            if best is None or q > best[0]:
                best = (q, cand)
    if best is None:
        return None
    c = best[1]
    N, fN, w = c["N"], c["frame"], c["winner"]
    end = fN + int(TAIL * SIM_FPS)
    frames = frames[:end]
    events = strip_events_after(events, fN)
    events.append(Event(fN, "win", *CENTER, cols[w], 1.0, 0))
    events.append(Event(fN, "boom", *CENTER, cols[w], 1.0, 4))
    # Interface : construite après coup, une fois N connu ; scores figés après la victoire.
    xs = np.linspace(520 - 120 * (k - 1), 520 + 120 * (k - 1), k) if k <= 4 else np.linspace(160, 880, k)
    accent = bgr(p["theme"]["accent"])
    for fi, fr in enumerate(frames):
        sc = hist[min(fi, fN)]
        lead = int(np.argmax(sc))
        hud = [] if "{N}" in TITLES[p["title_idx"]] else [("text", f"FIRST TO {N}", 520, 410, 44, accent, "c")]
        for ci in range(k):
            big = ci == lead and sc[ci] > 0
            hud.append(("circle", xs[ci], 490, 26 if big else 20, cols[ci], True))
            hud.append(("text", str(int(sc[ci])), xs[ci], 560, 60 if big else 50, cols[ci], "c"))
        fr.hud = hud
    title = TITLES[p["title_idx"]].format(N=N)
    name = p["colors"][w]
    return Timeline(
        mechanic=NAME, seed=seed, params=p, title=title, frames=frames, events=events,
        finale_frame=fN, finale_text=f"{name} WINS!", finale_color=cols[w],
        end_sub=" - ".join(f"{p['colors'][i][:1]}{p['colors'][i][1:].lower()} {hist[fN][i]}" for i in np.argsort(-hist[fN])[:3]),
        palette=p["theme"], trails=p["trails"],
        metrics=dict(N=N, winner=name, margin=c["margin"], leader_changes=c["changes"], finale_s=round(c["t"], 2),
                     max_balls=int(max(len(f.balls) for f in frames))),
    )


def _quality(c, N):
    close = 1 - min(1, (c["margin"] - 1) / max(1, N * 0.5))
    return 0.4 * close + 0.3 * min(c["changes"], 6) / 6 + 0.3 * duration_score(c["t"])


def score(tl):
    m = tl.metrics
    return _quality(dict(margin=m["margin"], changes=m["leader_changes"], t=m["finale_s"]), m["N"])
