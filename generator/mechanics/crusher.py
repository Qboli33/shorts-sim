"""Mécanique 5 : presse hydraulique. Chaque balle écrasée se brise en 4 balles deux fois plus petites.


Un plateau descend par à-coups dans le cercle, chaque coup un peu plus bas. Une balle trop comprimée
(somme des recouvrements > `crush` px) se brise en 4 balles de rayon r/2 (même surface) ; la plus petite taille
ne casse plus et fait caler la presse. Le compteur de balles ne fait que monter.
Objectif : un nombre rond de balles, choisi après la simulation (fin dans la fenêtre de durée).
Style « noir » : fond noir, anneau fin, pas de halo, beaucoup de balles brillantes.
"""
import numpy as np

from ..palettes import NAMED, bgr
from ..physics import Balls, collide_pairs, collide_ring, near_pairs
from ..scene import RING_R, Timeline
from .common import CENTER, MAX_FINALE, MIN_FINALE, SIM_FPS, TAIL, Event, duration_score, snapshot, strip_events_after

NAME = "crusher"
SUB = 4
SPLIT = 0.42     # rayon d'un éclat / rayon de la balle écrasée
NOIR = {"name": "noir", "bg": ((8, 8, 10), (0, 0, 0)), "ring": (215, 215, 225), "accent": (255, 255, 255), "flat": True}

TITLES = ["Every crush makes 4 more balls", "Can the press go all the way down?", "Hydraulic press vs {n0} balls",
          "1 crush = 4 balls. How deep can it go?"]
DESCRIPTIONS = ["Every ball the press crushes splits into 4 smaller ones.", "Crush one ball, get four. Watch the count explode.",
                "A hydraulic press, a lot of balls, and very satisfying crunches."]
HASHTAGS = ["#shorts", "#satisfying", "#hydraulicpress", "#oddlysatisfying", "#asmr"]
OVERLAYS = ["", "", "", "1 crush = 4 balls"]
COLORSETS = [["RED", "BLUE", "YELLOW", "GREEN"], ["PINK", "CYAN", "ORANGE"], ["ORANGE", "BLUE", "LIME", "PURPLE"],
             ["RED", "YELLOW"], ["PURPLE", "PINK", "CYAN", "YELLOW", "GREEN"]]


def sample(rng):
    return dict(
        theme=dict(NOIR),
        root=int(rng.integers(50, 60)),
        colors=COLORSETS[int(rng.integers(len(COLORSETS)))],
        fill=float(rng.uniform(0.32, 0.45)),       # part du cercle occupée par les balles
        r0=float(rng.uniform(32, 42)),
        levels=3,                                  # tailles : r0, r0/2, r0/4 (la plus petite ne casse plus)
        crush=float(rng.uniform(6.0, 10.0)),        # pression (somme des recouvrements, px) qui brise une balle
        t_full=float(rng.uniform(32, 46)),        # instant prévu du coup le plus profond
        curve=float(rng.uniform(0.9, 1.3)),       # accélération de la montée du compteur
        speed=float(rng.uniform(220, 360)),       # descente du plateau (px/s)
        wait=float(rng.uniform(0.25, 0.45)),      # pause plateau levé
        lift=float(rng.uniform(140, 260)),        # remontée entre deux coups (px)
        gravity=1600.0,
        title_idx=int(rng.integers(len(TITLES))),
        overlay_idx=int(rng.integers(len(OVERLAYS))),
        sfx="soft",
    )


def simulate(seed, p):
    rng = np.random.default_rng(seed)
    R = RING_R
    cx, cy = CENTER
    cols = [bgr(NAMED[c]) for c in p["colors"]]
    rmin = p["r0"] * SPLIT ** (p["levels"] - 1)
    b = Balls()
    n0 = int(p["fill"] * R ** 2 / p["r0"] ** 2)
    # Remplissage initial : balles posées au hasard dans le cercle, sans chevauchement (elles tombent ensuite).
    tries = 0
    while len(b) < n0 and tries < 20000:
        tries += 1
        a, d = rng.uniform(0, 2 * np.pi), np.sqrt(rng.uniform(0, 1)) * (R - p["r0"] - 4)
        q = np.array([cx + np.cos(a) * d, cy + np.sin(a) * d])
        if len(b) and np.min(np.hypot(*(b.pos - q).T) - b.rad) < p["r0"] + 2:
            continue
        b.add(q, rng.uniform(-80, 80, 2), p["r0"], int(rng.integers(len(cols))))
    born = np.zeros(len(b))
    dt = 1 / SIM_FPS / SUB
    top = cy - R - 10                    # plateau rangé hors du cercle
    # Hauteur d'une pile de surface S au fond du cercle (segment circulaire).
    hs = np.linspace(0, 2 * R, 400)
    seg = R ** 2 * np.arccos(1 - hs / R) - (R - hs) * np.sqrt(np.maximum(0, 2 * R * hs - hs ** 2))
    area = n0 * np.pi * p["r0"] ** 2
    pile = cy + R - hs[np.searchsorted(seg, area / 0.78)]
    # Chaque écrasement fait perdre de la surface (4 x (0,42 r)² = 71 %) : la pile se tasse et la presse descend.
    bottom = cy + R - 3 * rmin     # la plus petite taille part en poussière : la presse va jusqu'au fond
    # Seul le plateau écrase : chaque coup descend un peu plus bas (profondeur prévue = courbe vers `bottom`).
    def depth(t):
        return pile + 10 + (bottom - pile - 10) * min(1.0, t / p["t_full"]) ** p["curve"]
    state, t_state, goal, last = "wait", 0.0, depth(0), False
    squeeze = np.zeros(len(b))           # enfoncement cumulé par le plateau (balle soutenue par-dessous)
    frames, events, counts = [], [], []
    py = top
    fin = None
    max_frames = int((MAX_FINALE + TAIL + 0.5) * SIM_FPS)
    for f in range(max_frames):
        t = f / SIM_FPS
        t_state += 1 / SIM_FPS
        if state == "wait" and t_state > p["wait"]:
            state, t_state, goal = "down", 0.0, depth(t + 1.0)
            last = t + 1.0 >= p["t_full"]
            events.append(Event(f, "hiss", cx, py, (200, 200, 200), 0.8, 0))
        if state == "down" and py >= goal - 0.5:
            state, t_state = "hold", 0.0
            if last and fin is None and t >= MIN_FINALE:
                fin = f
        if state == "hold" and t_state > 0.3:
            state, t_state, up_to = "up", 0.0, max(top, py - p["lift"])
        if state == "up" and py <= up_to + 1:
            state, t_state = "wait", 0.0
        if fin is not None:
            target = py
        else:
            target = {"down": min(goal, py + p["speed"] / SIM_FPS), "hold": py,
                      "up": max(up_to if state == "up" else top, py - 900 / SIM_FPS), "wait": py}[state]
        py0 = py
        for s in range(SUB):
            py_prev = py
            py = py0 + (target - py0) * (s + 1) / SUB
            b.vel[:, 1] += p["gravity"] * dt
            b.vel *= 0.999
            b.pos += b.vel * dt
            i, j = near_pairs(b)
            over = np.zeros(len(b))
            for _ in range(2):
                over = np.maximum(over, collide_pairs(b, i, j, 0.15, 0.7))
            collide_ring(b, CENTER, R, 0.2)
            # Plateau : tout ce qui dépasse au-dessus de py est repoussé vers le bas (avec la vitesse du plateau).
            pen = py - (b.pos[:, 1] - b.rad)
            hitp = pen > -1.5
            if (pen > 0).any():
                m = pen > 0
                b.pos[m, 1] += pen[m]
                b.vel[m, 1] = np.maximum(b.vel[m, 1], (py - py_prev) / dt)
                sup = m & (over > 0.4)
                squeeze[sup] += pen[sup]
            squeeze[~hitp] *= 0.8
            if state != "down":
                continue
            # Écrasement : uniquement les balles coincées contre le plateau.
            thr = np.minimum(p["crush"], 0.45 * b.rad)
            crushed = np.nonzero(hitp & (squeeze > thr) & (t - born > 0.2))[0]
            if len(crushed):
                crushed = crushed[np.argsort(-squeeze[crushed])][:12]
                new = []
                for c in crushed:
                    x, y, r, kd = b.pos[c, 0], b.pos[c, 1], b.rad[c], int(b.kind[c])
                    if r <= rmin * 1.5:      # plus petite taille : poussière
                        events.append(Event(f, "crunch", x, y, cols[kd], 0.35, p["levels"]))
                        continue
                    for ang in rng.uniform(0, 2 * np.pi) + np.arange(4) * np.pi / 2:
                        new.append(([x + np.cos(ang) * r * 0.42, y + max(0.0, np.sin(ang)) * r * 0.42 + r * 0.1],
                                    [np.cos(ang) * 140, abs(np.sin(ang)) * 140], r * SPLIT, kd))
                    events.append(Event(f, "crunch", x, y, cols[kd], min(1, r / p["r0"] + 0.3), int(round(np.log(p["r0"] / r) / np.log(1 / SPLIT)))))
                b.remove(crushed)
                born = np.delete(born, crushed)
                squeeze = np.delete(squeeze, crushed)
                for q, v, r, kd in new:
                    b.add(q, v, r, kd)
                born = np.append(born, np.full(len(new), t))
                squeeze = np.append(squeeze, np.zeros(len(new)))
        counts.append(len(b))
        fr = snapshot(b, [cols[c] for c in b.kind], [(R, None, 0, bgr(NOIR["ring"]))])
        fr.press = float(py)
        frames.append(fr)
        if fin is not None and f >= fin + int(TAIL * SIM_FPS):
            break

    counts = np.array(counts)
    simulate.counts = counts     # pour le réglage
    simulate.fin = fin
    simulate.levels = (pile, bottom)
    if fin is None or not (MIN_FINALE <= fin / SIM_FPS <= MAX_FINALE):
        return None
    events = strip_events_after(events, fin)
    white = (255, 255, 255)
    events.append(Event(fin, "boom", cx, cy, white, 1.0, 2))
    events.append(Event(fin, "win", cx, cy, white, 1.0, 0))
    N = int(counts[fin])
    return Timeline(
        mechanic=NAME, seed=seed, params=p, title=TITLES[p["title_idx"]].format(n0=n0), frames=frames, events=events,
        finale_frame=fin, finale_text="FLAT!", finale_color=white, end_sub="", overlay=OVERLAYS[p["overlay_idx"]],
        palette=p["theme"], trails=0,
        metrics=dict(N=N, finale_s=round(fin / SIM_FPS, 2), n0=n0, crushed=int((N - n0) / 3)),
    )


def score(tl):
    m = tl.metrics
    return 0.6 * duration_score(m["finale_s"]) + 0.4 * min(1, m["N"] / 1000)


def comment(tl, rng):
    n = tl.metrics["N"]
    opts = ["What should the press crush next? 👇", f"Guess how long to {n * 2} balls! ⏱️👇",
            "That crunch sound though... 🔊 Rate it 1 to 10 👇"]
    return str(rng.choice(opts))
