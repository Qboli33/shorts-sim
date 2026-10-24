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
TARGETS = [200, 250, 300, 400, 500, 600, 750, 800, 1000, 1200, 1500, 2000]
NOIR = {"name": "noir", "bg": ((8, 8, 10), (0, 0, 0)), "ring": (215, 215, 225), "accent": (255, 255, 255), "flat": True}

TITLES = ["Every crush makes 4 more balls. {N}?", "Can the press make {N} balls?", "1 crush = 4 balls. How long to {N}?",
          "Hydraulic press vs balls: {N} incoming?"]
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
        fill=float(rng.uniform(0.3, 0.42)),       # part du cercle occupée par les balles
        r0=float(rng.uniform(32, 42)),
        levels=3,                                  # tailles : r0, r0/2, r0/4 (la plus petite ne casse plus)
        crush=float(rng.uniform(6.0, 10.0)),        # pression (somme des recouvrements, px) qui brise une balle
        t_full=float(rng.uniform(44, 56)),        # instant prévu où tout serait broyé
        curve=float(rng.uniform(1.4, 2.0)),       # accélération de la montée du compteur
        speed=float(rng.uniform(300, 460)),       # descente du plateau (px/s)
        gravity=1600.0,
        title_idx=int(rng.integers(len(TITLES))),
        overlay_idx=int(rng.integers(len(OVERLAYS))),
        sfx="crunch",
    )


def simulate(seed, p):
    rng = np.random.default_rng(seed)
    R = RING_R
    cx, cy = CENTER
    cols = [bgr(NAMED[c]) for c in p["colors"]]
    rmin = p["r0"] / 2 ** (p["levels"] - 1)
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
    # Automate de la presse : descente jusqu'à `budget` balles écrasées (ou blocage), pause, remontée, attente.
    # Le budget suit un plan : écrasements cumulés = total x (t / t_full) ^ curve (le compteur accélère).
    c_full = n0 * (4 ** (p["levels"] - 1) - 1) / 3
    total = 0
    # Fond de course : la zone sous le plateau garde de quoi loger toutes les balles (surface conservée).
    need = n0 * np.pi * p["r0"] ** 2 / 0.9
    hs = np.linspace(0, 2 * R, 400)
    seg = R ** 2 * np.arccos(1 - hs / R) - (R - hs) * np.sqrt(np.maximum(0, 2 * R * hs - hs ** 2))
    bottom = cy + R - hs[np.searchsorted(seg, need)]
    state, t_state, budget, n_crushed = "wait", 0.0, 1, 0
    frames, events, counts = [], [], []
    py = top
    max_frames = int((MAX_FINALE + TAIL + 0.5) * SIM_FPS)
    for f in range(max_frames):
        t = f / SIM_FPS
        t_state += 1 / SIM_FPS
        if state == "wait" and t_state > 0.55:
            state, t_state, n_crushed = "down", 0.0, 0
            budget = max(1, int(c_full * min(1, (t + 1.8) / p["t_full"]) ** p["curve"]) - total)
            events.append(Event(f, "hiss", cx, top, (200, 200, 200), 0.8, 0))
        if state == "down" and (n_crushed >= budget or py >= bottom - 0.5):
            state, t_state = "hold", 0.0
        if state == "hold" and t_state > 0.35:
            state, t_state = "up", 0.0
        if state == "up" and py <= top + 1:
            state, t_state = "wait", 0.0
        target = {"down": min(bottom, py + p["speed"] / SIM_FPS), "hold": py,
                  "up": max(top, py - 900 / SIM_FPS), "wait": top}[state]
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
            hitp = pen > 0
            if hitp.any():
                b.pos[hitp, 1] += pen[hitp]
                vp = (py - py_prev) / dt
                b.vel[hitp, 1] = np.maximum(b.vel[hitp, 1], vp)
                over[hitp] += pen[hitp]
            # Écrasement : recouvrement trop fort (balle coincée), hors balles qui viennent de naître.
            # Les balles sous le plateau cassent en premier ; plus bas, il faut une pression double (le plateau doit appuyer).
            thr = np.minimum(p["crush"], 0.45 * b.rad)
            under = (b.pos[:, 1] - b.rad) < py + 3
            crushed = np.nonzero(((under & (over > thr)) | ((over > 2 * thr) & (py > cy - R))) & (b.rad > rmin * 1.5) & (t - born > 0.25))[0]
            if state == "down" and n_crushed < budget and len(crushed):
                crushed = crushed[np.argsort(-over[crushed])][: min(40, int(np.ceil(budget - n_crushed)))]
                n_crushed += len(crushed)
                total += len(crushed)
                new = []
                for c in crushed:
                    x, y, r, kd = b.pos[c, 0], b.pos[c, 1], b.rad[c], int(b.kind[c])
                    for ang in rng.uniform(0, 2 * np.pi) + np.arange(4) * np.pi / 2:
                        new.append(([x + np.cos(ang) * r * 0.42, y + np.sin(ang) * r * 0.42],
                                    [np.cos(ang) * 120, np.sin(ang) * 120], r / 2, kd))
                    events.append(Event(f, "crunch", x, y, cols[kd], min(1, r / p["r0"] + 0.3), int(np.log2(p["r0"] / r))))
                b.remove(crushed)
                born = np.delete(born, crushed)
                for q, v, r, kd in new:
                    b.add(q, v, r, kd)
                born = np.append(born, np.full(len(new), t))
        counts.append(len(b))
        fr = snapshot(b, [cols[c] for c in b.kind], [(R, None, 0, bgr(NOIR["ring"]))])
        fr.press = float(py)
        frames.append(fr)

    counts = np.array(counts)
    simulate.counts = counts     # pour le réglage
    best = None
    for N in TARGETS:
        hit = np.nonzero(counts >= N)[0]
        if not len(hit):
            continue
        t = hit[0] / SIM_FPS
        if MIN_FINALE <= t <= MAX_FINALE:
            q = duration_score(t)
            if best is None or q > best[0] or (q > best[0] - 0.1 and N > best[1]):
                best = (q, N, int(hit[0]))
    if best is None:
        return None
    _, N, fin = best
    frames = frames[: fin + int(TAIL * SIM_FPS)]
    events = strip_events_after(events, fin)
    white = (255, 255, 255)
    events.append(Event(fin, "boom", cx, cy, white, 1.0, 2))
    events.append(Event(fin, "win", cx, cy, white, 1.0, 0))
    for fi, fr in enumerate(frames):
        c = int(counts[min(fi, fin)])
        fr.hud = [("text", f"{c}/{N}", 520, 420, 96, white, "c")]
    return Timeline(
        mechanic=NAME, seed=seed, params=p, title=TITLES[p["title_idx"]].format(N=N), frames=frames, events=events,
        finale_frame=fin, finale_text=f"{N}!", finale_color=white, end_sub="", overlay=OVERLAYS[p["overlay_idx"]],
        palette=p["theme"], trails=0,
        metrics=dict(N=N, finale_s=round(fin / SIM_FPS, 2), peak=int(counts.max()), n0=n0),
    )


def score(tl):
    m = tl.metrics
    return 0.6 * duration_score(m["finale_s"]) + 0.4 * min(1, m["N"] / 1000)


def comment(tl, rng):
    n = tl.metrics["N"]
    opts = ["What should the press crush next? 👇", f"Guess how long to {n * 2} balls! ⏱️👇",
            "That crunch sound though... 🔊 Rate it 1 to 10 👇"]
    return str(rng.choice(opts))
