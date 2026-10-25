"""Mécanique 7 : « dernier debout » avec des cordes. Règles strictes, toutes visibles :

- chaque balle est attachée à la paroi par des fils de sa couleur (ancre sur la paroi, autre bout sur la balle) ;
- chaque rebond sur la paroi accroche un nouveau fil au point d'impact ;
- une balle qui touche le fil d'une autre couleur le coupe (tous ceux qu'elle touche) ;
- une balle qui n'a plus aucun fil est éliminée. La dernière qui en garde gagne.
Variante chats : « Last Cat Standing ». Durée : on essaie des sous-graines jusqu'à une fin dans la fenêtre.
"""
import numpy as np

from ..palettes import NAMED, bgr, pick_theme
from ..physics import Balls, collide_balls, collide_ring, rotate
from ..scene import RING_R, Timeline
from .common import CENTER, SIM_FPS, TAIL, Event, snapshot

NAME = "strings"
SUB = 6
# Avec des fils attachés aux balles et des règles strictes, une partie dure rarement plus de 30 s :
# fenêtre plus courte que les autres mécaniques (vidéo de 18 à 36 s environ).
MIN_FIN, MAX_FIN = 12.0, 30.0

TITLES = {
    None: ["Last ball standing wins", "Who keeps their strings?", "Cut the strings! Who survives?",
           "Pick a color. Who keeps a string?"],
    "cats": ["Last Cat Standing", "Which cat keeps its strings?", "Cut the strings! Which cat survives?",
             "Pick a cat. Who survives?"],
}
DESCRIPTIONS = ["Every bounce ties a string. Touch a string and it snaps. Last one with strings wins.",
                "Strings, bounces and a lot of snapping. Who survives?",
                "No strings left = out. Pick your winner before it starts!"]
HASHTAGS = ["#shorts", "#satisfying", "#simulation", "#lastonestanding", "#oddlysatisfying"]
OVERLAYS = ["", "", "", "Pick one!"]
CAT_PICK = [2, 3, 1, 5, 7, 8]        # chats bien différents (roux, noir, blanc, siamois, tigré, cool)


def sample(rng):
    k = int(rng.choice([2, 3, 3]))
    cats = rng.random() < 0.55
    names = [str(x) for x in rng.choice(["RED", "BLUE", "YELLOW", "GREEN", "PURPLE", "ORANGE", "PINK", "CYAN"], k, replace=False)]
    p = dict(
        theme=pick_theme(rng),
        root=int(rng.integers(50, 62)),
        k=k,
        names=names,
        r0=float(rng.uniform(24, 30) if cats else rng.uniform(12, 20)),
        gravity=float(rng.uniform(800, 1000)),
        speed=float(rng.uniform(450, 560)),
        jitter=float(np.radians(rng.uniform(3, 9))),
        start=int(rng.integers(3, 6)),
        min_fin=MIN_FIN, max_fin=MAX_FIN,
        title_idx=int(rng.integers(4)),
        overlay_idx=int(rng.integers(len(OVERLAYS))),
        trails=int(rng.integers(4, 8)),
    )
    if cats:
        p["objects"] = "cats"
        p["cat_levels"] = [int(c) for c in rng.choice(CAT_PICK, k, replace=False)]
    return p


def _run(seed, p):
    rng = np.random.default_rng(seed)
    R = RING_R
    k = p["k"]
    cols = [bgr(NAMED[n]) for n in p["names"]]
    b = Balls()
    for i in range(k):
        a = 2 * np.pi * i / k + rng.uniform(-0.3, 0.3)
        v = rng.uniform(0, 2 * np.pi)
        b.add(CENTER + 0.45 * R * np.array([np.cos(a), np.sin(a)]), [np.cos(v) * p["speed"], np.sin(v) * p["speed"]],
              p["r0"], p["cat_levels"][i] if "cat_levels" in p else i)
    player = {int(b.uid[i]): i for i in range(k)}
    uid_of = {i: u for u, i in player.items()}
    # Fils attachés aux balles : une ancre sur la paroi, l'autre bout suit la balle de son propriétaire.
    A, O = [], []
    for i in range(k):
        d0 = b.pos[i] - CENTER
        a0 = np.arctan2(d0[1], d0[0])
        for a in a0 + np.linspace(-0.5, 0.5, p["start"]):
            A.append(CENTER + R * np.array([np.cos(a), np.sin(a)]))
            O.append(i)
    A, O = np.array(A), np.array(O)
    alive = [True] * k
    dt = 1 / SIM_FPS / SUB
    frames, events, history = [], [], []
    fin, winner = None, None
    max_frames = int((p["max_fin"] + 2) * SIM_FPS)
    for f in range(max_frames):
        if fin is None:
            for _ in range(SUB):
                b.vel[:, 1] += p["gravity"] * dt
                b.pos += b.vel * dt
                collide_balls(b, 1.0, 0.8)
                hit, vn, _ = collide_ring(b, CENTER, R, 1.0)
                for h in hit:
                    i = player[int(b.uid[h])]
                    b.vel[h] = rotate(b.vel[h:h + 1], rng.uniform(-p["jitter"], p["jitter"]))[0]
                    s = np.linalg.norm(b.vel[h])
                    if s < p["speed"] * 0.8:
                        b.vel[h] *= p["speed"] * 0.8 / (s + 1e-9)
                    d = b.pos[h] - CENTER
                    q = CENTER + d / (np.linalg.norm(d) + 1e-9) * R
                    # Règle stricte : chaque rebond sur la paroi accroche un fil au point d'impact.
                    A, O = np.vstack([A, q]), np.append(O, i)
                    events.append(Event(f, "bounce", *q, cols[i], 0.7, i * 2 + int((O == i).sum()) % 3))
                # Règle stricte : une balle qui touche le fil (visible) d'une autre couleur le coupe.
                if len(A):
                    idx = {player[int(u)]: j for j, u in enumerate(b.uid)}
                    E = np.array([b.pos[idx[o]] for o in O])
                    own_r = np.array([b.rad[idx[o]] for o in O])
                    seg = E - A
                    L2 = (seg ** 2).sum(-1) + 1e-9
                    cut = np.zeros(len(A), bool)
                    for i, j in idx.items():
                        tt = np.clip(((b.pos[j] - A) * seg).sum(-1) / L2, 0, 1)
                        dd = np.linalg.norm(A + seg * tt[:, None] - b.pos[j], axis=1)
                        # (la partie cachée sous la balle propriétaire ne se voit pas, donc ne compte pas)
                        c = (O != i) & (dd < b.rad[j] + 1.5) & (np.sqrt(L2) * (1 - tt) > own_r) & ~cut
                        for s_ in np.nonzero(c)[0]:
                            px, py = A[s_] + seg[s_] * tt[s_]
                            events.append(Event(f, "shard", px, py, cols[O[s_]], 1, 0))
                            events.append(Event(f, "pop", px, py, cols[O[s_]], 0.6, O[s_] * 2 + 1))
                        cut |= c
                    if cut.any():
                        A, O = A[~cut], O[~cut]
                        for i in range(k):
                            if alive[i] and not (O == i).any():
                                alive[i] = False
                                j = [player[int(u)] for u in b.uid].index(i)
                                events.append(Event(f, "boom", *b.pos[j], cols[i], 0.8, 1))
                                b.remove([j])
                        if sum(alive) == 1:
                            winner = alive.index(True)
                            fin = f
                            events.append(Event(f, "win", *CENTER, cols[winner], 1.0, 0))
                            break
                        if sum(alive) == 0:
                            return frames, events, None, None, history
        history.append([int((O == i).sum()) for i in range(k)])
        fr = snapshot(b, [cols[player[int(u)]] for u in b.uid], [(R, None, 0, bgr(p["theme"]["ring"]))])
        fr.strings = (A.astype(np.float32).copy(), np.array([uid_of[o] for o in O], int),
                      np.array([cols[o] for o in O], np.uint8).reshape(-1, 3))
        frames.append(fr)
        if fin is not None and f >= fin + int(TAIL * SIM_FPS):
            break
    return frames, events, fin, winner, history


def simulate(seed, p):
    for tr in range(25):
        frames, events, fin, winner, hist = _run(seed + 7919 * tr, p)
        t = fin / SIM_FPS if fin is not None else None
        if t is not None and p["min_fin"] <= t <= p["max_fin"]:
            break
    else:
        return None
    p["sub_seed"] = seed + 7919 * tr
    hist = np.array(hist)
    leaders = [int(np.argmax(h)) for h in hist[:: SIM_FPS // 2] if h.max() > 0]
    changes = sum(1 for a, c in zip(leaders, leaders[1:]) if a != c)
    # Dernier duel : combien de temps le perdant a tenu avec 1 seul fil (suspense) avant la fin.
    if "cat_levels" in p:
        from ..objects import CATS
        name = CATS[p["cat_levels"][winner]]
    else:
        name = p["names"][winner]
    title = TITLES[p.get("objects")][p["title_idx"]]
    return Timeline(
        mechanic=NAME, seed=seed, params=p, title=title, frames=frames, events=events,
        finale_frame=fin, finale_text=f"{p['names'][winner]} WINS!", finale_color=bgr(NAMED[p["names"][winner]]), end_sub="",
        overlay=OVERLAYS[p["overlay_idx"]], palette=p["theme"], trails=p["trails"],
        metrics=dict(finale_s=round(fin / SIM_FPS, 2), winner=name, leader_changes=changes,
                     peak=int(hist.max()), final=int(hist[-1].max())),
    )


def score(tl):
    m = tl.metrics
    s = 0.55 * max(0.0, 1 - abs(m["finale_s"] - 24) / 14)   # fenêtre 12-30 s : on vise environ 24 s
    s += 0.3 * min(1.0, m["leader_changes"] / 4)
    s += 0.15 * (1.0 if m["final"] <= 3 else 0.5)     # le gagnant finit sur peu de fils = fin serrée
    return s


def comment(tl, rng):
    cats = tl.params.get("objects") == "cats"
    opts = ["Who did you pick before it started? 👇", "That last cut was so close... who were you rooting for? 👀",
            "Team " + tl.finale_text.split()[0].title() + " or not? 👇"]
    if cats:
        opts.append("Which cat should fight next time? 🐱👇")
    return str(rng.choice(opts))
