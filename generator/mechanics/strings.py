"""Mécanique 7 : « dernier debout ». Chaque balle tend des fils vers la paroi ; les autres balles les coupent.

À chaque rebond sur la paroi, une balle peut accrocher un nouveau fil (ancré au point d'impact, tiré par la balle).
Une autre balle qui traverse un fil le coupe. Une balle qui n'a plus aucun fil est éliminée (pop).
La dernière qui garde des fils gagne. Variante chats : « Last Cat Standing » avec les chats dessinés par code.
Durée réglée par la probabilité d'accrocher un fil (réajustée entre deux essais).
"""
import numpy as np

from ..palettes import NAMED, bgr, pick_theme
from ..physics import Balls, collide_balls, collide_ring, rotate
from ..scene import RING_R, Timeline
from .common import CENTER, MAX_FINALE, MIN_FINALE, SIM_FPS, TAIL, Event, duration_score, snapshot, wall_point

NAME = "strings"
SUB = 6
MAX_STRINGS = 30
GRACE = 4.0          # pas de coupe au début : chacun tend d'abord ses fils

TITLES = {
    None: ["Last ball standing wins", "Who keeps their strings?", "Cut the strings! Who survives?",
           "Pick a color. Who keeps a string?"],
    "cats": ["Last Cat Standing", "Which cat keeps its strings?", "Cut the strings! Which cat survives?",
             "Pick a cat. Who survives?"],
}
DESCRIPTIONS = ["Every bounce ties a string. Cross a string and it snaps. Last one with strings wins.",
                "Strings, bounces and a lot of snapping. Who survives?",
                "No strings left = out. Pick your winner before it starts!"]
HASHTAGS = ["#shorts", "#satisfying", "#simulation", "#lastonestanding", "#oddlysatisfying"]
OVERLAYS = ["", "", "", "Pick one!"]
CAT_PICK = [2, 3, 1, 5, 7, 8]        # chats bien différents (roux, noir, blanc, siamois, tigré, cool)


def sample(rng):
    k = int(rng.choice([2, 3, 3, 4]))
    cats = rng.random() < 0.55
    names = [str(x) for x in rng.choice(["RED", "BLUE", "YELLOW", "GREEN", "PURPLE", "ORANGE", "PINK", "CYAN"], k, replace=False)]
    p = dict(
        theme=pick_theme(rng),
        root=int(rng.integers(50, 62)),
        k=k,
        names=names,
        r0=float(rng.uniform(38, 48) if cats else rng.uniform(24, 34)),
        gravity=float(rng.uniform(700, 1300)),
        speed=float(rng.uniform(600, 900)),
        jitter=float(np.radians(rng.uniform(3, 9))),
        add_p=float(rng.uniform(0.3, 0.6)),       # probabilité qu'un rebond accroche un fil
        cool=float(rng.uniform(0.3, 0.6)),        # repos d'une balle entre deux coupes (s)
        start=int(rng.integers(3, 6)),
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
    owner_uid = {int(b.uid[i]): i for i in range(k)}
    # Fils : ancre sur la paroi + indice du joueur qui le tient.
    anchors, owners = [], []
    for i in range(k):
        for _ in range(p["start"]):
            a = rng.uniform(0, 2 * np.pi)
            anchors.append(CENTER + R * np.array([np.cos(a), np.sin(a)]))
            owners.append(i)
    anchors, owners = np.array(anchors), np.array(owners)
    alive = [True] * k
    cooldown = [-10 ** 6] * k
    dt = 1 / SIM_FPS / SUB
    frames, events, history = [], [], []
    fin, winner = None, None
    max_frames = int((MAX_FINALE + 2) * SIM_FPS)
    for f in range(max_frames):
        if fin is None:
            for _ in range(SUB):
                b.vel[:, 1] += p["gravity"] * dt
                b.pos += b.vel * dt
                collide_balls(b, 1.0, 0.8)
                hit, vn, _ = collide_ring(b, CENTER, R, 1.0)
                for h in hit:
                    i = owner_uid[int(b.uid[h])]
                    b.vel[h] = rotate(b.vel[h:h + 1], rng.uniform(-p["jitter"], p["jitter"]))[0]
                    s = np.linalg.norm(b.vel[h])
                    if s < p["speed"] * 0.8:
                        b.vel[h] *= p["speed"] * 0.8 / (s + 1e-9)
                    x, y = wall_point(b.pos[h], b.rad[h])
                    n_own = int((owners == i).sum())
                    if n_own < MAX_STRINGS and rng.random() < p["add_p"]:
                        anchors = np.vstack([anchors, [x, y]])
                        owners = np.append(owners, i)
                        events.append(Event(f, "bounce", x, y, cols[i], 0.7, i * 2 + n_own % 3))
                    else:
                        events.append(Event(f, "hit", x, y, cols[i], 0.4, i * 2))
                # Coupe : une autre balle touche le segment ancre -> balle propriétaire.
                if len(anchors):
                    idx_of = {owner_uid[int(u)]: j for j, u in enumerate(b.uid)}
                    ends = np.array([b.pos[idx_of[o]] for o in owners])
                    seg = ends - anchors
                    L2 = (seg ** 2).sum(-1) + 1e-9
                    cut = np.zeros(len(anchors), bool)
                    for i, j in idx_of.items():
                        q = b.pos[j]
                        tt = np.clip(((q - anchors) * seg).sum(-1) / L2, 0, 1)
                        d = np.linalg.norm(anchors + seg * tt[:, None] - q, axis=1)
                        c = (owners != i) & (d < 0.6 * b.rad[j]) & (tt < 0.8) & (np.sqrt(L2) * (1 - tt) > b.rad[j] + 2.2 * p["r0"]) & ~cut
                        # Une balle coupe au plus un fil à la fois, puis souffle `cool` secondes.
                        if not c.any() or f - cooldown[i] < p["cool"] * SIM_FPS or f < GRACE * SIM_FPS:
                            continue
                        cooldown[i] = f
                        first = np.nonzero(c)[0][np.argmin(d[c])]
                        c = np.zeros_like(c)
                        c[first] = True
                        for s_ in [first]:
                            px, py = anchors[s_] + seg[s_] * tt[s_]
                            events.append(Event(f, "shard", px, py, cols[owners[s_]], 1, 0))
                            events.append(Event(f, "pop", px, py, cols[owners[s_]], 0.6, owners[s_] * 2 + 1))
                        cut |= c
                    if cut.any():
                        anchors, owners = anchors[~cut], owners[~cut]
                        for i in range(k):
                            if alive[i] and not (owners == i).any():
                                alive[i] = False
                                j = idx_of[i]
                                events.append(Event(f, "boom", *b.pos[j], cols[i], 0.8, 1))
                                b.remove([j])
                                idx_of = {owner_uid[int(u)]: jj for jj, u in enumerate(b.uid)}
                        if sum(alive) == 1:
                            winner = alive.index(True)
                            fin = f
                            events.append(Event(f, "win", *CENTER, cols[winner], 1.0, 0))
                            break
                        if sum(alive) == 0:
                            return frames, events, None, None, history
        history.append([int((owners == i).sum()) for i in range(k)])
        fr = snapshot(b, [cols[owner_uid[int(u)]] for u in b.uid], [(R, None, 0, bgr(p["theme"]["ring"]))])
        uid_of = {i: u for u, i in owner_uid.items()}
        fr.strings = (anchors.astype(np.float32).copy(), np.array([uid_of[o] for o in owners], int),
                      np.array([cols[o] for o in owners], np.uint8).reshape(-1, 3))
        frames.append(fr)
        if fin is not None and f >= fin + int(TAIL * SIM_FPS):
            break
    return frames, events, fin, winner, history


def simulate(seed, p):
    for tr in range(12):
        frames, events, fin, winner, hist = _run(seed + 7919 * tr, p)
        t = fin / SIM_FPS if fin is not None else None
        if t is not None and MIN_FINALE <= t <= MAX_FINALE:
            break
        # Fin trop tôt : plus de fils ; jamais de fin : moins de fils.
        if t is None and winner is None and len(frames) >= int((MAX_FINALE + 1) * SIM_FPS):
            p["add_p"] = max(0.08, p["add_p"] * 0.8)
        elif t is not None and t < MIN_FINALE:
            p["add_p"] = min(1.0, p["add_p"] * 1.15)
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
    s = 0.55 * duration_score(m["finale_s"])
    s += 0.3 * min(1.0, m["leader_changes"] / 4)
    s += 0.15 * (1.0 if m["final"] <= 3 else 0.5)     # le gagnant finit sur peu de fils = fin serrée
    return s


def comment(tl, rng):
    cats = tl.params.get("objects") == "cats"
    opts = ["Who did you pick before it started? 👇", "That last cut was so close... who were you rooting for? 👀",
            "Team " + tl.metrics["winner"].title() + " or not? 👇"]
    if cats:
        opts.append("Which cat should fight next time? 🐱👇")
    return str(rng.choice(opts))
