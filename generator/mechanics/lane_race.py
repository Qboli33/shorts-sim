"""Mécanique 8 : course de deux couloirs, « multiplier ou ajouter ? ».

Deux couloirs côte à côte, chacun fermé par une pile de barrières aux points de vie croissants (×1,3 à ×1,5).
Chaque balle qui frappe la barrière active lui retire 1 point. Barrière à 0 : elle éclate et la règle du couloir
s'applique (×m : chaque balle se dédouble en m balles ; +a : a balles de plus). Le premier couloir qui casse
sa dernière barrière gagne. Le couloir ×m part d'une seule balle (l'outsider), le couloir +a de plusieurs.
Équilibre : le nombre de balles de départ du couloir +a est choisi par un petit modèle (taux de frappe par balle,
saturation quand la pile s'épaissit) pour une arrivée serrée ; la simulation tranche.
Durée : points de vie mis à l'échelle entre essais pour une fin dans la fenêtre.
"""
import numpy as np

from ..palettes import NAMED, bgr, pick_theme
from ..physics import Balls, collide_pairs, near_pairs
from ..scene import Frame, Timeline
from .common import MAX_FINALE, MIN_FINALE, SIM_FPS, TAIL, Event, duration_score

NAME = "lane_race"
SUB = 3
LANES = ((70.0, 490.0), (530.0, 950.0))     # le bord droit de l'écran est masqué par l'interface
Y_TOP, Y_FIRST, Y_LAST, Y_FLOOR = 470.0, 640.0, 1480.0, 1545.0
PAIRS = [("PURPLE", "LIME"), ("PINK", "CYAN"), ("ORANGE", "BLUE"), ("RED", "GREEN"), ("YELLOW", "PURPLE"), ("CYAN", "ORANGE")]

TITLES = ["{rm} or {ra}: which lane wins?", "1 ball {rm} vs {s} balls {ra}", "Multiply or add? Pick a lane!",
          "Can 1 ball beat {s}?"]
DESCRIPTIONS = ["One lane multiplies, the other adds. Every hit chips a barrier.",
                "Starting with 1 ball vs {s} balls. Multiply or add?",
                "Break every barrier first to win. Which rule is stronger?"]
HASHTAGS = ["#shorts", "#satisfying", "#simulation", "#exponential", "#oddlysatisfying"]
OVERLAYS = ["", "", "Pick a lane!", "Underdog?"]


def _lane_model(hp, n_of_k, rate, sat):
    """Temps prévu pour casser toutes les barrières : chaque balle frappe `rate` fois/s, moins quand la pile s'épaissit."""
    return sum(h / (rate * n / (1 + n / sat)) for h, n in zip(hp, n_of_k))


def sample(rng):
    mult = int(rng.choice([2, 3, 3]))
    nb = int(rng.integers(6, 8)) if mult == 2 else int(rng.integers(5, 7))
    add = int(rng.choice([1, 2, 2, 3, 5]))
    r = float(rng.uniform(7.5, 9) if mult == 3 else rng.uniform(9, 11.5))   # ×3 : jusqu'à 243 balles dans un couloir
    s = int(rng.integers(4, 13))
    # Croissance des points de vie qui équilibre la course d'après le modèle ; la simulation la corrige ensuite.
    rate, sat = 1.1, 15.0
    def gap(g):
        hp = [g ** k for k in range(nb)]
        return _lane_model(hp, [mult ** k for k in range(nb)], rate, sat) / _lane_model(hp, [s + add * k for k in range(nb)], rate, sat)
    growth = min(np.linspace(1.2, 2.2, 101), key=lambda g: abs(gap(g) - 1))
    left_mult = bool(rng.random() < 0.5)
    cm, ca = PAIRS[int(rng.integers(len(PAIRS)))][:: 1 if rng.random() < 0.5 else -1]
    return dict(
        theme=pick_theme(rng),
        root=int(rng.integers(50, 62)),
        mult=mult, add=add, s=s, nb=nb, growth=float(growth), r=r,
        rm=f"×{mult}", ra=f"+{add}",
        left_mult=left_mult, color_mult=cm, color_add=ca,
        gravity=float(rng.uniform(1300, 1500)),
        bounce_h=float(rng.uniform(110, 150)),      # hauteur minimale du rebond sur une barrière
        hp_scale=float(rng.uniform(8, 14)),          # premier point de vie, réajusté entre essais
        title_idx=int(rng.integers(len(TITLES))),
        overlay_idx=int(rng.integers(len(OVERLAYS))),
        trails=int(rng.integers(2, 4)),
    )


def _run(seed, p, hp_scale, growth):
    rng = np.random.default_rng(seed)
    nb, r = p["nb"], p["r"]
    ys = np.linspace(Y_FIRST, Y_LAST, nb)
    hp0 = np.maximum(1, np.round(hp_scale * growth ** np.arange(nb))).astype(int)
    lane_mult = 0 if p["left_mult"] else 1
    rule = {lane_mult: ("mult", p["mult"]), 1 - lane_mult: ("add", p["add"])}
    cols = {lane_mult: bgr(NAMED[p["color_mult"]]), 1 - lane_mult: bgr(NAMED[p["color_add"]])}
    hp = [hp0.copy(), hp0.copy()]
    broken = [0, 0]
    b = Balls()

    def spawn(lane, n, y_max):
        x0, x1 = LANES[lane]
        for _ in range(n):
            q = [rng.uniform(x0 + r + 2, x1 - r - 2), rng.uniform(Y_TOP + r + 4, max(Y_TOP + r + 6, y_max - r - 4))]
            b.add(q, [rng.uniform(-250, 250), rng.uniform(-200, 50)], r, lane)

    spawn(lane_mult, 1, Y_FIRST)
    spawn(1 - lane_mult, p["s"], Y_FIRST)
    vmin = np.sqrt(2 * p["gravity"] * p["bounce_h"])
    dt = 1 / SIM_FPS / SUB
    frames, events, progress = [], [], []
    fin, winner = None, None
    last_ev = [-9, -9]
    max_frames = int((MAX_FINALE + TAIL + 1) * SIM_FPS)
    for f in range(max_frames):
        for _ in range(SUB):
            b.vel[:, 1] += p["gravity"] * dt
            b.pos += b.vel * dt
            i, j = near_pairs(b)
            collide_pairs(b, i, j, 0.85, 0.7)
            for lane in (0, 1):
                m = b.kind == lane
                if not m.any():
                    continue
                x0, x1 = LANES[lane]
                lo, hi = b.pos[:, 0] < x0 + r, b.pos[:, 0] > x1 - r
                b.pos[m & lo, 0] = x0 + r
                b.vel[m & lo, 0] = np.abs(b.vel[m & lo, 0])
                b.pos[m & hi, 0] = x1 - r
                b.vel[m & hi, 0] = -np.abs(b.vel[m & hi, 0])
                top = m & (b.pos[:, 1] < Y_TOP + r)
                b.pos[top, 1] = Y_TOP + r
                b.vel[top, 1] = np.abs(b.vel[top, 1])
                k = broken[lane]
                if k >= nb:                       # couloir terminé : les balles franchissent la ligne d'arrivée
                    continue
                yb = ys[k]
                hit = m & (b.pos[:, 1] > yb - r)
                if not hit.any():
                    continue
                idx = np.nonzero(hit)[0]
                down = b.vel[idx, 1] > 0
                b.pos[idx, 1] = yb - r
                vy = b.vel[idx, 1]
                b.vel[idx, 1] = -np.maximum(np.abs(vy) * 0.9, vmin * rng.uniform(0.85, 1.0, len(idx)))
                nh = int(down.sum())
                if nh == 0 or fin is not None:
                    continue
                hp[lane][k] -= nh
                if f - last_ev[lane] >= 3:
                    last_ev[lane] = f
                    x = float(b.pos[idx[down][0], 0])
                    events.append(Event(f, "bounce", x, yb, cols[lane], 0.5, (lane * 2 + k) % 5))
                if hp[lane][k] <= 0:
                    hp[lane][k] = 0
                    broken[lane] += 1
                    xm = (x0 + x1) / 2
                    for xx in np.linspace(x0 + 30, x1 - 30, 6):
                        events.append(Event(f, "shard", float(xx), yb, cols[lane], 1, 0))
                    if broken[lane] == nb:
                        fin, winner = f, lane
                        events.append(Event(f, "win", xm, yb, cols[lane], 1.0, 0))
                        continue
                    events.append(Event(f, "chime", xm, yb, cols[lane], 0.9, broken[lane] % 5))
                    kind, v = rule[lane]
                    if kind == "mult":
                        src = np.nonzero(b.kind == lane)[0]
                        for s_ in src:
                            for _c in range(v - 1):
                                a = rng.uniform(0, 2 * np.pi)
                                q = b.pos[s_] + 1.6 * r * np.array([np.cos(a), -abs(np.sin(a))])
                                q[0] = np.clip(q[0], x0 + r, x1 - r)
                                b.add(q, b.vel[s_] + rng.uniform(-200, 200, 2), r, lane)
                    else:
                        spawn(lane, v, yb)
                    events.append(Event(f, "pop", xm, yb - 40, cols[lane], 0.8, broken[lane] % 5))
        gone = np.nonzero(b.pos[:, 1] > Y_FLOOR + 60)[0]
        if len(gone):
            b.remove(gone)
        progress.append([broken[l] + (1 - hp[l][broken[l]] / hp0[broken[l]] if broken[l] < nb else 0) for l in (0, 1)])
        frames.append(_frame(b, cols, rule, ys, hp, hp0, broken, nb, p))
        if fin is not None and f >= fin + int(TAIL * SIM_FPS):
            break
        if len(b) > 700:
            return None
    return frames, events, fin, winner, np.array(progress), lane_mult


def _frame(b, cols, rule, ys, hp, hp0, broken, nb, p):
    walls, hud = [], []
    for lane in (0, 1):
        x0, x1 = LANES[lane]
        c = cols[lane]
        dim = tuple(int(v * 0.45) for v in c)
        walls += [(x0 - 4, Y_TOP - 4, x0 - 4, Y_FLOOR + 4, dim, 4), (x1 + 4, Y_TOP - 4, x1 + 4, Y_FLOOR + 4, dim, 4),
                  (x0 - 4, Y_TOP - 4, x1 + 4, Y_TOP - 4, dim, 4)]
        for xx in np.arange(x0, x1, 28):           # ligne d'arrivée en pointillés
            walls.append((xx, Y_FLOOR + 4, min(x1, xx + 14), Y_FLOOR + 4, dim, 4))
        xm = (x0 + x1) / 2
        kind, v = rule[lane]
        hud.append(("text", f"×{v}" if kind == "mult" else f"+{v}", xm, Y_TOP - 70, 92, c, "c", 6))
        for k in range(broken[lane], nb):
            active = k == broken[lane]
            walls.append((x0, ys[k] + 3, x1, ys[k] + 3, c if active else dim, 7 if active else 5))
            hud.append(("text", str(int(hp[lane][k] if active else hp0[k])), xm, ys[k] + 38, 40 if active else 32,
                        (255, 255, 255) if active else tuple(int(v * 0.7) for v in c), "c", 3))
    return Frame(
        balls=np.column_stack([b.pos, b.rad]).astype(np.float32) if len(b) else np.zeros((0, 3), np.float32),
        colors=np.array([cols[int(k)] for k in b.kind], np.uint8).reshape(-1, 3),
        uids=b.uid.copy(), rings=[], hud=hud, walls=walls)


def simulate(seed, p):
    """Essais successifs : points de vie mis à l'échelle (durée) et croissance corrigée (arrivée serrée)."""
    scale, growth = p["hp_scale"], p["growth"]
    best = None
    for tr in range(10):
        out = _run(seed + 7919 * tr, p, scale, growth)
        if out is None:
            return None
        frames, events, fin, winner, prog, lane_mult = out
        t = fin / SIM_FPS if fin is not None else None
        if t is None:
            scale *= 0.6
            continue
        margin = float(p["nb"] - prog[fin, 1 - winner])
        if MIN_FINALE <= t <= MAX_FINALE and (best is None or margin < best[0]):
            best = (margin, tr, scale, growth, out)
            if margin < 0.35:
                break
        scale *= 0.5 * (MIN_FINALE + MAX_FINALE) / t
        # L'outsider (×m) profite des grosses barrières du bas : plus de croissance l'aide, moins le freine.
        step = min(0.25, 0.06 * margin)
        growth = max(1.1, growth + (step if winner != lane_mult else -step))
    if best is None:
        return None
    margin, tr, scale, growth, (frames, events, fin, winner, prog, lane_mult) = best
    t = fin / SIM_FPS
    p["growth_used"] = round(growth, 3)
    p["hp_scale_used"], p["sub_seed"] = scale, seed + 7919 * tr
    lead = [int(np.argmax(x)) for x in prog[: fin + 1: SIM_FPS // 2] if abs(x[0] - x[1]) > 0.05]
    changes = sum(1 for a, c in zip(lead, lead[1:]) if a != c)
    under = winner == lane_mult
    win_rule = p["rm"] if under else p["ra"]
    title = TITLES[p["title_idx"]].format(**p)
    col = bgr(NAMED[p["color_mult"] if under else p["color_add"]])
    return Timeline(
        mechanic=NAME, seed=seed, params=p, title=title, frames=frames, events=events,
        finale_frame=fin, finale_text=f"{win_rule} WINS!", finale_color=col, end_sub="",
        overlay=OVERLAYS[p["overlay_idx"]], palette=p["theme"], trails=p["trails"],
        metrics=dict(finale_s=round(t, 2), winner=win_rule, underdog_won=under, margin=round(margin, 2),
                     leader_changes=changes, balls=len(frames[fin].balls)),
    )


def score(tl):
    m = tl.metrics
    s = 0.35 * duration_score(m["finale_s"])
    s += 0.4 * max(0.0, 1 - m["margin"] / 1.5)          # arrivée serrée
    s += 0.25 * min(1.0, m["leader_changes"] / 2)        # l'outsider remonte
    return s


def comment(tl, rng):
    p = tl.params
    opts = [f"Did you pick {p['rm']} or {p['ra']}? 👇", "Multiply or add: which one would you bet on next time? 👇",
            f"Would 1 ball {p['rm']} still win against {p['s'] + 5} balls? 🤔👇"]
    return str(rng.choice(opts))
