"""Mécanique 9 : la balle qui joue un air connu. Chaque rebond sur une plateforme = la note suivante, pile en rythme.

Idée de vassili (2026-10-26) : « les balles tombent en rythme pour qu'on reconnaisse la musique ».
Le rythme est garanti par construction : la note k tombe à l'instant t_k, et la balle va de la plateforme k à la
plateforme k+1 en exactement t_{k+1} - t_k (tir balistique résolu, gravité fixe pour toute la vidéo).
La hauteur de la plateforme suit la hauteur de la note : on « voit » la mélodie.
Airs uniquement du domaine public (compositeurs morts depuis plus de 70 ans, comptines traditionnelles).
"""
import numpy as np

from .. import beats
from ..palettes import pick_theme
from ..scene import Frame, Timeline
from .common import MAX_FINALE, MIN_FINALE, SIM_FPS, TAIL, Event, duration_score


def _ph(*notes):
    """Notes MIDI, durée 1 temps par défaut ; (note, durée) pour une autre durée."""
    return [(n, 1.0) if isinstance(n, int) else (int(n[0]), float(n[1])) for n in notes]


_ODE_A = _ph(64, 64, 65, 67, 67, 65, 64, 62, 60, 60, 62, 64)
_ODE_B = _ph(62, 62, 64, 60, 62, (64, .5), (65, .5), 64, 60, 62, (64, .5), (65, .5), 64, 62, 60, 62, (55, 2))
_TW_A = _ph(60, 60, 67, 67, 69, 69, (67, 2), 65, 65, 64, 64, 62, 62, (60, 2))
_TW_B = _ph(67, 67, 65, 65, 64, 64, (62, 2))
_FE_A = _ph(*[(n, .5) for n in (76, 75, 76, 75, 76, 71, 74, 72)], (69, 1.5), (60, .5), (64, .5), (69, .5), (71, 1.5))
_FE_B = _ph((64, .5), (68, .5), (71, .5), (72, 1.5), (64, .5))
_FE_C = _ph(*[(n, .5) for n in (76, 75, 76, 75, 76, 71, 74, 72)], (69, 1.5), (60, .5), (64, .5), (69, .5), (71, 1.5),
            (64, .5), (72, .5), (71, .5), (69, 2))
_MK = _ph(*[(n, .5) for n in (69, 71, 72, 74, 76, 72)], 76, *[(n, .5) for n in (75, 71)], 75,
          *[(n, .5) for n in (74, 70)], 74, *[(n, .5) for n in (69, 71, 72, 74, 76, 72, 76, 81, 79, 76, 72, 76)], (79, 2))
_JB_A = _ph(64, 64, (64, 2), 64, 64, (64, 2), 64, 67, (60, 1.5), (62, .5), (64, 4))
_JB_B1 = _ph(65, 65, (65, 1.5), (65, .5), 65, 64, 64, (64, .5), (64, .5), 64, 62, 62, 64, (62, 2), (67, 2))
_JB_B2 = _ph(65, 65, (65, 1.5), (65, .5), 65, 64, 64, (64, .5), (64, .5), 67, 67, 65, 62, (60, 4))
_FJ = (_ph(60, 62, 64, 60) * 2 + _ph(64, 65, (67, 2)) * 2 + _ph((67, .5), (69, .5), (67, .5), (65, .5), 64, 60) * 2
       + _ph(60, 55, (60, 2)) * 2)
_MARY = _ph(64, 62, 60, 62, 64, 64, (64, 2), 62, 62, (62, 2), 64, 67, (67, 2), 64, 62, 60, 62, 64, 64, 64, 64, 62, 62,
            64, 62, (60, 4))
_EKN = _ph((67, 1.5), (62, .5), (67, 1.5), (62, .5), (67, .5), (62, .5), (67, .5), (71, .5), (74, 2),
           (72, 1.5), (69, .5), (72, 1.5), (69, .5), (72, .5), (69, .5), (66, .5), (69, .5), (62, 2))
_TURK = _ph(*[(n, .25) for n in (71, 69, 68, 69)], 72, *[(n, .25) for n in (74, 72, 71, 72)], 76,
            *[(n, .25) for n in (77, 76, 75, 76, 83, 81, 80, 81, 83, 81, 80, 81)], (84, 2))

# nom affiché à la fin, notes, plage de tempo (noires par minute), accélération à chaque reprise
SONGS = {
    "ode": ("ODE TO JOY!", _ODE_A + _ph((64, 1.5), (62, .5), (62, 2)) + _ODE_A + _ph((62, 1.5), (60, .5), (60, 2))
            + _ODE_B + _ODE_A + _ph((62, 1.5), (60, .5), (60, 2)), (100, 132), 1.0),
    "twinkle": ("TWINKLE TWINKLE!", _TW_A + _TW_B * 2 + _TW_A, (90, 120), 1.0),
    "elise": ("FÜR ELISE!", _FE_A + _FE_B + _FE_C, (66, 84), 1.0),
    "mountain": ("MOUNTAIN KING!", _MK, (96, 112), 1.22),
    "jingle": ("JINGLE BELLS!", _JB_A + _JB_B1 + _JB_A + _JB_B2, (110, 140), 1.0),
    "frere": ("FRÈRE JACQUES!", _FJ, (96, 126), 1.0),
    "mary": ("MARY'S LAMB!", _MARY, (100, 130), 1.0),
    "nacht": ("EINE KLEINE!", _EKN, (110, 132), 1.0),
    "turkish": ("TURKISH MARCH!", _TURK, (96, 116), 1.0),
}

NAME = "melody_bounce"
TITLES = ["Can you guess the song?", "Every bounce is a note", "Guess the song before the end!",
          "This ball knows a famous song"]
# Beats originaux (trap / phonk / drill) : pas d'air à deviner, on attend que le beat tombe.
BEAT_TITLES = ["This ball makes a beat 🔥", "Every bounce is a note... wait for the drop", "Wait for the beat drop 🔥",
               "Bouncing ball {style} beat"]
BEAT_OVERLAYS = ["", "", "Wait for it..."]
DESCRIPTIONS = ["Every bounce plays the next note. Did you recognize it?", "A bouncing ball plays a famous melody, note by note.",
                "Perfect timing: each bounce lands exactly on the beat."]
HASHTAGS = ["#shorts", "#satisfying", "#music", "#guessthesong", "#oddlysatisfying"]
OVERLAYS = ["", "", "Guess the song!", "Name it!"]
X_MIN, X_MAX = 120.0, 900.0
Y_HIGH, Y_LOW = 820.0, 1460.0     # plateforme de la note la plus aiguë / la plus grave
PLAT_W = 150.0
# couleur par classe de hauteur (do = rouge … si = rose), BGR
_HUES = [(80, 80, 255), (60, 140, 255), (60, 200, 255), (60, 235, 220), (90, 240, 150), (160, 230, 70),
         (230, 210, 60), (255, 160, 70), (255, 110, 120), (240, 100, 200), (200, 110, 255), (150, 110, 255)]


def sample(rng):
    if rng.random() < 0.5:                 # beat original style Gen Z
        style = str(rng.choice(list(beats.STYLES)))
        return dict(
            theme=pick_theme(rng), song="beat", beat_style=style, beat_seed=int(rng.integers(1 << 30)),
            r=float(rng.uniform(24, 32)), step=float(rng.uniform(150, 230)), h_max=float(rng.uniform(380, 470)),
            title_idx=int(rng.integers(len(BEAT_TITLES))), overlay_idx=int(rng.integers(len(BEAT_OVERLAYS))),
            trails=int(rng.integers(5, 9)), sfx="soft",
        )
    song = str(rng.choice(list(SONGS)))
    lo, hi = SONGS[song][2]
    return dict(
        theme=pick_theme(rng),
        song=song,
        bpm=float(rng.uniform(lo, hi)),
        r=float(rng.uniform(24, 32)),
        step=float(rng.uniform(150, 230)),     # déplacement horizontal par temps
        h_max=float(rng.uniform(380, 470)),    # hauteur de l'arc le plus long
        title_idx=int(rng.integers(len(TITLES))),
        overlay_idx=int(rng.integers(len(OVERLAYS))),
        trails=int(rng.integers(5, 9)),
        sfx="soft",
    )


def _schedule(p):
    """Instants des notes : l'air est repris jusqu'à durer au moins 28 s (avec accélération éventuelle)."""
    if p["song"] == "beat":
        return list(beats.make(p["beat_seed"], p["beat_style"])["notes"])
    if p["song"] == "groove":
        return list(beats.make_groove(p["beat_style"])["notes"])
    name, notes, _, accel = SONGS[p["song"]]
    t, out, rep, beat = 0.8, [], 0, 60.0 / p["bpm"]
    while True:
        for n, d in notes:
            out.append((t, n))
            t += d * beat
        rep += 1
        beat /= accel
        if out[-1][0] >= 28 or rep >= 6:
            return out


def simulate(seed, p):
    rng = np.random.default_rng(seed)
    sched = _schedule(p)
    times = np.array([s[0] for s in sched])
    midis = np.array([s[1] for s in sched])
    if not (MIN_FINALE <= times[-1] <= MAX_FINALE):
        return None
    lo, hi = midis.min(), midis.max()
    ys = Y_LOW + (midis - lo) / max(1, hi - lo) * (Y_HIGH - Y_LOW) - p["r"] - 2
    # Parcours horizontal en zigzag : plus la note est longue, plus la balle va loin.
    gaps = np.diff(times)
    beat = np.median(gaps)
    xs = [float(rng.uniform(300, 700))]
    d = 1 if rng.random() < 0.5 else -1
    for gp in gaps:
        dx = float(np.clip(p["step"] * gp / beat, 70, 300)) * (1 + rng.uniform(-0.15, 0.15))
        nx = xs[-1] + d * dx
        if nx < X_MIN or nx > X_MAX or rng.random() < 0.18:
            d = -d
            nx = xs[-1] + d * dx
        xs.append(float(np.clip(nx, X_MIN, X_MAX)))
    xs = np.array(xs)
    # Vol au plus `cap` secondes : sur une note longue, la balle attend sur sa plateforme puis repart,
    # et retombe toujours pile sur le temps. Gravité unique : le vol le plus long monte de h_max (h = g T² / 8).
    cap = min(gaps.max(), 1.6 * beat)
    g = float(np.clip(8 * p["h_max"] / cap ** 2, 900, 12000))
    starts = np.concatenate([[[xs[0] - 60, 330.0]], np.column_stack([xs, ys])[:-1]])
    T = np.minimum(np.diff(np.concatenate([[0.0], times])), cap)
    # Première note : la balle est lâchée sans vitesse au-dessus de la première plateforme.
    T[0] = min(times[0], np.sqrt(2 * max(50.0, ys[0] - 380) / g))
    starts[0] = (xs[0], ys[0] - 0.5 * g * T[0] ** 2)
    t0s = times - T                           # instant du départ (après l'attente éventuelle)
    ends = np.column_stack([xs, ys])
    v = np.column_stack([(ends[:, 0] - starts[:, 0]) / T, (ends[:, 1] - starts[:, 1]) / T - g * T / 2])
    apex = starts[:, 1] - np.where(v[:, 1] < 0, v[:, 1] ** 2 / (2 * g), 0)
    if apex.min() < 300:                    # l'arc sortirait par le haut (zone du titre)
        return None
    fin = int(round(times[-1] * SIM_FPS))
    n_frames = fin + int(TAIL * SIM_FPS) + 1
    cols = [_HUES[int(m) % 12] for m in midis]
    theme = p["theme"]
    events = []
    beat = (beats.make(p["beat_seed"], p["beat_style"]) if p["song"] == "beat"
            else beats.make_groove(p["beat_style"]) if p["song"] == "groove" else None)
    if beat:
        for t, kind in beat["drums"]:
            if t <= times[-1] + 1e-6:
                events.append(Event(int(round(t * SIM_FPS)), kind, 540.0, 1500.0, (255, 255, 255), 1.0, 0))
        for t, m in beat["bass"]:
            if t <= times[-1] + 1e-6:
                events.append(Event(int(round(t * SIM_FPS)), "808", 540.0, 1500.0, (255, 255, 255), 1.0, int(m)))
    lead = "cowbell" if beat and beat["lead"] == "cowbell" else "tone"
    for k, (t, m) in enumerate(sched):
        f = int(round(t * SIM_FPS))
        events.append(Event(f, lead, float(xs[k]), float(ys[k] + p["r"]), cols[k], 1.0, int(m)))
        events.append(Event(f, "shard", float(xs[k]), float(ys[k] + p["r"] + 4), cols[k], 1, 0))
    frames = []
    seg = 0
    ball_col = np.array((240, 240, 250), float)
    for f in range(n_frames):
        t = f / SIM_FPS
        while seg < len(times) - 1 and t >= times[seg]:
            seg += 1
        if t >= times[-1]:
            pos = ends[-1]
            k_last = len(times) - 1
        else:
            dt = max(0.0, t - t0s[seg])
            pos = starts[seg] + v[seg] * dt + np.array([0, 0.5 * g * dt * dt])
            k_last = seg - 1
        if k_last >= 0:
            ball_col += (np.array(cols[k_last], float) - ball_col) * 0.25
        walls = []
        for k in range(max(0, seg - 14), min(len(times), seg + 2)):
            age = t - times[k]
            if age < -0.6 or age > 3.0:
                continue
            if age < 0:                                 # plateforme suivante : apparaît en avance, discrète
                a, th = 0.4 * (1 + age / 0.6), 12
            else:                                       # touchée : s'allume puis s'éteint doucement
                a, th = max(0.0, 1 - age / 3.0) ** 0.8, 20 if age < 0.12 else 14
            c = tuple(int(v_ * (0.25 + 0.75 * a)) for v_ in cols[k])
            if a > 0.02:
                yp = ys[k] + p["r"] + 9
                walls.append((xs[k] - PLAT_W / 2, yp, xs[k] + PLAT_W / 2, yp, c, th))
        frames.append(Frame(
            balls=np.array([[pos[0], pos[1], p["r"]]], np.float32),
            colors=np.array([ball_col], np.uint8), uids=np.array([0]), rings=[], walls=walls))
    name = beat["name"] if beat else SONGS[p["song"]][0]
    if beat:
        p["bpm"] = round(beat["bpm"], 1)
    p["gravity"] = round(g)
    p["notes"] = len(times)
    return Timeline(
        mechanic=NAME, seed=seed, params=p, title=(BEAT_TITLES[p["title_idx"]].format(style=p["beat_style"]) if beat else TITLES[p["title_idx"]]),
        frames=frames, events=events,
        finale_frame=fin, finale_text=name, finale_color=(255, 255, 255), end_sub="",
        overlay=(BEAT_OVERLAYS if beat else OVERLAYS)[p["overlay_idx"]], palette=theme, trails=p["trails"],
        metrics=dict(finale_s=round(float(times[-1]), 2), song=p.get("beat_style", p["song"]), notes=len(times), gravity=round(g)),
    )


def score(tl):
    return 0.6 * duration_score(tl.metrics["finale_s"]) + 0.4 * (1.0 if 40 <= tl.metrics["notes"] <= 110 else 0.6)


def comment(tl, rng):
    if tl.params["song"] in ("beat", "groove"):
        return str(rng.choice(["Phonk, trap or drill: what should the ball play next? 👇", "Did the drop hit? 🔥👇",
                               "Should I make a full song with this beat? 👇"]))
    opts = ["Did you guess it before the end? 👇", "What song should the ball play next? 🎵👇",
            "How many notes did it take you to recognize it? 👇"]
    return str(rng.choice(opts))
