"""Sons synthétisés en numpy : rebonds, pops, carillons sur gamme pentatonique + nappe douce.

Aucun échantillon externe. Limite de voix simultanées et limiteur doux contre la saturation.
"""
import wave

import numpy as np

SR = 44100
PENTA = [0, 2, 4, 7, 9]   # majeure pentatonique


def note_freq(root_midi, degree):
    octave, step = divmod(int(degree), len(PENTA))
    midi = root_midi + 12 * octave + PENTA[step]
    return 440.0 * 2 ** ((midi - 69) / 12)


def _env(n, attack, decay):
    t = np.arange(n) / SR
    a = np.minimum(1.0, t / max(attack, 1e-4))
    return a * np.exp(-t / decay)


def _bounce(f):
    n = int(SR * 0.35)
    t = np.arange(n) / SR
    w = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t) + 0.12 * np.sin(2 * np.pi * 3 * f * t)
    return w * _env(n, 0.002, 0.09) * 0.55


def _hit(f):
    n = int(SR * 0.12)
    t = np.arange(n) / SR
    w = np.sin(2 * np.pi * f * 2 * t) * 0.7 + 0.3 * np.sin(2 * np.pi * f * 3.01 * t)
    return w * _env(n, 0.001, 0.03) * 0.35


def _pop(f, rng):
    n = int(SR * 0.22)
    t = np.arange(n) / SR
    sweep = f * 1.8 * np.exp(-t * 18) + f * 0.6
    ph = 2 * np.pi * np.cumsum(sweep) / SR
    noise = rng.standard_normal(n) * np.exp(-t * 60) * 0.25
    return (np.sin(ph) * _env(n, 0.001, 0.06) + noise) * 0.6


def _chime(f):
    n = int(SR * 1.4)
    t = np.arange(n) / SR
    parts = [(1.0, 1.0, 0.6), (2.0, 0.45, 0.4), (2.76, 0.3, 0.25), (5.4, 0.12, 0.12)]
    w = sum(a * np.sin(2 * np.pi * f * m * t) * np.exp(-t / d) for m, a, d in parts)
    return w * np.minimum(1, t / 0.003) * 0.5


def _boom(f, rng):
    n = int(SR * 2.2)
    t = np.arange(n) / SR
    sub = np.sin(2 * np.pi * (f / 4) * t * (1 - 0.3 * t / 2.2)) * np.exp(-t / 0.7)
    noise = rng.standard_normal(n)
    noise = np.convolve(noise, np.ones(30) / 30, mode="same") * np.exp(-t / 0.35)
    shimmer = sum(np.sin(2 * np.pi * f * m * t) * np.exp(-t / 0.9) * a for m, a in [(1, .4), (1.5, .3), (2, .25), (3, .15)])
    return (sub * 0.9 + noise * 1.2 + shimmer * 0.6) * np.minimum(1, t / 0.004) * 0.8


def _win(root):
    # Petit arpège montant de carillons.
    seq = [0, 2, 4, 5, 7]
    n = int(SR * 2.0)
    out = np.zeros(n)
    for k, d in enumerate(seq):
        c = _chime(note_freq(root + 12, d))
        s = int(k * 0.09 * SR)
        m = min(len(c), n - s)
        out[s:s + m] += c[:m] * 0.8
    return out


def pad(duration, root, rng, level=0.07):
    """Nappe de fond : accord pentatonique désaccordé avec un vibrato lent."""
    n = int(SR * duration)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    chord = [note_freq(root - 12, d) for d in (0, 2, 4, 7)]
    for k, f in enumerate(chord):
        lfo = 1 + 0.5 * np.sin(2 * np.pi * (0.05 + 0.03 * k) * t + rng.uniform(0, 6))
        for side, det in ((0, -0.15), (1, 0.15)):
            out[:, side] += np.sin(2 * np.pi * (f + det) * t + rng.uniform(0, 6)) * lfo
    fade = np.minimum(1, t / 2.0) * np.minimum(1, (duration - t) / 1.0).clip(0, 1)
    return out / len(chord) * level * fade[:, None]


def _formant_voice(f0, F, dur, rng, K=16, breath=0.04):
    """Synthèse additive : harmoniques de f0(t) pondérées par des formants F(t) = [(freq, largeur), ...]."""
    n = len(f0)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    out = np.zeros(n)
    for k in range(1, K + 1):
        fk = f0 * k
        g = 0.12 + sum(a / (1 + ((fk - fc) / bw) ** 2) for fc, bw, a in F)
        g = np.where(fk < SR / 2.2, g, 0)
        out += np.sin(k * ph) * g / k ** 0.6
    out += rng.standard_normal(n) * breath
    return out


def _meow(f, rng, dur=0.55):
    """Miaou synthétique : montée puis descente de hauteur, voyelles i -> a -> ou."""
    n = int(SR * dur)
    t = np.linspace(0, 1, n)
    contour = 1 + 0.35 * np.sin(np.pi * np.minimum(1, t / 0.7)) - 0.25 * t
    f0 = f * contour * (1 + 0.012 * np.sin(2 * np.pi * 6 * t * dur))
    def lerp3(a, b, c):
        return np.where(t < 0.4, a + (b - a) * t / 0.4, b + (c - b) * (t - 0.4) / 0.6)
    F1 = lerp3(400, 900, 450)
    F2 = lerp3(2200, 1500, 900)
    w = _formant_voice(f0, [(F1, 250, 1.0), (F2, 300, 0.6), (3000, 400, 0.15)], dur, rng, K=10)
    env = np.minimum(1, t / 0.08) * np.minimum(1, (1 - t) / 0.25) ** 1.5
    return w * env * 0.5


def _boing(f, rng):
    n = int(SR * 0.45)
    t = np.arange(n) / SR
    fr = f * 0.5 * (1 + 0.6 * np.exp(-t * 6)) * (1 + 0.08 * np.sin(2 * np.pi * 18 * t) * np.exp(-t * 4))
    ph = 2 * np.pi * np.cumsum(fr) / SR
    return (np.sin(ph) + 0.3 * np.sin(2 * ph)) * _env(n, 0.003, 0.18) * 0.6


def _quack(f, rng):
    dur = 0.22
    n = int(SR * dur)
    t = np.linspace(0, 1, n)
    f0 = f * 0.45 * (1.15 - 0.3 * t)
    w = _formant_voice(f0, [(1100, 300, 1.0), (1800, 250, 0.7), (2800, 400, 0.3)], dur, rng, K=24, breath=0.08)
    return np.tanh(w * 2.5) * np.minimum(1, t / 0.05) * np.minimum(1, (1 - t) / 0.3) * 0.45


def _squeak(f, rng):
    n = int(SR * 0.25)
    t = np.arange(n) / SR
    fr = f * 3 * (1 + 0.25 * np.sin(np.pi * t / 0.25)) * (1 + 0.03 * np.sin(2 * np.pi * 30 * t))
    ph = 2 * np.pi * np.cumsum(fr) / SR
    return (np.sin(ph) + 0.2 * np.sin(3 * ph)) * np.minimum(1, t / 0.02) * np.minimum(1, (0.25 - t) / 0.08).clip(0, 1) * 0.4


def _crunch(f, rng, size=1.0):
    """Craquement : bruit filtré très court + petit choc grave (plus grave pour une grosse balle)."""
    n = int(SR * 0.16)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    k = max(2, int(6 * size))
    noise = np.convolve(noise, np.ones(k) / k, mode="same")
    crackle = noise * np.exp(-t / 0.035) * (1 + 0.6 * (rng.random(n) < 0.02))
    thud = np.sin(2 * np.pi * (f * 0.5) * t * (1 - 0.4 * t / 0.16)) * np.exp(-t / 0.05)
    return (crackle * 1.1 + thud * 0.7) * np.minimum(1, t / 0.0015) * 0.55


def _hiss(f, rng):
    """Vérin hydraulique : souffle grave qui monte puis retombe."""
    n = int(SR * 0.9)
    t = np.arange(n) / SR
    noise = np.convolve(rng.standard_normal(n), np.ones(40) / 40, mode="same")
    hum = np.sin(2 * np.pi * 70 * t) * 0.3
    env = np.minimum(1, t / 0.12) * np.exp(-np.maximum(0, t - 0.3) / 0.25)
    return (noise * 2.2 + hum) * env * 0.35


def _heartbeat(f, rng):
    n = int(SR * 0.5)
    t = np.arange(n) / SR
    beat = lambda t0: np.sin(2 * np.pi * 55 * (t - t0)) * np.exp(-np.maximum(0, t - t0) / 0.07) * (t >= t0)
    return (beat(0) + 0.7 * beat(0.18)) * 0.9


SILLY = {"meow": _meow, "boing": _boing, "quack": _quack, "squeak": _squeak}
PACKS = ["meow", "meow", "meow", "boing", "boing", "quack", "squeak", "classic"]


def render_audio(events, duration, time_of, root, seed, path, pack="classic", finale_frame=None, slow=None,
                 max_voices=6, min_gap=0.035, pitch_by_note=False):
    """`time_of(frame)` donne l'instant de sortie (prise en compte du ralenti).

    Pack rigolo : rebonds et pops deviennent des miaous / boings / coin-coins / couics, de plus en plus
    graves à l'approche de la fin. `slow` = (début, fin) du ralenti en secondes de sortie : battements de coeur.
    """
    rng = np.random.default_rng(seed)
    n = int(SR * duration) + 1
    mix = np.zeros((n, 2))
    cache = {}
    active_until = []
    last_t = {}
    silly = SILLY.get(pack)
    gap_silly = 0.14
    evs = sorted(events, key=lambda e: e.frame)
    if slow:
        t = slow[0]
        while t < slow[1] - 0.2:
            evs.append(Event_(t, "heart"))
            t += 0.75
    for ev in sorted(evs, key=lambda e: time_of(e.frame) if not isinstance(e, Event_) else e.t):
        t0 = ev.t if isinstance(ev, Event_) else time_of(ev.frame)
        kind = ev.kind
        if silly and kind in ("bounce", "pop", "chime"):
            prog = min(1.0, ev.frame / max(1, finale_frame)) if finale_frame else 0.0
            if t0 - last_t.get("silly", -1) < gap_silly:
                kind = "hit" if kind == "bounce" else kind
                if kind == "pop":
                    continue
            else:
                # Plus on approche de la fin, plus c'est grave (jusqu'à -1,6 octave).
                # pitch_by_note : la hauteur suit le niveau de l'objet (petit chaton aigu, gros chat grave).
                step = min(18, int(ev.note * 1.6)) if pitch_by_note else int(round(prog * 16))
                key = ("silly", step, ev.note % 3)
                if key not in cache:
                    base = {"meow": 560, "boing": 520, "quack": 700, "squeak": 900}[pack]
                    f = base * 2 ** (-step / 10) * 2 ** ((ev.note % 3) / 12)
                    cache[key] = silly(f, rng, min(0.85, 0.32 + 0.03 * step)) if pack == "meow" else silly(f, rng)
                w = cache[key] * 0.9
                last_t["silly"] = t0
                _add(mix, w, t0, ev.x, n)
                continue
        if kind in ("bounce", "hit", "pop", "crunch") and t0 - last_t.get(kind, -1) < min_gap:
            continue
        active_until = [u for u in active_until if u > t0]
        if kind in ("bounce", "hit", "pop", "crunch") and len(active_until) >= max_voices:
            continue
        key = (kind, ev.note)
        if key not in cache:
            f = note_freq(root, ev.note)
            cache[key] = {
                "bounce": lambda: _bounce(f), "hit": lambda: _hit(f), "pop": lambda: _pop(f, rng),
                "chime": lambda: _chime(f), "boom": lambda: _boom(f, rng), "win": lambda: _win(root),
                "spawn": lambda: _hit(f * 2) * 0.5, "heart": lambda: _heartbeat(f, rng),
                "crunch": lambda: _crunch(f, rng, 1.0 + 0.6 * (2 - min(2, ev.note))), "hiss": lambda: _hiss(f, rng),
            }[kind]()
        strength = 1.0 if kind == "heart" else min(1.0, max(0.15, ev.strength))
        if silly and kind == "hit":
            strength *= 0.5
        w = cache[key] * strength
        _add(mix, w, t0, getattr(ev, "x", 540), n)
        active_until.append(t0 + len(w) / SR * 0.5)
        last_t[kind] = t0
    p = pad(duration, root, rng)
    mix[: len(p)] += p
    # Normalisation puis limiteur doux.
    peak = np.percentile(np.abs(mix), 99.9) + 1e-9
    mix = np.tanh(mix / peak * 0.75) * 0.89
    pcm = (mix * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


class Event_:
    """Événement audio placé directement en secondes de sortie (battements du ralenti)."""
    def __init__(self, t, kind):
        self.t, self.kind, self.frame, self.note, self.strength, self.x = t, kind, 0, 0, 1.0, 540


def _add(mix, w, t0, x, n):
    s = int(t0 * SR)
    m = min(len(w), n - s)
    if m <= 0 or s < 0:
        return
    pan = np.clip((x - 540) / 540, -1, 1) * 0.5
    mix[s:s + m, 0] += w[:m] * (1 - pan) * 0.7
    mix[s:s + m, 1] += w[:m] * (1 + pan) * 0.7
