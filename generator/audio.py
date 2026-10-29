"""Sons synthétisés en numpy : rebonds, pops, carillons sur gamme pentatonique + nappe douce.

Aucun échantillon externe. Limite de voix simultanées et limiteur doux contre la saturation.
"""
import wave

import numpy as np

SR = 44100
PENTA = [0, 2, 4, 7, 9]   # majeure pentatonique
SCALES = {"penta": PENTA, "minor_penta": [0, 3, 5, 7, 10], "lydian": [0, 2, 4, 6, 7, 9, 11],
          "dorian": [0, 2, 3, 5, 7, 9, 10], "japan": [0, 2, 3, 7, 8], "major": [0, 2, 4, 5, 7, 9, 11]}


def note_freq(root_midi, degree, scale=PENTA):
    octave, step = divmod(int(degree), len(scale))
    midi = root_midi + 12 * octave + scale[step]
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


# --- Instruments doux (retour de vassili du 2026-10-26 : « des trucs qui agressent moins les oreilles ») ---
# Attaques adoucies, peu d'harmoniques aiguës, décroissance naturelle. Aucun échantillon : tout est synthétisé.

def _soft_attack(n, a=0.004):
    t = np.arange(n) / SR
    return np.sin(0.5 * np.pi * np.minimum(1, t / a)) ** 2


def _kalimba(f, rng):
    n = int(SR * 0.9)
    t = np.arange(n) / SR
    w = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.45) + 0.18 * np.sin(2 * np.pi * f * 5.93 * t) * np.exp(-t / 0.05)
    return w * _soft_attack(n, 0.003) * 0.5


def _marimba(f, rng):
    n = int(SR * 0.7)
    t = np.arange(n) / SR
    w = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.28) + 0.25 * np.sin(2 * np.pi * f * 3.98 * t) * np.exp(-t / 0.06)
    return w * _soft_attack(n, 0.004) * 0.55


def _felt(f, rng):
    """Piano feutré : harmoniques qui s'éteignent vite, attaque douce."""
    n = int(SR * 1.2)
    t = np.arange(n) / SR
    w = sum(a * np.sin(2 * np.pi * f * k * t * (1 + 0.0004 * k * k)) * np.exp(-t / d)
            for k, a, d in ((1, 1.0, 0.7), (2, 0.35, 0.3), (3, 0.12, 0.15), (4, 0.05, 0.08)))
    return w * _soft_attack(n, 0.008) * 0.45


def _musicbox(f, rng):
    n = int(SR * 1.0)
    t = np.arange(n) / SR
    f = f * 2
    w = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.5) + 0.2 * np.sin(2 * np.pi * f * 3 * t) * np.exp(-t / 0.12)
    return w * _soft_attack(n, 0.002) * 0.3


def _bloop(f, rng):
    """Balle qui tape le cercle : la hauteur plonge à l'impact puis se stabilise (son de rebond, demande du 2026-10-29)."""
    n = int(SR * 0.22)
    t = np.arange(n) / SR
    ph = 2 * np.pi * np.cumsum(f * (1 + 0.9 * np.exp(-t * 28))) / SR
    w = np.sin(ph) + 0.2 * np.sin(2 * ph)
    return w * _env(n, 0.002, 0.07) * 0.5


# 2026-10-26 : vassili a gardé boîte à musique, piano feutré, kalimba et marimba (harpe et gouttes retirées).
# 2026-10-29 : « bloop » ajouté, son de rebond sur le cercle demandé par vassili.
SOFT = {"kalimba": _kalimba, "marimba": _marimba, "felt": _felt, "musicbox": _musicbox, "bloop": _bloop}


# --- Batterie et basse pour les beats (trap / phonk / drill), mixées douces ---

def _kick(rng):
    n = int(SR * 0.35)
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16) * _soft_attack(n, 0.002) * 0.9


def _clap(rng):
    n = int(SR * 0.22)
    t = np.arange(n) / SR
    noise = np.convolve(rng.standard_normal(n), np.ones(5) / 5, mode="same")
    env = sum(np.exp(-np.maximum(0, t - d) / 0.012) * (t >= d) for d in (0, 0.011, 0.022)) * 0.5 + np.exp(-t / 0.09) * 0.6
    return noise * env * 0.45


def _hat(rng):
    n = int(SR * 0.05)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    noise = noise - np.convolve(noise, np.ones(6) / 6, mode="same")    # passe-haut grossier
    return noise * np.exp(-t / 0.012) * 0.22


def _808(f, rng):
    n = int(SR * 0.9)
    t = np.arange(n) / SR
    w = np.sin(2 * np.pi * f * t * (1 + 0.04 * np.exp(-t * 20)))
    return np.tanh(1.6 * w) * np.exp(-t / 0.45) * _soft_attack(n, 0.004) * 0.55


def _cowbell(f, rng):
    """Cloche phonk : deux ondes presque carrées (rapport 1,48), décroissance courte."""
    n = int(SR * 0.4)
    t = np.arange(n) / SR
    w = np.tanh(3 * np.sin(2 * np.pi * f * t)) + 0.8 * np.tanh(3 * np.sin(2 * np.pi * f * 1.48 * t))
    w = np.convolve(w, np.ones(3) / 3, mode="same")
    return w * np.exp(-t / 0.11) * _soft_attack(n, 0.002) * 0.28


BEAT_KINDS = ("kick", "clap", "hat", "808", "cowbell")


def _swell(root, scale, inst, rng):
    """Fin douce à la place de l'explosion : accord qui s'épanouit (grave + instrument)."""
    n = int(SR * 2.6)
    t = np.arange(n) / SR
    out = np.sin(2 * np.pi * note_freq(root - 24, 0, scale) * t) * np.minimum(1, t / 0.05) * np.exp(-t / 1.0) * 0.5
    for k, d in enumerate((0, 2, 4)):
        w = inst(note_freq(root, d, scale), rng)
        s0 = int(k * 0.03 * SR)
        m = min(len(w), n - s0)
        out[s0:s0 + m] += w[:m] * 0.7
    return out


def _arp(root, scale, inst, rng):
    n = int(SR * 2.4)
    out = np.zeros(n)
    for k, d in enumerate((0, 2, 4, len(scale), len(scale) + 2)):
        w = inst(note_freq(root, d, scale), rng)
        s0 = int(k * 0.12 * SR)
        m = min(len(w), n - s0)
        out[s0:s0 + m] += w[:m] * 0.8
    return out


def _reverb(mix, rng, amount):
    """Réverbération légère : convolution par une queue de bruit amortie et filtrée."""
    from scipy.signal import fftconvolve
    n = int(SR * 1.6)
    t = np.arange(n) / SR
    out = mix.copy()
    for ch in (0, 1):
        ir = np.convolve(rng.standard_normal(n), np.ones(12) / 12, mode="same") * np.exp(-t / 0.45)
        ir[: int(0.012 * SR)] = 0
        ir /= np.sqrt((ir ** 2).sum()) + 1e-9
        out[:, ch] += fftconvolve(mix[:, ch], ir)[: len(mix)] * amount
    return out


def _lowpass(mix, fc=5000):
    a = np.exp(-2 * np.pi * fc / SR)
    from scipy.signal import lfilter
    return lfilter([1 - a], [1, -a], mix, axis=0)


def soft_style(rng):
    """Ambiance tirée au hasard pour chaque vidéo : instrument, gamme, octave, réverbération."""
    return dict(inst=str(rng.choice(list(SOFT))), scale=str(rng.choice(list(SCALES))),
                shift=int(rng.choice([0, 0, 12, -12])), reverb=float(rng.uniform(0.12, 0.28)))


SILLY = {"meow": _meow, "boing": _boing, "quack": _quack, "squeak": _squeak}
SILLY_PACKS = ["meow", "boing", "quack", "squeak"]
PACKS = ["soft"]       # depuis le 2026-10-26 : sons doux, ambiance différente à chaque vidéo (les packs rigolos restent possibles via --sfx)


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
    style = soft_style(rng) if pack.startswith("soft") else None
    if style and ":" in pack:           # « soft:kalimba » : instrument imposé (le lot répartit les instruments)
        style["inst"] = pack.split(":", 1)[1]
    if style:
        return _render_soft(events, duration, time_of, root, rng, path, style, slow)
    evs = sorted(events, key=lambda e: e.frame)
    if slow:
        t = slow[0]
        while t < slow[1] - 0.2:
            evs.append(Event_(t, "heart"))
            t += 0.75
    for ev in sorted(evs, key=lambda e: time_of(e.frame) if not isinstance(e, Event_) else e.t):
        t0 = ev.t if isinstance(ev, Event_) else time_of(ev.frame)
        kind = ev.kind
        if kind == "shard":          # éclats visuels seulement
            continue
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


def _render_soft(events, duration, time_of, root, rng, path, style, slow):
    """Sons doux : chaque rebond joue une note de l'instrument, densité limitée, réverbération et filtre doux."""
    n = int(SR * duration) + 1
    mix = np.zeros((n, 2))
    inst, scale = SOFT[style["inst"]], SCALES[style["scale"]]
    root = root + style["shift"]
    cache, last_t, active = {}, {}, []
    gap = {"bounce": 0.1, "hit": 0.14, "pop": 0.09, "crunch": 0.07}
    evs = [(time_of(e.frame), e) for e in events if e.kind != "shard"]
    if slow:
        t = slow[0]
        while t < slow[1] - 0.2:
            evs.append((t, Event_(t, "heart")))
            t += 0.75
    has_tune = any(e.kind in ("tone", "cowbell") for _, e in evs)
    has_beat = any(e.kind == "kick" for _, e in evs)
    for t0, ev in sorted(evs, key=lambda x: x[0]):
        kind = ev.kind
        if kind == "tone":                # note d'une mélodie (hauteur MIDI exacte) : jamais sautée
            key = ("tone", int(ev.note))
            if key not in cache:
                cache[key] = inst(440.0 * 2 ** ((int(ev.note) - 69) / 12), rng)
            _add(mix, cache[key] * 1.1, t0, getattr(ev, "x", 540), n)
            continue
        if kind in BEAT_KINDS:            # beat : chaque coup est joué (le rythme compte)
            key = (kind, int(ev.note))
            if key not in cache:
                f = 440.0 * 2 ** ((int(ev.note) - 69) / 12)
                cache[key] = {"kick": lambda: _kick(rng), "clap": lambda: _clap(rng), "hat": lambda: _hat(rng),
                              "808": lambda: _808(f, rng), "cowbell": lambda: _cowbell(f, rng)}[kind]()
            _add(mix, cache[key] * ev.strength, t0, getattr(ev, "x", 540), n)
            continue
        if kind in gap and t0 - last_t.get(kind, -1) < gap[kind]:
            continue
        active = [u for u in active if u > t0]
        if kind in gap and len(active) >= 4:
            continue
        note = int(ev.note)
        key = (kind, note)
        if key not in cache:
            f = note_freq(root, note % (2 * len(scale)), scale)
            cache[key] = {
                "bounce": lambda: inst(f, rng), "hit": lambda: inst(f * 2, rng) * 0.4, "spawn": lambda: inst(f * 2, rng) * 0.4,
                "pop": lambda: inst(f * 2, rng) * 0.8, "chime": lambda: inst(f, rng) + 0.5 * inst(f * 2, rng),
                "boom": lambda: _swell(root, scale, inst, rng), "win": lambda: _arp(root + 12, scale, inst, rng),
                "heart": lambda: _heartbeat(f, rng) * 0.7,
                "crunch": lambda: _crunch(f, rng, 1.0 + 0.6 * (2 - min(2, note))) * 0.6, "hiss": lambda: _hiss(f, rng) * 0.6,
            }.get(kind, lambda: inst(f, rng))()
        strength = 1.0 if kind in ("heart", "win", "boom") else min(1.0, max(0.25, ev.strength))
        w = cache[key] * strength
        _add(mix, w, t0, getattr(ev, "x", 540), n)
        active.append(t0 + min(0.25, len(w) / SR * 0.3))
        last_t[kind] = t0
    if not has_tune:                      # la nappe jurerait avec une mélodie dans une autre tonalité
        p = pad(duration, root, rng, level=0.05)
        mix[: len(p)] += p
    mix = _lowpass(_reverb(mix, rng, style["reverb"] * (0.5 if has_beat else 1)), 9000 if has_beat else 5000)
    peak = np.percentile(np.abs(mix), 99.95) + 1e-9
    mix = np.tanh(mix / peak * 0.6) * 0.85
    pcm = (mix * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    return style
