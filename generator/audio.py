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


def render_audio(events, duration, fps, root, seed, path, max_voices=6, min_gap=0.035):
    rng = np.random.default_rng(seed)
    n = int(SR * duration) + 1
    mix = np.zeros((n, 2))
    cache = {}
    active_until = []
    last_t = {}
    for ev in sorted(events, key=lambda e: e.frame):
        t0 = ev.frame / fps
        # Limiter la densité : un même type de son ne se répète pas trop vite.
        if ev.kind in ("bounce", "hit", "pop") and t0 - last_t.get(ev.kind, -1) < min_gap:
            continue
        active_until = [u for u in active_until if u > t0]
        if ev.kind in ("bounce", "hit", "pop") and len(active_until) >= max_voices:
            continue
        key = (ev.kind, ev.note)
        if key not in cache:
            f = note_freq(root, ev.note)
            cache[key] = {
                "bounce": lambda: _bounce(f), "hit": lambda: _hit(f), "pop": lambda: _pop(f, rng),
                "chime": lambda: _chime(f), "boom": lambda: _boom(f, rng), "win": lambda: _win(root),
                "spawn": lambda: _hit(f * 2) * 0.5,
            }[ev.kind]()
        w = cache[key] * min(1.0, max(0.15, ev.strength))
        s = int(t0 * SR)
        m = min(len(w), n - s)
        if m <= 0:
            continue
        pan = np.clip((ev.x - 540) / 540, -1, 1) * 0.5
        mix[s:s + m, 0] += w[:m] * (1 - pan) * 0.7
        mix[s:s + m, 1] += w[:m] * (1 + pan) * 0.7
        active_until.append(t0 + len(w) / SR * 0.5)
        last_t[ev.kind] = t0
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
