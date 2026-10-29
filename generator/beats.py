"""Beats originaux style trap / phonk / drill, générés par code (demande de vassili du 2026-10-26).

Aucun morceau existant n'est repris : chaque vidéo tire un nouveau refrain (motif de 2 mesures sur une grille de
doubles croches, joué en boucle sur une suite d'accords), une batterie et une basse 808. Le refrain arrive seul,
puis le beat « tombe » au bout de 2 mesures.
"""
import numpy as np

STYLES = {
    # gamme, plage de tempo, suites d'accords (degrés en demi-tons sous la tonique), mélodie
    "phonk": dict(scale=[0, 1, 3, 5, 7, 8, 10], bpm=(128, 148), prog=[[0, 0, 8, 10], [0, 8, 5, 7], [0, 10, 8, 7]],
                  lead="cowbell", name="PHONK!"),
    "trap": dict(scale=[0, 2, 3, 5, 7, 8, 10], bpm=(132, 150), prog=[[0, 8, 3, 10], [0, 5, 8, 7], [0, 8, 10, 7]],
                 lead="soft", name="TRAP BEAT!"),
    "drill": dict(scale=[0, 2, 3, 5, 7, 8, 11], bpm=(138, 146), prog=[[0, 8, 5, 7], [0, 3, 8, 7], [0, 0, 8, 11]],
                  lead="soft", name="DRILL BEAT!"),
}
RHYTHMS = {      # positions (doubles croches) du motif sur 2 mesures
    "phonk": [[0, 3, 6, 8, 11, 14, 16, 19, 22, 24, 26, 28], [0, 3, 6, 10, 12, 16, 19, 22, 26, 28, 30],
              [0, 2, 3, 6, 8, 11, 14, 16, 18, 19, 22, 24]],
    "trap": [[0, 2, 4, 7, 10, 12, 16, 18, 20, 23, 26, 28, 30], [0, 3, 6, 8, 12, 14, 16, 19, 22, 24, 28],
             [0, 4, 6, 7, 10, 12, 16, 20, 22, 23, 26, 28]],
    "drill": [[0, 3, 6, 9, 12, 14, 16, 19, 22, 25, 28], [0, 3, 6, 8, 11, 14, 16, 19, 22, 24, 27, 30],
              [0, 2, 6, 9, 12, 16, 18, 22, 25, 28]],
}
DRUMS = {        # par mesure de 16 doubles croches
    "phonk": dict(kick=[0, 7, 10], snare=[4, 12], hat=list(range(0, 16, 2))),
    "trap": dict(kick=[0, 10], snare=[8], hat=list(range(0, 16, 2))),
    "drill": dict(kick=[0, 11], snare=[8, 14], hat=[0, 3, 6, 8, 10, 12, 14]),
}


def make(seed, style, t0=0.8, target=(32.0, 44.0)):
    rng = np.random.default_rng(seed)
    st = STYLES[style]
    bpm = float(rng.uniform(*st["bpm"]))
    s16 = 60.0 / bpm / 4
    bar = 16 * s16
    k = int(np.clip(round(np.mean(target) / (4 * bar)), 3, 8))      # nombre de phrases de 4 mesures
    tonic = int(rng.integers(68, 75))                               # mélodie autour de La4 - Ré5
    scale = st["scale"]
    prog = st["prog"][int(rng.integers(len(st["prog"])))]
    # Motif de 2 mesures : marche aléatoire sur la gamme, attirée par la tonique et la quinte.
    pos = list(RHYTHMS[style][int(rng.integers(len(RHYTHMS[style])))])
    for _ in range(int(rng.integers(0, 2))):
        if len(pos) > 8:
            pos.pop(int(rng.integers(1, len(pos) - 1)))
    deg, degs = 0, []
    for i in range(len(pos)):
        step = int(rng.choice([-2, -1, -1, 0, 1, 1, 2, 3]))
        deg = int(np.clip(deg + step, -2, 8))
        if i == 0 or rng.random() < 0.2:
            deg = int(rng.choice([0, 4, 7]))
        degs.append(deg)
    answer = list(degs)                     # 2e moitié de la phrase : même début, fin qui retombe
    answer[-3:] = [int(np.clip(d - 1 - j, -2, 8)) for j, d in enumerate(answer[-3:])]
    answer[-1] = 0

    def midi(d):
        o, s = divmod(d, len(scale))
        return tonic + 12 * o + scale[s]

    notes, drums, bass = [], [], []
    n_bars = 4 * k
    for b in range(0, n_bars, 2):
        motif = degs if (b // 2) % 2 == 0 else answer
        for p_, d in zip(pos, motif):
            notes.append((t0 + b * bar + p_ * s16, midi(d)))
    notes.append((t0 + n_bars * bar, tonic))             # note finale sur le premier temps suivant
    pat = DRUMS[style]
    for b in range(n_bars):
        tb = t0 + b * bar
        root = 36 + (tonic - 12 * 3 + prog[b % 4]) % 12      # basse 808 entre Do2 et Si2
        if b < 2:                                            # intro : refrain seul, puis le beat tombe
            continue
        for s in pat["kick"]:
            drums.append((tb + s * s16, "kick"))
            bass.append((tb + s * s16, root))
        for s in pat["snare"]:
            drums.append((tb + s * s16, "clap"))
        hats = list(pat["hat"])
        if b % 2 == 1 and rng.random() < 0.7:               # roulement de charleston en fin de mesure
            hats = [h for h in hats if h < 12] + [12, 13, 14, 15]
        for s in hats:
            drums.append((tb + s * s16, "hat"))
    tend = t0 + n_bars * bar
    drums.append((tend, "kick"))
    bass.append((tend, 36 + (tonic - 36) % 12))
    return dict(style=style, name=st["name"], bpm=bpm, notes=notes, drums=drums, bass=bass, lead=st["lead"])


# 2026-10-29 : « grooves » = seulement le tempo et la grille rythmique d'un style (demande de vassili : le rythme de ses
# morceaux, jamais leur son ni leur mélodie). La balle rebondit sur les temps forts ; batterie et basse sont synthétisées.
# `hits` = (position en doubles croches, note MIDI) ; la hauteur sert à étager les plateformes, ce n'est pas une mélodie.
GROOVES = {
    "swing": dict(name="SWING!", bpm=94, kick=[0, 7, 10], snare=[4, 12], hat=list(range(0, 16, 2)),
                  hits=[(0, 55), (4, 67), (7, 55), (10, 61), (12, 67)], root=38),
    "edm": dict(name="DROP!", bpm=128, kick=[0, 4, 8, 12], snare=[4, 12], hat=[2, 6, 10, 14],
                hits=[(0, 55), (4, 67), (8, 61), (12, 67)], root=36),
    "dembow": dict(name="DEMBOW!", bpm=96, kick=[0, 4, 8, 12], snare=[3, 6, 11, 14], hat=list(range(0, 16, 2)),
                   hits=[(0, 55), (3, 67), (6, 61), (8, 55), (11, 67), (14, 61)], root=38),
    "reggae": dict(name="REGGAE!", bpm=106, kick=[0, 8], snare=[4, 12], hat=list(range(0, 16, 2)),
                   hits=[(0, 55), (4, 67), (8, 55), (12, 67)], root=40),
}


def make_groove(style, t0=0.8, target=40.0):
    g = GROOVES[style]
    s16 = 60.0 / g["bpm"] / 4
    bar = 16 * s16
    n_bars = max(4, int(round(target / bar)))
    notes, drums, bass = [], [], []
    for b in range(n_bars):
        tb = t0 + b * bar
        for step, m in g["hits"]:
            notes.append((tb + step * s16, m))
        for step in g["kick"]:
            drums.append((tb + step * s16, "kick"))
            bass.append((tb + step * s16, g["root"]))
        for step in g["snare"]:
            drums.append((tb + step * s16, "clap"))
        for step in g["hat"]:
            drums.append((tb + step * s16, "hat"))
    tend = t0 + n_bars * bar
    notes.append((tend, g["hits"][0][1]))
    drums.append((tend, "kick"))
    bass.append((tend, g["root"]))
    return dict(style=style, name=g["name"], bpm=float(g["bpm"]), notes=notes, drums=drums, bass=bass, lead="soft")
