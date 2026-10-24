"""Objets dessinés par code (aucune image externe) pour remplacer les balles : planètes du plus petit au plus gros.

Chaque objet est une sphère ombrée avec une texture procédurale (cratères, bandes, continents, anneaux).
Sprites BGR + alpha mis en cache par (niveau, rayon).
"""
from functools import lru_cache

import numpy as np

PLANETS = ["MOON", "MERCURY", "MARS", "VENUS", "EARTH", "NEPTUNE", "URANUS", "SATURN", "JUPITER", "SUN"]
# Couleur dominante (BGR) : particules, icônes, sons.
PLANET_COLORS = [(200, 200, 200), (140, 150, 165), (60, 90, 210), (150, 210, 235), (210, 140, 60),
                 (230, 120, 50), (225, 220, 140), (130, 190, 225), (110, 160, 215), (40, 200, 255)]


def _bands(ny, cols, freq, rng, wobble=0.0, nx=None):
    t = ny * freq + (wobble * np.sin(nx * 6 + rng.uniform(0, 6)) if nx is not None else 0)
    k = (np.floor(t) % len(cols)).astype(int)
    out = np.array(cols, float)[k]
    return out


def _spots(nx, ny, base, spots):
    img = base.copy()
    for (cx, cy, r, col, soft) in spots:
        d = np.hypot(nx - cx, ny - cy) / r
        a = np.clip((1 - d) / soft, 0, 1)[..., None]
        img = img * (1 - a) + np.array(col, float) * a
    return img


@lru_cache(maxsize=512)
def sprite(level, r):
    """Sprite de l'objet `level` de rayon `r` (px). Saturne déborde (anneaux) : sprite plus large."""
    r = max(4, int(r))
    rng = np.random.default_rng(100 + level)
    ring = level == 7
    half = int(r * (1.9 if ring else 1.25 if level == 9 else 1.0)) + 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    nx, ny = xx / r, yy / r
    d = np.hypot(nx, ny)
    inside = d < 1
    nz = np.sqrt(np.clip(1 - d ** 2, 0, 1))
    light = np.clip(-0.45 * nx - 0.55 * ny + 0.7 * nz, 0, 1)
    shade = (0.28 + 0.85 * light)[..., None]
    if level in (0, 1):            # Lune, Mercure : cratères
        base = np.full(nx.shape + (3,), [185, 185, 190] if level == 0 else [130, 140, 155], float)
        spots = [(rng.uniform(-.8, .8), rng.uniform(-.8, .8), rng.uniform(.08, .25),
                  [120, 120, 128] if level == 0 else [90, 100, 112], 0.4) for _ in range(9)]
        tex = _spots(nx, ny, base, spots)
    elif level == 2:               # Mars
        base = np.full(nx.shape + (3,), [50, 85, 205], float)
        tex = _spots(nx, ny, base, [(rng.uniform(-.7, .7), rng.uniform(-.6, .7), rng.uniform(.2, .4), [35, 55, 140], .8) for _ in range(5)]
                     + [(0, -0.95, 0.35, [230, 235, 245], 0.3)])
    elif level == 3:               # Vénus : voiles
        tex = _bands(ny, [[140, 200, 230], [160, 215, 240], [125, 185, 220]], 3.5, rng, 0.6, nx)
    elif level == 4:               # Terre : océans, continents, nuages
        base = np.full(nx.shape + (3,), [200, 120, 40], float)
        land = [(rng.uniform(-.7, .7), rng.uniform(-.6, .6), rng.uniform(.25, .45), [70, 160, 70], .5) for _ in range(5)]
        clouds = [(rng.uniform(-.8, .8), rng.uniform(-.8, .8), rng.uniform(.1, .25), [245, 245, 245], .9) for _ in range(6)]
        tex = _spots(nx, ny, base, land + clouds)
    elif level == 5:               # Neptune
        tex = _bands(ny, [[220, 110, 40], [235, 130, 60], [210, 100, 35]], 4, rng, 0.2, nx)
        tex = _spots(nx, ny, tex, [(0.3, 0.2, 0.18, [140, 60, 20], 0.6)])
    elif level == 6:               # Uranus
        tex = _bands(ny, [[225, 220, 150], [230, 228, 165]], 6, rng)
    elif level == 7:               # Saturne
        tex = _bands(ny, [[130, 190, 225], [110, 170, 210], [150, 205, 235], [120, 180, 215]], 6, rng, 0.1, nx)
    elif level == 8:               # Jupiter : bandes + grande tache
        tex = _bands(ny, [[150, 190, 225], [90, 140, 200], [200, 220, 240], [70, 120, 180]], 7, rng, 0.25, nx)
        tex = _spots(nx, ny, tex, [(0.35, 0.35, 0.2, [60, 80, 200], 0.5)])
    else:                          # Soleil
        tex = np.full(nx.shape + (3,), [40, 200, 255], float)
        tex = _spots(nx, ny, tex, [(rng.uniform(-.7, .7), rng.uniform(-.7, .7), rng.uniform(.1, .25), [30, 150, 255], .8) for _ in range(10)])
        shade = (0.85 + 0.25 * light)[..., None]
    rgb = tex * shade
    if level == 9:
        rgb = rgb + 60 * (1 - d)[..., None].clip(0, 1)
    alpha = np.clip((1 - d) * r, 0, 1)
    if level == 9:                 # couronne
        corona = np.clip(1 - (d - 1) / 0.25, 0, 1) * (d >= 1) * 0.55
        rgb = np.where(inside[..., None], rgb, np.array([60, 190, 255], float))
        alpha = np.maximum(alpha, corona)
    if ring:                       # anneaux de Saturne : ellipse aplatie, devant la planète en bas
        ex, ey = nx / 1.8, ny / 0.5
        er = np.hypot(ex, ey)
        ring_a = ((er > 0.62) & (er < 1.0)).astype(float) * 0.9
        ring_a *= 1 - 0.4 * ((er > 0.78) & (er < 0.82))
        front = (ny > 0) | ~inside
        ring_a = ring_a * front
        rc = np.array([150, 200, 225], float) * (0.6 + 0.4 * (1 - er))[..., None]
        rgb = rgb * (1 - ring_a[..., None]) + rc * ring_a[..., None]
        alpha = np.maximum(alpha, ring_a)
    return np.clip(rgb, 0, 255).astype(np.uint8), alpha.astype(np.float32)
