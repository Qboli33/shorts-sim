"""Objets dessinés par code (aucune image externe) pour remplacer les balles.

Séries : planètes (Lune -> Soleil, sphères à texture procédurale) et chats de dessin animé
(têtes de chat avec pelage et expression différents à chaque niveau, du chaton au roi).
Sprites BGR + alpha mis en cache par (série, niveau, rayon).
"""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

SETS = ["planets", "cats"]

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


def get(set_name, level, r):
    return cat_sprite(level, r) if set_name == "cats" else sprite(level, r)


def rotate(spr, ang):
    """Tourne un sprite autour du centre de l'objet (le centre du sprite). `ang` en rad, sens horaire."""
    import cv2
    deg = -np.degrees(ang) % 360
    if deg < 0.5 or deg > 359.5:
        return spr
    rgb, a = spr
    h, w = a.shape
    s = int(np.ceil(np.hypot(h, w))) | 1
    py, px = (s - h) // 2, (s - w) // 2
    rgb2 = np.zeros((s, s, 3), np.uint8)
    a2 = np.zeros((s, s), np.float32)
    rgb2[py:py + h, px:px + w] = rgb
    a2[py:py + h, px:px + w] = a
    M = cv2.getRotationMatrix2D((s / 2 - 0.5, s / 2 - 0.5), deg, 1.0)
    return (cv2.warpAffine(rgb2, M, (s, s), flags=cv2.INTER_LINEAR),
            cv2.warpAffine(a2, M, (s, s), flags=cv2.INTER_LINEAR))


def names(set_name):
    return CATS if set_name == "cats" else PLANETS


def colors(set_name):
    return CAT_COLORS if set_name == "cats" else PLANET_COLORS


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


# ---------------------------------------------------------------- chats
CATS = ["KITTEN", "SNOWBALL", "GINGER", "SMUG CAT", "SAD CAT", "ANGRY CAT", "LOVE CAT", "SLEEPY CHONK",
        "COOL CAT", "KING CAT"]
# pelage (RGB), motif, expression
_CAT_FUR = [
    ((250, 240, 225), None, "kitten"), ((245, 245, 250), None, "happy"), ((245, 150, 60), "stripes", "shock"),
    ((60, 60, 70), None, "smug"), ((150, 160, 175), None, "sad"), ((235, 225, 205), "siamese", "angry"),
    ((250, 245, 240), "calico", "love"), ((175, 135, 95), "stripes", "sleepy"), ((240, 170, 80), "stripes", "cool"),
    ((255, 210, 90), None, "king"),
]
CAT_COLORS = [(c[2], c[1], c[0]) for c, _, _ in _CAT_FUR]


@lru_cache(maxsize=512)
def cat_sprite(level, r):
    """Tête de chat de rayon `r` (la tête = le cercle physique ; oreilles et couronne débordent au-dessus)."""
    r = max(6, int(r))
    S = 3                                   # sur-échantillonnage pour l'anticrénelage
    R = r * S
    W = int(R * 2.3)
    Hh = int(R * 2.9)
    cx, cy = W / 2, Hh - R * 1.15
    img = Image.new("RGBA", (W, Hh), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fur, pattern, expr = _CAT_FUR[level % len(_CAT_FUR)]
    dark = tuple(int(v * 0.55) for v in fur)
    pink = (255, 150, 170)
    P = lambda x, y: (cx + x * R, cy + y * R)
    lw = max(2, int(R * 0.06))
    # oreilles
    ear_col = (90, 60, 50) if pattern == "siamese" else fur
    for sg in (-1, 1):
        d.polygon([P(sg * 0.92, -0.35), P(sg * 0.78, -1.3), P(sg * 0.2, -0.88)], fill=ear_col, outline=dark, width=lw)
        d.polygon([P(sg * 0.75, -0.5), P(sg * 0.7, -1.05), P(sg * 0.38, -0.82)], fill=pink)
    # tête
    d.ellipse([P(-1, -1), P(1, 1)], fill=fur, outline=dark, width=lw)
    if pattern == "stripes":
        for k, x in enumerate((-0.3, 0, 0.3)):
            d.line([P(x, -0.98 + 0.05 * abs(k - 1)), P(x * 0.8, -0.6)], fill=dark, width=int(lw * 1.6))
        for sg in (-1, 1):
            d.line([P(sg * 0.98, 0.0), P(sg * 0.7, 0.05)], fill=dark, width=int(lw * 1.4))
            d.line([P(sg * 0.95, 0.25), P(sg * 0.7, 0.22)], fill=dark, width=int(lw * 1.4))
    elif pattern == "calico":
        d.pieslice([P(-1, -1), P(1, 1)], 200, 290, fill=(240, 150, 60))
        d.pieslice([P(-1, -1), P(1, 1)], 290, 340, fill=(60, 50, 50))
        d.ellipse([P(-1, -1), P(1, 1)], outline=dark, width=lw)
    elif pattern == "siamese":
        d.ellipse([P(-0.55, -0.15), P(0.55, 0.75)], fill=(120, 90, 75))
    # joues
    for sg in (-1, 1):
        d.ellipse([P(sg * 0.62 - 0.17, 0.25), P(sg * 0.62 + 0.17, 0.42)], fill=(255, 170, 185))
    eye_y = -0.12
    black = (25, 20, 25)
    white = (255, 255, 255)
    for sg in (-1, 1):
        ex = sg * 0.4
        if expr in ("kitten", "king"):
            d.ellipse([P(ex - 0.2, eye_y - 0.24), P(ex + 0.2, eye_y + 0.2)], fill=black)
            d.ellipse([P(ex - 0.1, eye_y - 0.18), P(ex + 0.02, eye_y - 0.06)], fill=white)
            d.ellipse([P(ex + 0.04, eye_y), P(ex + 0.11, eye_y + 0.07)], fill=white)
        elif expr == "happy":
            d.arc([P(ex - 0.18, eye_y - 0.12), P(ex + 0.18, eye_y + 0.2)], 200, 340, fill=black, width=int(lw * 1.6))
        elif expr == "shock":
            d.ellipse([P(ex - 0.22, eye_y - 0.25), P(ex + 0.22, eye_y + 0.2)], fill=white, outline=black, width=lw)
            d.ellipse([P(ex - 0.05, eye_y - 0.07), P(ex + 0.05, eye_y + 0.03)], fill=black)
        elif expr == "smug":
            d.chord([P(ex - 0.2, eye_y - 0.15), P(ex + 0.2, eye_y + 0.15)], 0, 180, fill=(230, 200, 60))
            d.line([P(ex - 0.22, eye_y), P(ex + 0.22, eye_y)], fill=black, width=int(lw * 1.5))
            d.ellipse([P(ex - 0.03, eye_y), P(ex + 0.03, eye_y + 0.13)], fill=black)
        elif expr == "sad":
            d.ellipse([P(ex - 0.2, eye_y - 0.2), P(ex + 0.2, eye_y + 0.2)], fill=black)
            d.ellipse([P(ex - 0.12, eye_y - 0.15), P(ex - 0.01, eye_y - 0.04)], fill=white)
            d.line([P(ex - sg * 0.22, eye_y - 0.34), P(ex + sg * 0.15, eye_y - 0.26)], fill=black, width=lw)
            d.ellipse([P(ex - 0.08, eye_y + 0.12), P(ex + 0.08, eye_y + 0.45)], fill=(110, 190, 255))
        elif expr == "angry":
            d.ellipse([P(ex - 0.17, eye_y - 0.12), P(ex + 0.17, eye_y + 0.16)], fill=(240, 220, 60), outline=black, width=lw)
            d.ellipse([P(ex - 0.04, eye_y - 0.1), P(ex + 0.04, eye_y + 0.14)], fill=black)
            d.line([P(ex - sg * 0.25, eye_y - 0.3), P(ex + sg * 0.15, eye_y - 0.12)], fill=black, width=int(lw * 1.8))
        elif expr == "love":
            hx, hy, hs = ex, eye_y, 0.17
            d.ellipse([P(hx - hs, hy - hs * 0.9), P(hx, hy + hs * 0.1)], fill=(240, 50, 90))
            d.ellipse([P(hx, hy - hs * 0.9), P(hx + hs, hy + hs * 0.1)], fill=(240, 50, 90))
            d.polygon([P(hx - hs, hy - hs * 0.35), P(hx + hs, hy - hs * 0.35), P(hx, hy + hs * 1.1)], fill=(240, 50, 90))
        elif expr == "sleepy":
            d.line([P(ex - 0.18, eye_y + 0.02), P(ex + 0.18, eye_y + 0.02)], fill=black, width=int(lw * 1.6))
        elif expr == "cool":
            pass
    if expr == "cool":                      # lunettes de soleil
        d.rounded_rectangle([P(-0.72, eye_y - 0.2), P(-0.08, eye_y + 0.18)], radius=int(R * 0.1), fill=black)
        d.rounded_rectangle([P(0.08, eye_y - 0.2), P(0.72, eye_y + 0.18)], radius=int(R * 0.1), fill=black)
        d.line([P(-0.1, eye_y - 0.12), P(0.1, eye_y - 0.12)], fill=black, width=int(lw * 1.5))
        d.line([P(-0.6, eye_y - 0.12), P(-0.35, eye_y + 0.05)], fill=(120, 120, 140), width=lw)
    if expr == "sleepy":
        zz = [P(0.8, -0.95), P(1.0, -0.95), P(0.8, -0.75), P(1.0, -0.75)]
        d.line(zz, fill=(255, 255, 255), width=lw)
    # nez et bouche
    d.polygon([P(-0.09, 0.14), P(0.09, 0.14), P(0, 0.25)], fill=pink, outline=dark)
    if expr in ("shock", "angry"):
        d.ellipse([P(-0.13, 0.3), P(0.13, 0.62)], fill=(90, 20, 40), outline=black, width=lw)
    elif expr == "sad":
        d.arc([P(-0.18, 0.38), P(0.18, 0.6)], 200, 340, fill=black, width=lw)
    else:
        d.arc([P(-0.2, 0.12), P(0.0, 0.38)], 20, 160, fill=black, width=lw)
        d.arc([P(0.0, 0.12), P(0.2, 0.38)], 20, 160, fill=black, width=lw)
    # moustaches
    for sg in (-1, 1):
        for k, dy in enumerate((-0.06, 0.06, 0.18)):
            d.line([P(sg * 0.32, 0.22 + dy * 0.5), P(sg * 1.12, 0.12 + dy * 1.4)], fill=(60, 55, 60), width=max(1, lw // 2))
    if expr == "king":                      # couronne
        pts = [P(-0.5, -0.92), P(-0.55, -1.45), P(-0.25, -1.15), P(0, -1.55), P(0.25, -1.15), P(0.55, -1.45), P(0.5, -0.92)]
        d.polygon(pts, fill=(255, 200, 40), outline=(180, 120, 20), width=lw)
        for x in (-0.3, 0, 0.3):
            d.ellipse([P(x - 0.06, -1.05), P(x + 0.06, -0.95)], fill=(220, 40, 70))
    # reflet doux
    hl = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(hl).ellipse([P(-0.62, -0.85), P(-0.2, -0.5)], fill=(255, 255, 255, 70))
    img = Image.alpha_composite(img, hl)
    img = img.resize((max(1, W // S), max(1, Hh // S)), Image.LANCZOS)
    a = np.asarray(img).astype(np.float32)
    rgb = np.ascontiguousarray(a[..., 2::-1]).astype(np.uint8)
    alpha = a[..., 3] / 255.0
    # Recentrer : blit centre le sprite ; on ajoute du vide en bas pour que la tête soit au centre.
    h, w = alpha.shape
    head_cy = int(round(cy / S))
    pad = max(0, 2 * head_cy - h)
    if pad:
        rgb = np.vstack([rgb, np.zeros((pad, w, 3), np.uint8)])
        alpha = np.vstack([alpha, np.zeros((pad, w), np.float32)])
    return rgb, alpha
