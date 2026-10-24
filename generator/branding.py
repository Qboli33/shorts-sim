"""Avatar et bannière de la chaîne, générés avec le même style que les vidéos.

    python3 -m generator.branding   -> assets/branding/avatar.png (800x800) et banner.png (2560x1440)
"""
import os

import cv2
import numpy as np

from . import objects, text

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "branding")
NAME = "Bounce Lab"
TAGLINE = "Satisfying physics sims  •  New Shorts every day"


def _bg(w, h, top=(40, 10, 14), bot=(10, 2, 4)):
    t = np.linspace(0, 1, h)[:, None, None]
    img = np.repeat(np.array(top, float) * (1 - t) + np.array(bot, float) * t, w, axis=1)
    return img.astype(np.uint8)


def _neon_ring(img, c, r, col, th):
    glow = np.zeros_like(img)
    cv2.circle(glow, c, r, col, th * 4, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), th * 2)
    img[:] = cv2.add(img, glow)
    cv2.circle(img, c, r, col, th, cv2.LINE_AA)
    cv2.circle(img, c, r - th // 2, (235, 250, 255), max(1, th // 4), cv2.LINE_AA)


def avatar():
    img = _bg(800, 800, (60, 14, 30), (14, 4, 10))
    _neon_ring(img, (400, 400), 330, (230, 220, 90), 16)
    for (x, y, r, col) in [(250, 560, 34, (90, 90, 255)), (560, 590, 28, (60, 220, 120)), (600, 250, 22, (255, 160, 70))]:
        cv2.circle(img, (x, y), r, col, -1, cv2.LINE_AA)
        cv2.circle(img, (x - r // 3, y - r // 3), r // 4, (255, 255, 255), -1, cv2.LINE_AA)
    text.blit(img, objects.rotate(objects.cat_sprite(0, 170), 0.25), 400, 400)
    return img


def banner():
    W, H = 2560, 1440
    img = _bg(W, H, (46, 12, 26), (8, 2, 6))
    rng = np.random.default_rng(3)
    _neon_ring(img, (W // 2, H // 2), 600, (230, 220, 90), 10)
    # objets éparpillés hors de la zone sûre centrale (1546x423), qui roulent dans des angles variés
    for k in range(26):
        x, y = rng.uniform(80, W - 80), rng.uniform(120, H - 120)
        if abs(x - W / 2) < 820 and abs(y - H / 2) < 260:
            continue
        lvl, r = int(rng.integers(10)), int(rng.uniform(40, 85))
        spr = objects.cat_sprite(lvl, r) if k % 2 else objects.sprite(lvl, r)
        text.blit(img, objects.rotate(spr, rng.uniform(-0.6, 0.6)), x, y)
    text.blit(img, text.sprite(NAME, 170, (255, 255, 255), 12, (30, 10, 20)), W // 2, H // 2 - 70)
    text.blit(img, text.sprite(TAGLINE, 64, (120, 230, 255), 6, (30, 10, 20)), W // 2, H // 2 + 95)
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    cv2.imwrite(os.path.join(OUT, "avatar.png"), avatar())
    cv2.imwrite(os.path.join(OUT, "banner.png"), banner())
    print("ok", OUT)


if __name__ == "__main__":
    main()
