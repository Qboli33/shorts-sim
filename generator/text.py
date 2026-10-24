"""Texte avec Pillow (Poppins Bold) mis en cache sous forme de sprites BGR + alpha."""
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))
FONT_CANDIDATES = [
    os.path.join(_HERE, "..", "assets", "fonts", "Poppins-Bold.ttf"),
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
FONT_PATH = next(p for p in FONT_CANDIDATES if os.path.exists(p))


@lru_cache(maxsize=64)
def font(size):
    return ImageFont.truetype(FONT_PATH, size)


@lru_cache(maxsize=4096)
def sprite(text, size, color_bgr, stroke=0, stroke_bgr=(0, 0, 0)):
    f = font(size)
    l, t, r, b = f.getbbox(text, stroke_width=stroke)
    w, h = r - l + 4, b - t + 4
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rgb = (color_bgr[2], color_bgr[1], color_bgr[0], 255)
    srgb = (stroke_bgr[2], stroke_bgr[1], stroke_bgr[0], 255)
    d.text((2 - l, 2 - t), text, font=f, fill=rgb, stroke_width=stroke, stroke_fill=srgb)
    a = np.asarray(img)
    return np.ascontiguousarray(a[..., 2::-1]), a[..., 3].astype(np.float32) / 255.0


def fit_size(text, size, max_w):
    while size > 20 and font(size).getlength(text) > max_w:
        size -= 4
    return size


def wrap(text, size, max_w):
    f = font(size)
    words, lines, cur = text.split(), [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        if f.getlength(trial) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def blit(img, spr, x, y, anchor="c", alpha=1.0, scale=1.0):
    """Colle un sprite centré verticalement sur y ; anchor c/l/r horizontal."""
    rgb, a = spr
    if scale != 1.0:
        import cv2
        nw, nh = max(1, int(rgb.shape[1] * scale)), max(1, int(rgb.shape[0] * scale))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_LINEAR)
    h, w = a.shape
    x0 = int(x - w / 2) if anchor == "c" else (int(x) if anchor == "l" else int(x - w))
    y0 = int(y - h / 2)
    H, W = img.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    ex, ey = min(w, W - x0), min(h, H - y0)
    if ex <= sx0 or ey <= sy0:
        return
    roi = img[y0 + sy0:y0 + ey, x0 + sx0:x0 + ex]
    al = (a[sy0:ey, sx0:ex] * alpha)[..., None]
    roi[:] = (roi * (1 - al) + rgb[sy0:ey, sx0:ex] * al).astype(np.uint8)
