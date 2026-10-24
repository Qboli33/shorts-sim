"""Rendu d'une Timeline en MP4 1080x1920 (OpenCV + Pillow), encodé par ffmpeg via un tube."""
import os
import subprocess
import tempfile
import time

import cv2
import numpy as np

from . import audio, objects, text
from .scene import CENTER, H, SIM_FPS, W, Frame

SH = 4                      # sous-pixel pour cv2 (coordonnées x16)
K = 1 << SH
GS = 4                      # facteur de réduction de la couche de halo
END_CARD = 1.5              # secondes


def _p(x, y):
    return int(round(x * K)), int(round(y * K))


def make_background(theme):
    top, bot = np.array(theme["bg"][0][::-1], float), np.array(theme["bg"][1][::-1], float)
    t = np.linspace(0, 1, H)[:, None, None]
    img = np.repeat(top * (1 - t) + bot * t, W, axis=1)
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt(((xx - CENTER[0]) / W) ** 2 + ((yy - CENTER[1]) / H) ** 2)
    ring = np.array(theme["ring"][::-1], float)
    img += ring * np.exp(-(d / 0.28) ** 2)[..., None] * 0.07     # lueur douce derrière l'anneau
    img *= (1 - 0.35 * np.clip(d - 0.25, 0, 1))[..., None]       # vignette
    return np.clip(img, 0, 255).astype(np.uint8)


def lighten(c, k):
    return tuple(int(v + (255 - v) * k) for v in c)


class Particles:
    def __init__(self, rng):
        self.rng = rng
        self.p = np.zeros((0, 2))
        self.v = np.zeros((0, 2))
        self.life = np.zeros(0)
        self.max_life = np.zeros(0)
        self.col = np.zeros((0, 3))
        self.size = np.zeros(0)

    def burst(self, x, y, color, n, speed, life=0.8, size=5):
        ang = self.rng.uniform(0, 2 * np.pi, n)
        sp = self.rng.uniform(0.3, 1.0, n) * speed
        self.p = np.vstack([self.p, np.tile([x, y], (n, 1))])
        self.v = np.vstack([self.v, np.stack([np.cos(ang) * sp, np.sin(ang) * sp], 1)])
        lf = self.rng.uniform(0.5, 1.0, n) * life
        self.life = np.append(self.life, lf)
        self.max_life = np.append(self.max_life, lf)
        col = np.array(color, float)[None].repeat(n, 0)
        col += self.rng.uniform(0, 80, (n, 1))   # quelques étincelles plus claires
        self.col = np.vstack([self.col, np.clip(col, 0, 255)])
        self.size = np.append(self.size, self.rng.uniform(0.5, 1.0, n) * size)
        if len(self.life) > 1500:                 # plafond
            keep = slice(len(self.life) - 1500, None)
            self.p, self.v, self.life, self.max_life = self.p[keep], self.v[keep], self.life[keep], self.max_life[keep]
            self.col, self.size = self.col[keep], self.size[keep]

    def step(self, dt):
        self.v[:, 1] += 900 * dt
        self.v *= 0.985
        self.p += self.v * dt
        self.life -= dt
        k = self.life > 0
        self.p, self.v, self.life, self.max_life = self.p[k], self.v[k], self.life[k], self.max_life[k]
        self.col, self.size = self.col[k], self.size[k]

    def draw(self, img, glow):
        f = self.life / self.max_life
        for (x, y), c, s, a in zip(self.p, self.col, self.size, f):
            r = max(1.0, s * (0.4 + 0.6 * a))
            cc = tuple(int(v * (0.3 + 0.7 * a)) for v in c)
            cv2.circle(img, _p(x, y), int(r * K), cc, -1, cv2.LINE_AA, SH)
            cv2.circle(glow, _p(x / GS, y / GS), int(max(1, r * 1.5 / GS) * K), cc, -1, cv2.LINE_AA, SH)


def draw_hud(img, cmds):
    for c in cmds:
        kind = c[0]
        if kind == "text":
            _, s, x, y, size, col, anchor = c[:7]
            stroke = c[7] if len(c) > 7 else max(2, size // 14)
            text.blit(img, text.sprite(s, size, col, stroke), x, y, anchor)
        elif kind == "circle":
            _, x, y, r, col, filled = c
            cv2.circle(img, _p(x, y), int(r * K), col, -1 if filled else 3, cv2.LINE_AA, SH)
            if filled:
                cv2.circle(img, _p(x - r * .3, y - r * .3), int(r * .3 * K), lighten(col, .55), -1, cv2.LINE_AA, SH)
        elif kind == "qmark":
            _, x, y, r = c
            cv2.circle(img, _p(x, y), int(r * K), (90, 90, 100), 3, cv2.LINE_AA, SH)
            text.blit(img, text.sprite("?", int(r * 1.2), (170, 170, 180), 0), x, y + 2, "c")
        elif kind == "obj":
            _, oset, level, x, y, r = c
            text.blit(img, objects.get(oset, int(level), int(r)), x, y, "c")
        elif kind == "bar":
            _, x, y, w, h, col, frac = c
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (40, 40, 48), -1)
            cv2.rectangle(img, (int(x), int(y)), (int(x + w * min(1, frac)), int(y + h)), col, -1)


SLOW_SIM = 1.2      # secondes de simulation passées au ralenti avant l'événement final
SLOW_SPEED = 0.35
_yy, _xx = np.mgrid[0:H, 0:W]
VIGNETTE = (1 - 0.45 * np.clip(np.hypot((_xx - W / 2) / W, (_yy - H / 2) / H) * 1.6 - 0.35, 0, 1))[..., None].astype(np.float32)
del _yy, _xx


def _interp(frames, sf):
    """Interpole positions et rotation entre deux images de simulation (pour un ralenti fluide)."""
    i = int(sf)
    a = sf - i
    f0 = frames[i]
    if a < 1e-6 or i + 1 >= len(frames):
        return f0
    f1 = frames[i + 1]
    if len(f0.uids) != len(f1.uids) or not np.array_equal(f0.uids, f1.uids):
        return f0
    rings = [(r0[0], None if r0[1] is None else r0[1] + (r1[1] - r0[1]) * a, r0[2], r0[3])
             for r0, r1 in zip(f0.rings, f1.rings)]
    return Frame(balls=f0.balls + (f1.balls - f0.balls) * a, colors=f0.colors, uids=f0.uids, rings=rings,
                 hud=f0.hud, labels=f0.labels, kinds=f0.kinds,
                 angles=None if f0.angles is None else f0.angles + (f1.angles - f0.angles) * a)


def _uid_maps(frames):
    return [{int(u): i for i, u in enumerate(f.uids)} for f in frames]


def render(tl, out_path, fps=60, theme=None, log=print):
    """Rend la timeline ; retourne le temps de rendu en secondes."""
    t_start = time.time()
    theme = theme or tl.palette
    rng = np.random.default_rng(tl.seed + 7)
    bg = make_background(theme)
    n_sim = len(tl.frames)
    n_end = int(END_CARD * fps)
    # Ralenti juste avant l'événement final : position (fractionnaire) dans la simulation pour chaque image.
    fin = tl.finale_frame
    s0 = max(0, fin - int(SLOW_SIM * SIM_FPS))
    sched, sf = [], 0.0
    while sf < n_sim - 1:
        sched.append(sf)
        sf += SIM_FPS / fps * (SLOW_SPEED if s0 <= sf < fin else 1.0)
    n_out = len(sched) + n_end
    duration = n_out / fps
    slow_t = (s0 / SIM_FPS, s0 / SIM_FPS + (fin - s0) / SIM_FPS / SLOW_SPEED)

    def time_of(f):
        if f < s0:
            return f / SIM_FPS
        if f < fin:
            return slow_t[0] + (f - s0) / SIM_FPS / SLOW_SPEED
        return slow_t[1] + (f - fin) / SIM_FPS

    if tl.params.get("objects") == "cats":
        tl.params["sfx"] = "meow"            # chats => miaous, plus graves quand le chat est plus gros
    pack = tl.params.setdefault("sfx", audio.PACKS[int(rng.integers(len(audio.PACKS)))])
    tmp = tempfile.mkdtemp(prefix="sim_")
    wav = os.path.join(tmp, "a.wav")
    events = list(tl.events)
    audio.render_audio(events, duration, time_of, tl.params.get("root", 60), tl.seed, wav, pack=pack,
                       finale_frame=fin, slow=slow_t, pitch_by_note=tl.params.get("objects") == "cats")

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    uid_maps = _uid_maps(tl.frames)
    ev_by_frame = {}
    for e in events:
        ev_by_frame.setdefault(e.frame, []).append(e)
    parts = Particles(rng)
    overlay = tl.overlay
    title_sprites = [text.sprite(l, 84, (255, 255, 255), 7, (20, 10, 10)) for l in text.wrap(overlay, 84, 900)] if overlay else []
    fs = text.fit_size(tl.finale_text, 150, 900)
    finale_spr = text.sprite(tl.finale_text, fs, tl.finale_color, max(6, fs // 12), (10, 10, 10))
    sub_spr = None     # pas de sous-titre : écran le plus épuré possible
    flash = 0.0
    last = None

    prev_sf = -1.0
    for k in range(n_out):
        in_end = k >= len(sched)
        if not in_end:
            sf = sched[k]
            f_idx = int(sf)
            speed = SLOW_SPEED if s0 <= sf < fin else 1.0
        if in_end:
            # Carton de fin : dernière image figée, assombrie, texte final.
            e = (k - (n_out - n_end)) / max(1, n_end)
            img = (last * (1 - 0.55 * min(1, e * 3))).astype(np.uint8)
            text.blit(img, finale_spr, 540, 960, "c", scale=1.0 + 0.04 * np.sin(e * 6))
            if sub_spr:
                text.blit(img, sub_spr, 540, 1110, "c", alpha=min(1, e * 4))
            proc.stdin.write(img.tobytes())
            continue

        fr = _interp(tl.frames, sf)
        img = bg.copy()
        glow = np.zeros((H // GS, W // GS, 3), np.uint8)

        for (R, gphi, ghalf, col) in fr.rings:
            th = 12
            rr = R + th / 2
            if gphi is None or ghalf <= 0:
                cv2.circle(img, _p(*CENTER), int(rr * K), col, th, cv2.LINE_AA, SH)
                cv2.circle(glow, _p(*(CENTER / GS)), int(rr / GS * K), col, 5, cv2.LINE_AA, SH)
            else:
                a0 = np.degrees(gphi + ghalf)
                a1 = np.degrees(gphi - ghalf + 2 * np.pi)
                cv2.ellipse(img, _p(*CENTER), (int(rr * K), int(rr * K)), 0, a0, a1, col, th, cv2.LINE_AA, SH)
                cv2.ellipse(glow, _p(*(CENTER / GS)), (int(rr / GS * K), int(rr / GS * K)), 0, a0, a1, col, 5, cv2.LINE_AA, SH)
            if gphi is None or ghalf <= 0:
                cv2.circle(img, _p(*CENTER), int((rr - th / 2 + 1) * K), lighten(col, .6), 2, cv2.LINE_AA, SH)
            else:
                ri = int((rr - th / 2 + 1) * K)
                cv2.ellipse(img, _p(*CENTER), (ri, ri), 0, a0, a1, lighten(col, .6), 2, cv2.LINE_AA, SH)

        # Traînées : positions passées, couleur fondue vers le fond.
        if tl.trails:
            for back in range(tl.trails, 0, -1):
                pf = f_idx - back * 2
                if pf < 0:
                    continue
                prev, pmap = tl.frames[pf], uid_maps[pf]
                a = 0.45 * (1 - back / (tl.trails + 1))
                for i, u in enumerate(fr.uids):
                    j = pmap.get(int(u))
                    if j is None:
                        continue
                    x, y, r = prev.balls[j]
                    bgc = bg[int(min(H - 1, max(0, y))), int(min(W - 1, max(0, x)))]
                    c = fr.colors[i]
                    cc = tuple(int(c[m] * a + bgc[m] * (1 - a)) for m in range(3))
                    cv2.circle(img, _p(x, y), int(r * (0.55 + 0.4 * (1 - back / tl.trails)) * K), cc, -1, cv2.LINE_AA, SH)

        oset = tl.params.get("objects")
        use_obj = oset in objects.SETS and fr.kinds is not None
        for idx, ((x, y, r), c) in enumerate(zip(fr.balls, fr.colors)):
            c = tuple(int(v) for v in c)
            if use_obj:
                lvl = int(fr.kinds[idx])
                if oset == "planets" and lvl == 9:      # seul le Soleil brille ; sinon le halo délave les dessins
                    cv2.circle(glow, _p(x / GS, y / GS), int(min(r * 1.15, r + 30) / GS * K), tuple(int(v * .5) for v in c), -1, cv2.LINE_AA, SH)
                spr = objects.get(oset, lvl, int(round(r)))
                if fr.angles is not None:
                    spr = objects.rotate(spr, float(fr.angles[idx]))
                text.blit(img, spr, x, y, "c")
                continue
            cv2.circle(glow, _p(x / GS, y / GS), int(min(r * 1.25, r + 40) / GS * K), c, -1, cv2.LINE_AA, SH)
            cv2.circle(img, _p(x, y), int(r * K), tuple(int(v * .7) for v in c), -1, cv2.LINE_AA, SH)
            cv2.circle(img, _p(x - r * .06, y - r * .06), int(r * .9 * K), c, -1, cv2.LINE_AA, SH)
            cv2.circle(img, _p(x - r * .35, y - r * .35), int(max(1, r * .28) * K), lighten(c, .6), -1, cv2.LINE_AA, SH)
            if r > 10:
                cv2.circle(img, _p(x - r * .42, y - r * .42), int(max(1, r * .1) * K), (255, 255, 255), -1, cv2.LINE_AA, SH)
        if fr.labels:
            for (x, y, r), lab in zip(fr.balls, fr.labels):
                if lab and r >= 26:
                    text.blit(img, text.sprite(lab, int(r * 0.8), (255, 255, 255), max(2, int(r / 14))), x, y, "c")

        # Particules générées par les événements du pas (et des pas sautés à 30 i/s).
        for ef in range(int(prev_sf) + 1, f_idx + 1):
            for e in ev_by_frame.get(ef, ()):
                if e.kind == "bounce":
                    parts.burst(e.x, e.y, e.color, 6, 260, 0.4, 4)
                elif e.kind == "pop":
                    parts.burst(e.x, e.y, e.color, 16, 520, 0.6, 6)
                elif e.kind == "chime":
                    parts.burst(e.x, e.y, e.color, 18, 380, 0.9, 5)
                elif e.kind in ("boom", "win"):
                    parts.burst(e.x, e.y, e.color, 260, 1500, 1.6, 9)
                    flash = 1.0
        prev_sf = sf
        parts.step(speed / fps)
        parts.draw(img, glow)

        glow = cv2.GaussianBlur(glow, (0, 0), 5)
        img = cv2.add(img, cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR))
        if flash > 0.02:
            img = cv2.addWeighted(img, 1.0, np.full_like(img, 255), flash * 0.45, 0)
            flash *= 0.85

        y = 230
        for s in title_sprites:
            text.blit(img, s, 520, y, "c")
            y += 100
        if speed < 1:
            # Ralenti : léger assombrissement des bords pour la tension.
            img = (img * VIGNETTE).astype(np.uint8)
        draw_hud(img, fr.hud)

        if f_idx >= tl.finale_frame:
            e = (f_idx - tl.finale_frame) / SIM_FPS
            sc = 0.3 + 0.85 * min(1, e / 0.18) - 0.15 * max(0, min(1, (e - 0.18) / 0.15))
            text.blit(img, finale_spr, 540, 960, "c", scale=sc)
        last = img
        proc.stdin.write(img.tobytes())

    proc.stdin.close()
    rc = proc.wait()
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    if rc != 0:
        raise RuntimeError(f"ffmpeg a échoué ({rc}) pour {out_path}")
    el = time.time() - t_start
    log(f"rendu {os.path.basename(out_path)} : {duration:.1f}s de vidéo en {el:.0f}s")
    return el
