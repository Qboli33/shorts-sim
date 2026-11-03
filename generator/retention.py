"""Shorts « rétention » (consignes de vassili du 2026-11-03) : 12 à 22 s, accroche dès la 1re image, boucle.

    python3 -m generator.retention --out-dir out/essai-2

Chaque vidéo de SPECS fixe sa mécanique, sa palette, la forme des objets, l'instrument, le type d'accroche et sa
durée cible (au moins 3 paramètres différents d'une vidéo à l'autre). Pour chaque vidéo, on simule des graines en
parallèle avec une fenêtre de fin resserrée, on garde la meilleure, on rend en mode rétention, puis
`check()` applique la checklist (durée, 60 i/s, débit, audio, aucun temps mort > 1 s, raccord de boucle).
"""
import argparse
import json
import os
import subprocess
from multiprocessing import Pool

import numpy as np

from . import qc
from .batch import ROOT, load_log, save_log
from .mechanics import MECHANICS, common
from .palettes import THEMES, bgr
from .render import render

SLOW_SIM, SLOW_SPEED, TAIL = 0.8, 0.4, common.TAIL
EXTRA = SLOW_SIM * (1 / SLOW_SPEED - 1) + TAIL       # durée vidéo = fin de simulation + EXTRA

SPECS = [
    dict(key="cat-fill", mechanic="grow_ball", D=12.8, theme="violet", inst="kalimba", hook_type="question",
         want=lambda p: p["variant"] == "fill", force=dict(objects="cats", cat_level=3), shape="chat",
         hook=lambda tl: "Will the cat fill the circle?"),
    dict(key="color-race", mechanic="color_escape", D=15.4, theme="ember", inst="marimba", hook_type="compte",
         want=lambda p: len(p["colors"]) in (3, 4), force={}, shape="balles",
         hook=lambda tl: f"{len(tl.params['colors'])} colors. 1 exit."),
    dict(key="bounce-spawn", mechanic="bounce_spawn", D=13.8, theme="deepsea", inst="felt", hook_type="défi",
         want=lambda p: True, force={}, shape="balles (dégradé)",
         hook=lambda tl: f"It needs {tl.metrics['N']} balls to win"),
    dict(key="planet-merge", mechanic="merge_balls", D=14.6, theme="midnight", inst="bloop", hook_type="pari",
         want=lambda p: True, force=dict(objects="planets"), shape="planètes",
         hook=lambda tl: "Wait for the last merge..."),
]


def _window(D):
    t = D - EXTRA
    common.TARGET = t
    for m in MECHANICS.values():
        for k, v in (("MIN_FINALE", t - 0.7), ("MAX_FINALE", t + 0.7), ("TARGET", t)):
            if hasattr(m, k):
                setattr(m, k, v)


def _params(spec, seed):
    m = MECHANICS[spec["mechanic"]]
    p = m.sample(np.random.default_rng(seed))
    if not spec["want"](p):
        return None
    p["theme"] = dict(next(t for t in THEMES if t["name"] == spec["theme"]))
    p.update(spec["force"])
    p["sfx"] = "soft:" + spec["inst"]
    return p


def _sim(job):
    i, seed = job
    spec = SPECS[i]
    _window(spec["D"])
    p = _params(spec, seed)
    if p is None:
        return None
    try:
        tl = MECHANICS[spec["mechanic"]].simulate(seed, p)
    except Exception as e:
        return dict(seed=seed, error=repr(e))
    if tl is None:
        return None
    return dict(seed=seed, score=float(MECHANICS[spec["mechanic"]].score(tl)), fin=tl.finale_frame / 60)


def _render(job):
    i, seed, path = job
    spec = SPECS[i]
    _window(spec["D"])
    m = MECHANICS[spec["mechanic"]]
    p = _params(spec, seed)
    tl = m.simulate(seed, p)
    hook = spec["hook"](tl)
    render(tl, path, fps=60, log=lambda *_: None,
           retention=dict(hook=hook, zoom=0.07, bed=True, slow_sim=SLOW_SIM, slow_speed=SLOW_SPEED))
    return dict(params=tl.params, metrics=tl.metrics, hook=hook, finale_s=tl.finale_frame / 60 + SLOW_SIM * (1 / SLOW_SPEED - 1),
                finale_text=tl.finale_text, comment=m.comment(tl, np.random.default_rng(seed)) if hasattr(m, "comment") else "")


def _gray_frames(path, fps=10, size=(108, 192)):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-vf", f"fps={fps},scale={size[0]}:{size[1]},format=gray",
                          "-f", "rawvideo", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, size[1], size[0]).astype(np.float32)


def check(path, finale_s):
    """Checklist automatique ; renvoie (ok, détails)."""
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                                     capture_output=True, text=True).stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = [s for s in info["streams"] if s["codec_type"] == "audio"]
    dur = float(info["format"]["duration"])
    vbr = int(v.get("bit_rate", 0)) / 1e6
    fr = _gray_frames(path)
    diff = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
    still = 0
    longest = 0
    for d in diff:
        still = still + 1 if d < 0.25 else 0
        longest = max(longest, still)
    last = subprocess.run(["ffmpeg", "-loglevel", "error", "-sseof", "-0.05", "-i", path, "-vf", "scale=108:192,format=gray",
                           "-f", "rawvideo", "-"], capture_output=True).stdout
    last = np.frombuffer(last, np.uint8).reshape(-1, 192, 108).astype(np.float32)[-1]
    loop_err = float(np.abs(last - fr[0]).mean())        # dernière image décodée contre la première
    first_mean = float(fr[0].mean())
    vol = subprocess.run(["ffmpeg", "-i", path, "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True).stderr
    mean_db = float(vol.split("mean_volume:")[1].split("dB")[0]) if "mean_volume:" in vol else -99
    d = dict(
        format=(v["width"], v["height"]) == (1080, 1920) and v["codec_name"] == "h264" and v["r_frame_rate"] == "60/1",
        bitrate_ok=vbr >= 12, video_mbps=round(vbr, 1),
        audio=bool(a) and a[0]["codec_name"] == "aac" and int(a[0].get("bit_rate", 0)) >= 180000 and mean_db > -40,
        audio_kbps=int(a[0].get("bit_rate", 0)) // 1000 if a else 0, mean_db=mean_db,
        duration_ok=12 <= dur <= 22, duration=round(dur, 2),
        no_dead_time=longest < 10, longest_still_s=round(longest / 10, 1),
        climax_last_2s=dur - finale_s <= 2.0, climax_from_end=round(dur - finale_s, 2),
        loop_ok=loop_err < 2.0, loop_err=round(loop_err, 2),
        first_frame_not_black=first_mean > 8, size_mb=round(int(info["format"]["size"]) / 1e6, 1),
    )
    ok = all(d[k] for k in ("format", "bitrate_ok", "audio", "duration_ok", "no_dead_time", "climax_last_2s", "loop_ok",
                            "first_frame_not_black"))
    return ok, d


def combo(e):
    r = e.get("retention", {})
    p = e.get("params", {})
    return (e.get("mechanic"), (p.get("theme") or {}).get("name"), p.get("objects"), str(p.get("sfx")),
            r.get("hook_type"), round(e.get("duration", 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="out/essai-2")
    ap.add_argument("--candidates", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    idx = [i for i, s in enumerate(SPECS) if not args.only or s["key"] in args.only.split(",")]
    jobs = [(i, args.seed + 1000 * i + j) for i in idx for j in range(args.candidates)]
    with Pool(os.cpu_count() or 2) as pool:
        res = pool.map(_sim, jobs)
    picks = {}
    for (i, _), r in zip(jobs, res):
        if r and "error" not in r and (i not in picks or r["score"] > picks[i]["score"]):
            picks[i] = r
    for i in idx:
        print(SPECS[i]["key"], "->", picks.get(i), f"({sum(1 for (j, _), r in zip(jobs, res) if j == i and r and 'error' not in r)} valides)")
    rjobs = [(i, picks[i]["seed"], os.path.join(out_dir, f"{k + 1:02d}-{SPECS[i]['key']}.mp4")) for k, i in enumerate(idx) if i in picks]
    with Pool(2) as pool:
        rres = pool.map(_render, rjobs)
    log = load_log()
    recent = {combo(e) for e in log[-30:]}
    for (i, seed, path), rr in zip(rjobs, rres):
        spec = SPECS[i]
        ok, d = check(path, rr["finale_s"])
        entry = dict(id=f"essai2-{os.path.basename(path)[:-4]}", mechanic=spec["mechanic"], seed=seed, params=rr["params"],
                     metrics=rr["metrics"], duration=d["duration"], file=os.path.relpath(path, ROOT), status="essai",
                     created="2026-11-03", first_comment=rr["comment"],
                     retention=dict(hook=rr["hook"], hook_type=spec["hook_type"], shape=spec["shape"], checks=d, ok=ok))
        d["new_combo"] = combo(entry) not in recent
        print(f"{'OK ' if ok and d['new_combo'] else 'NON'} {os.path.basename(path)} « {rr['hook']} » -> {rr['finale_text']}  {d}")
        qc.contact_sheet(path, os.path.join(out_dir, "qc", os.path.basename(path)[:-4] + ".jpg"), rr["finale_s"])
        log = [e for e in log if e.get("id") != entry["id"]] + [entry]
    save_log(log)


if __name__ == "__main__":
    main()
