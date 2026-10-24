"""Génère un lot de Shorts : simulation des candidates -> sélection -> rendu -> contrôle -> log.

Exemple :
    python3 -m generator.batch --date 2026-10-27 --count 10 --candidates 15
    python3 -m generator.batch --date 2026-10-27 --count 10 --weights grow_ball=2,merge_balls=1
"""
import argparse
import hashlib
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np

from . import qc
from .mechanics import MECHANICS
from .render import END_CARD, SLOW_SIM, SLOW_SPEED, render

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "state", "log.json")
RAW = "https://raw.githubusercontent.com/Qboli33/shorts-sim/main/"
MAX_SHARE = 0.3      # une mécanique ne dépasse jamais 3 vidéos sur 10


def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def load_log():
    if os.path.exists(LOG):
        with open(LOG) as f:
            return json.load(f)
    return []


def save_log(entries):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w") as f:
        json.dump(entries, f, indent=1, default=_jsonable, ensure_ascii=False)


def quotas(count, weights):
    """Répartit `count` vidéos selon les poids, chaque mécanique plafonnée à MAX_SHARE."""
    cap = max(1, math.floor(count * MAX_SHARE)) if count >= 10 else max(1, math.ceil(count * 0.4))
    names = [n for n in weights if weights[n] > 0]
    if cap * len(names) < count:
        raise SystemExit(f"Impossible : {len(names)} mécaniques x {cap} max < {count} vidéos. Ajouter des mécaniques.")
    total = sum(weights[n] for n in names)
    q = {n: min(cap, int(count * weights[n] / total)) for n in names}
    while sum(q.values()) < count:
        n = max((n for n in names if q[n] < cap), key=lambda n: weights[n] / total * count - q[n])
        q[n] += 1
    return q


def _simulate(job):
    name, seed = job
    m = MECHANICS[name]
    try:
        tl = m.simulate(seed, m.sample(np.random.default_rng(seed)))
    except Exception as e:     # une candidate cassée ne bloque pas le lot
        return dict(mechanic=name, seed=seed, ok=False, error=repr(e))
    if tl is None:
        return dict(mechanic=name, seed=seed, ok=False)
    return dict(mechanic=name, seed=seed, ok=True, score=float(m.score(tl)), title=tl.title,
                duration=tl.duration + END_CARD + SLOW_SIM * (1 / SLOW_SPEED - 1), metrics=tl.metrics)


def _render(job):
    name, seed, path, fps, sfx = job
    m = MECHANICS[name]
    p = m.sample(np.random.default_rng(seed))
    if sfx:
        p["sfx"] = sfx
    tl = m.simulate(seed, p)
    el = render(tl, path, fps=fps, log=lambda *_: None)
    rng = np.random.default_rng(seed + 3)
    descs, tags = m.DESCRIPTIONS, m.HASHTAGS
    if isinstance(descs, dict):          # textes propres à la série d'objets (planètes, chats…)
        descs = descs[tl.params["objects"]]
    if isinstance(tags, dict):
        tags = tags[tl.params["objects"]]
    desc = f"{descs[rng.integers(len(descs))]} {' '.join(tags)}"
    first_comment = m.comment(tl, rng) if hasattr(m, "comment") else ""
    return dict(render_s=round(el, 1), params=tl.params, description=desc, first_comment=first_comment,
                finale_s=tl.finale_frame / 60 + SLOW_SIM * (1 / SLOW_SPEED - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--candidates", type=int, default=15)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--weights", default="")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--index-start", type=int, default=1)
    ap.add_argument("--sfx", default="", help="packs sonores imposés, en cycle (ex. meow,boing,classic)")
    args = ap.parse_args()

    weights = {n: 1.0 for n in MECHANICS}
    for kv in filter(None, args.weights.split(",")):
        k, v = kv.split("=")
        weights[k] = float(v)
    q = quotas(args.count, weights)
    base = args.seed if args.seed is not None else int(hashlib.sha1(args.date.encode()).hexdigest()[:7], 16)
    ratio = args.candidates / args.count
    jobs = []
    for i, (n, k) in enumerate(q.items()):
        for j in range(max(k + 1, math.ceil(k * ratio))):
            jobs.append((n, base + 1000 * i + j))
    t0 = time.time()
    with Pool(os.cpu_count() or 2) as pool:
        res = pool.map(_simulate, jobs)
    print(f"{len(jobs)} candidates simulées en {time.time() - t0:.0f}s ; {sum(r['ok'] for r in res)} valides")
    for r in res:
        if r.get("error"):
            print("  erreur :", r["mechanic"], r["seed"], r["error"])

    chosen, used_titles = [], set()
    for n, k in q.items():
        ok = sorted((r for r in res if r["ok"] and r["mechanic"] == n), key=lambda r: -r["score"])
        # Éviter deux titres identiques dans le même lot quand c'est possible.
        pick = []
        for r in ok:
            if len(pick) < k and r["title"] not in used_titles:
                pick.append(r)
                used_titles.add(r["title"])
        pick += [r for r in ok if r not in pick][: k - len(pick)]
        chosen += pick
    if len(chosen) < args.count:
        print(f"Attention : seulement {len(chosen)} candidates valides pour {args.count} demandées.")
    # Mélanger l'ordre pour ne jamais enchaîner deux fois la même mécanique si possible.
    rng = np.random.default_rng(base)
    order = []
    pool_ = list(rng.permutation(len(chosen)))
    while pool_:
        idx = next((i for i in pool_ if not order or chosen[i]["mechanic"] != chosen[order[-1]]["mechanic"]), pool_[0])
        order.append(idx)
        pool_.remove(idx)
    chosen = [chosen[i] for i in order]

    sfx_cycle = [x for x in args.sfx.split(",") if x]
    out_dir = os.path.join(ROOT, "out", args.date)
    os.makedirs(out_dir, exist_ok=True)
    rjobs = []
    for k, r in enumerate(chosen):
        r["id"] = f"{args.date}-{args.index_start + k:02d}-{r['mechanic']}"
        r["file"] = f"out/{args.date}/{r['id']}.mp4"
        sfx = sfx_cycle[k % len(sfx_cycle)] if sfx_cycle else None
        rjobs.append((r["mechanic"], r["seed"], os.path.join(ROOT, r["file"]), args.fps, sfx))
    t0 = time.time()
    with Pool(args.workers) as pool:
        rres = pool.map(_render, rjobs)
    total = time.time() - t0
    print(f"{len(rjobs)} vidéos rendues en {total:.0f}s ({total / max(1, len(rjobs)):.0f}s par vidéo en moyenne, {args.workers} en parallèle)")

    log = load_log()
    qc_dir = os.path.join(out_dir, "qc")
    for r, rr in zip(chosen, rres):
        path = os.path.join(ROOT, r["file"])
        check = qc.probe(path)
        sheet = qc.contact_sheet(path, os.path.join(qc_dir, r["id"] + ".jpg"), rr["finale_s"])
        status = "rendered" if check["ok"] else "qc_failed"
        print(f"  {r['id']}  {check['duration']:.1f}s  {check['size_mb']:.1f} Mo  score {r['score']:.2f}  {status}  « {r['title']} »"
              + ("" if check["ok"] else f"  PROBLÈME: {check['problems']}"))
        log.append(dict(
            id=r["id"], mechanic=r["mechanic"], seed=r["seed"], params=rr["params"], title=r["title"],
            description=rr["description"], first_comment=rr["first_comment"], duration=round(check["duration"], 2), file=r["file"],
            url=RAW + r["file"], size_bytes=check["size_bytes"], score=round(r["score"], 3), metrics=r["metrics"],
            render_s=rr["render_s"], fps=args.fps, scheduled_at=None, metricool_id=None, status=status,
            qc_sheet=os.path.relpath(sheet, ROOT) if sheet else None, created=args.date,
        ))
    save_log(log)


if __name__ == "__main__":
    main()
