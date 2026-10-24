"""Contrôle après rendu : ffprobe (durée, 1080x1920, piste audio) et planche de 3 images."""
import json
import os
import subprocess


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                         capture_output=True, text=True)
    problems = []
    try:
        info = json.loads(out.stdout)
    except json.JSONDecodeError:
        return dict(ok=False, problems=["ffprobe illisible"], duration=0, size_mb=0, size_bytes=0)
    v = [s for s in info.get("streams", []) if s["codec_type"] == "video"]
    a = [s for s in info.get("streams", []) if s["codec_type"] == "audio"]
    dur = float(info.get("format", {}).get("duration", 0))
    size = int(info.get("format", {}).get("size", 0))
    if not v or (v[0]["width"], v[0]["height"]) != (1080, 1920):
        problems.append("résolution")
    if v and v[0].get("pix_fmt") != "yuv420p":
        problems.append("pix_fmt")
    if not a:
        problems.append("pas d'audio")
    if not 30 <= dur <= 58:
        problems.append(f"durée {dur:.1f}s")
    return dict(ok=not problems, problems=problems, duration=dur, size_bytes=size, size_mb=size / 1e6)


def contact_sheet(path, out_jpg, finale_s):
    """3 images : 1 s (accroche), milieu, juste après l'événement final."""
    os.makedirs(os.path.dirname(out_jpg), exist_ok=True)
    ts = [1.0, finale_s / 2, finale_s + 0.3]
    tmp = []
    for k, t in enumerate(ts):
        f = out_jpg + f".{k}.png"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1",
                        "-vf", "scale=360:-1", f], check=False)
        if os.path.exists(f):
            tmp.append(f)
    if len(tmp) != 3:
        return None
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *sum([["-i", f] for f in tmp], []),
                    "-filter_complex", "hstack=3", out_jpg], check=False)
    for f in tmp:
        os.remove(f)
    return out_jpg if os.path.exists(out_jpg) else None
