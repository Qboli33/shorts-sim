import numpy as np

from ..scene import CENTER, SIM_FPS, Event, Frame

TAIL = 1.0    # secondes de simulation conservées après l'événement final (avant le carton de fin)
MIN_FINALE, MAX_FINALE = 27.5, 54.5    # => vidéo de 30 à 57 s avec le carton de fin
TARGET = 42.0


def snapshot(balls, colors_bgr, rings, hud=None, labels=None):
    return Frame(
        balls=np.column_stack([balls.pos, balls.rad]).astype(np.float32) if len(balls) else np.zeros((0, 3), np.float32),
        colors=np.asarray(colors_bgr, np.uint8).reshape(-1, 3),
        uids=balls.uid.copy(),
        rings=rings,
        hud=hud or [],
        labels=labels,
    )


def wall_point(pos, r):
    d = pos - CENTER
    n = d / (np.linalg.norm(d) + 1e-9)
    return pos + n * r


def duration_score(t):
    return max(0.0, 1 - abs(t - TARGET) / 18)


def frame_of(t):
    return int(round(t * SIM_FPS))


def strip_events_after(events, frame, keep=("boom", "win")):
    return [e for e in events if e.frame <= frame or e.kind in keep]


__all__ = ["Event", "snapshot", "wall_point", "duration_score", "frame_of", "strip_events_after",
           "TAIL", "MIN_FINALE", "MAX_FINALE", "TARGET", "SIM_FPS", "CENTER"]
