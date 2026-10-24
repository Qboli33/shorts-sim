"""Physique maison en numpy : balles dans un cercle, ouverture tournante, collisions vectorisées.

Unités : pixels et secondes, repère image (y vers le bas).
"""
import numpy as np

TAU = 2 * np.pi


def wrap_angle(a):
    return (a + np.pi) % TAU - np.pi


class Balls:
    """Ensemble de balles à taille variable (ajout / retrait en cours de simulation)."""

    def __init__(self):
        self.pos = np.zeros((0, 2))
        self.vel = np.zeros((0, 2))
        self.rad = np.zeros(0)
        self.kind = np.zeros(0, dtype=int)   # couleur / niveau selon la mécanique
        self.uid = np.zeros(0, dtype=int)    # identifiant stable (traînées)
        self.ang = np.zeros(0)                # angle de rotation propre (rad, sens horaire à l'écran)
        self.spin = np.zeros(0)               # vitesse de rotation (rad/s)
        self._next = 0

    def __len__(self):
        return len(self.rad)

    def add(self, pos, vel, rad, kind, spin=0.0, ang=0.0):
        self.pos = np.vstack([self.pos, np.asarray(pos, float).reshape(1, 2)])
        self.vel = np.vstack([self.vel, np.asarray(vel, float).reshape(1, 2)])
        self.rad = np.append(self.rad, float(rad))
        self.kind = np.append(self.kind, int(kind))
        self.uid = np.append(self.uid, self._next)
        self.ang = np.append(self.ang, float(ang))
        self.spin = np.append(self.spin, float(spin))
        self._next += 1
        return len(self.rad) - 1

    def remove(self, idx):
        keep = np.ones(len(self.rad), bool)
        keep[np.atleast_1d(idx)] = False
        self.pos, self.vel = self.pos[keep], self.vel[keep]
        self.rad, self.kind, self.uid = self.rad[keep], self.kind[keep], self.uid[keep]
        self.ang, self.spin = self.ang[keep], self.spin[keep]

    def advance_spin(self, dt):
        self.ang += self.spin * dt

    @property
    def mass(self):
        return self.rad ** 2


def collide_balls(b, e=0.95, correct=0.8, min_speed_event=60.0, touching=None):
    """Collisions balle-balle (impulsions + correction de position).

    Retourne (i, j, vitesse_relative) des chocs assez forts pour un son.
    `touching` : ensemble de paires déjà en contact au pas précédent (pas de son répété).
    """
    n = len(b)
    if n < 2:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    d = b.pos[:, None, :] - b.pos[None, :, :]
    dist2 = (d ** 2).sum(-1)
    rs = b.rad[:, None] + b.rad[None, :]
    mask = np.triu(dist2 < rs ** 2, 1)
    i, j = np.nonzero(mask)
    if len(i) == 0:
        return i, j, np.zeros(0)
    # Frottement entre balles en contact : la rotation s'amortit dans les tas.
    b.spin[np.unique(np.concatenate([i, j]))] *= 0.985
    dist = np.sqrt(dist2[i, j]) + 1e-9
    nrm = d[i, j] / dist[:, None]
    m = b.mass
    inv_i, inv_j = 1.0 / m[i], 1.0 / m[j]
    vn = ((b.vel[i] - b.vel[j]) * nrm).sum(-1)
    approaching = vn < 0
    jimp = np.where(approaching, -(1 + e) * vn / (inv_i + inv_j), 0.0)
    np.add.at(b.vel, i, (jimp * inv_i)[:, None] * nrm)
    np.add.at(b.vel, j, -(jimp * inv_j)[:, None] * nrm)
    overlap = rs[i, j] - dist
    w_i = inv_i / (inv_i + inv_j)
    push = (overlap * correct)[:, None] * nrm
    np.add.at(b.pos, i, push * w_i[:, None])
    np.add.at(b.pos, j, -push * (1 - w_i)[:, None])
    strong = approaching & (-vn > min_speed_event)
    return i[strong], j[strong], -vn[strong]


def collide_ring(b, center, R, e=1.0, gap_phi=None, gap_half=0.0, active=None):
    """Collision avec la paroi intérieure d'un cercle de rayon R (ouverture optionnelle).

    Retourne (indices des balles qui ont rebondi, vitesse normale, indices dans l'ouverture).
    """
    n = len(b)
    if n == 0:
        z = np.zeros(0, int)
        return z, np.zeros(0), z
    act = np.ones(n, bool) if active is None else active
    rel = b.pos - center
    dist = np.sqrt((rel ** 2).sum(-1)) + 1e-9
    nrm = rel / dist[:, None]
    in_gap = np.zeros(n, bool)
    if gap_phi is not None and gap_half > 0:
        theta = np.arctan2(rel[:, 1], rel[:, 0])
        dth = np.abs(wrap_angle(theta - gap_phi))
        in_gap = dth < gap_half
        # Les deux extrémités de l'arc sont des obstacles ponctuels.
        for sgn in (-1, 1):
            p = center + R * np.array([np.cos(gap_phi + sgn * gap_half), np.sin(gap_phi + sgn * gap_half)])
            dd = b.pos - p
            dl = np.sqrt((dd ** 2).sum(-1)) + 1e-9
            hit = act & in_gap & (dl < b.rad)
            if hit.any():
                nn = dd[hit] / dl[hit, None]
                vn = (b.vel[hit] * nn).sum(-1)
                b.vel[hit] -= ((1 + e) * np.minimum(vn, 0))[:, None] * nn
                b.pos[hit] = p + nn * b.rad[hit, None]
    hit = act & ~in_gap & (dist + b.rad > R)
    idx = np.nonzero(hit)[0]
    if len(idx) == 0:
        return idx, np.zeros(0), np.nonzero(in_gap & act)[0]
    nn = nrm[idx]
    vn = (b.vel[idx] * nn).sum(-1)
    out = vn > 0
    # Roulement contre la paroi : la rotation tend vers v_tangentielle / r (comme une vraie balle).
    tan = np.stack([-nn[:, 1], nn[:, 0]], 1)
    vt = (b.vel[idx] * tan).sum(-1)
    b.spin[idx] += 0.5 * (-vt / b.rad[idx] - b.spin[idx])
    b.vel[idx] -= ((1 + e) * np.where(out, vn, 0))[:, None] * nn
    b.pos[idx] = center + nn * (R - b.rad[idx])[:, None]
    return idx[out], vn[out], np.nonzero(in_gap & act)[0]


def escaped(b, center, R):
    rel = b.pos - center
    dist = np.sqrt((rel ** 2).sum(-1))
    return np.nonzero(dist - b.rad > R + 2)[0]


def rotate(v, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def near_pairs(b, pad=1.0):
    """Paires candidates (i < j) pour beaucoup de balles : arbre k-d (scipy), sinon méthode directe."""
    n = len(b)
    if n < 2:
        z = np.zeros(0, int)
        return z, z
    if n < 120:
        d2 = ((b.pos[:, None] - b.pos[None]) ** 2).sum(-1)
        rs = b.rad[:, None] + b.rad[None] + pad
        return np.nonzero(np.triu(d2 < rs ** 2, 1))
    from scipy.spatial import cKDTree
    pr = cKDTree(b.pos).query_pairs(2 * b.rad.max() + pad, output_type="ndarray")
    if len(pr) == 0:
        z = np.zeros(0, int)
        return z, z
    return pr[:, 0], pr[:, 1]


def collide_pairs(b, i, j, e=0.2, correct=0.8):
    """Comme collide_balls, sur des paires données. Retourne la somme des recouvrements de chaque balle (sa « pression »)."""
    over = np.zeros(len(b))
    if len(i) == 0:
        return over
    d = b.pos[i] - b.pos[j]
    dist = np.sqrt((d ** 2).sum(-1)) + 1e-9
    rs = b.rad[i] + b.rad[j]
    hit = dist < rs
    i, j, d, dist, rs = i[hit], j[hit], d[hit], dist[hit], rs[hit]
    if len(i) == 0:
        return over
    nrm = d / dist[:, None]
    m = b.mass
    inv_i, inv_j = 1.0 / m[i], 1.0 / m[j]
    vn = ((b.vel[i] - b.vel[j]) * nrm).sum(-1)
    jimp = np.where(vn < 0, -(1 + e) * vn / (inv_i + inv_j), 0.0)
    n = len(b)
    ov = rs - dist
    w_i = inv_i / (inv_i + inv_j)
    dv_i, dv_j = (jimp * inv_i)[:, None] * nrm, -(jimp * inv_j)[:, None] * nrm
    push = (ov * correct)[:, None] * nrm
    dp_i, dp_j = push * w_i[:, None], -push * (1 - w_i)[:, None]
    for k in (0, 1):     # np.bincount : bien plus rapide que np.add.at
        b.vel[:, k] += np.bincount(i, dv_i[:, k], n) + np.bincount(j, dv_j[:, k], n)
        b.pos[:, k] += np.bincount(i, dp_i[:, k], n) + np.bincount(j, dp_j[:, k], n)
    # Recouvrement par balle : la somme des recouvrements (pression) plutôt que le max.
    return np.bincount(i, ov, n) + np.bincount(j, ov, n)
