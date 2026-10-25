"""Format commun entre simulation et rendu.

Une simulation produit une `Timeline` : un état par image (60 i/s), une liste d'événements
(sons + particules) et des éléments d'interface. Le rendu ne refait aucune physique.
"""
from dataclasses import dataclass, field

import numpy as np

W, H = 1080, 1920
SIM_FPS = 60
CENTER = np.array([520.0, 1060.0])   # légèrement à gauche : le bord droit est masqué par l'interface
RING_R = 410.0
SAFE_BOTTOM = H - 300


@dataclass
class Frame:
    balls: np.ndarray            # (n, 3) float32 : x, y, r
    colors: np.ndarray           # (n, 3) uint8 BGR
    uids: np.ndarray             # (n,) int, pour les traînées
    rings: list                  # [(R, gap_phi | None, gap_half, (B,G,R))]
    hud: list = field(default_factory=list)   # commandes d'interface (voir render.draw_hud)
    labels: list | None = None   # texte dans les balles (ex. niveau)
    kinds: np.ndarray | None = None   # niveau / type par balle (objets dessinés à la place des balles)
    angles: np.ndarray | None = None  # rotation propre (rad, sens horaire à l'écran)
    press: float | None = None   # y du bas du plateau de presse (mécanique crusher)
    strings: tuple | None = None  # fils tendus : (ancres (m, 2), uid de la balle tenue (m,), couleurs BGR (m, 3))


@dataclass
class Event:
    frame: int
    kind: str      # bounce | hit | pop | chime | boom | win | spawn
    x: float
    y: float
    color: tuple
    strength: float = 1.0
    note: int = 0  # degré dans la gamme pentatonique


@dataclass
class Timeline:
    mechanic: str
    seed: int
    params: dict
    title: str
    frames: list
    events: list
    finale_frame: int            # image où apparaît le texte final
    finale_text: str
    finale_color: tuple
    end_sub: str = ""            # sous-titre du carton de fin
    metrics: dict = field(default_factory=dict)
    palette: dict = field(default_factory=dict)
    trails: int = 6
    overlay: str = ""            # texte à l'écran (très court, ou vide) ; `title` sert au titre YouTube

    @property
    def duration(self):
        return len(self.frames) / SIM_FPS
