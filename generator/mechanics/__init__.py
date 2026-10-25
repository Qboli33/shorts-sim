"""Registre des mécaniques.

Chaque module expose :
  NAME                  identifiant court
  sample(rng) -> dict   paramètres aléatoires (palette, durées, physique, titre…)
  simulate(seed, params) -> Timeline | None   None si la candidate est rejetée
  score(tl) -> float    qualité mesurable entre 0 et 1 (fin serrée, durée, événement final…)
"""
from . import bounce_spawn, color_escape, crusher, grow_ball, merge_balls, rings_escape, strings

MECHANICS = {m.NAME: m for m in (grow_ball, color_escape, merge_balls, bounce_spawn, crusher, rings_escape, strings)}
