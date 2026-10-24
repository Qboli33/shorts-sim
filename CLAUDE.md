# shorts-sim : chaîne YouTube Shorts de simulations physiques

Dépôt = mémoire du projet. Chaque session repart de zéro : lire ce fichier, `state/log.json` et `state/performance.md`.

## Démarrage de session
```bash
git clone --depth 1 https://github.com/Qboli33/shorts-sim && cd shorts-sim
./setup.sh                    # vérifie numpy/Pillow/OpenCV/ffmpeg ; installe opencv-python-headless si pip marche
```
Fin de session : `git fetch origin main && git rebase origin/main`, puis commit + `git push -u origin main`.

## Lancer un lot
```bash
python3 -m generator.batch --date 2026-10-27 --count 10 --candidates 15
python3 -m generator.batch --date 2026-10-27 --count 10 --weights grow_ball=2,merge_balls=0.5
```
- Simule les candidates en parallèle (tous les cœurs), garde les meilleures selon `score()`, rend, puis ffprobe + planche de 3 images dans `out/<date>/qc/` (non versionnée, à regarder avec Read).
- Ajoute chaque vidéo à `state/log.json` (statut `rendered`, ou `qc_failed`).
- Graine de base = hash de la date (reproductible) ; `--seed` pour forcer. `--index-start` pour numéroter un 2e passage le même jour.
- Une mécanique ne dépasse jamais 30 % d'un lot de 10 : `quotas()` refuse le lot s'il n'y a pas assez de mécaniques. **Avec 3 mécaniques, un lot de 10 est impossible : il en faut au moins 4.**

## Mesures (2026-10-24, 4 CPU, 60 i/s)
- Simulation : 0,3 à 8 s par candidate.
- Rendu : environ 88 s par vidéo de 45 s sur un worker ; 2 workers en parallèle donnent environ 44 s par vidéo. Un lot de 70 prend environ 55 min. Si ça ne tient pas, passer `--fps 30` (rendu environ 2 fois plus rapide).
- Taille : 6 à 22 Mo par vidéo (la course de couleurs avec beaucoup de particules est la plus lourde). 70 vidéos font environ 0,7 Go par semaine.
- Audio : -14 à -19 LUFS, crête à -1,5 dB environ, sans saturation.

## Architecture
- `generator/physics.py` : balles (`Balls`), collisions balle-balle vectorisées (impulsions + correction de position), paroi circulaire avec ouverture (les extrémités de l'arc sont des obstacles ponctuels), détection d'évasion.
- `generator/scene.py` : `Timeline` / `Frame` / `Event`, le format entre simulation et rendu. Simulation toujours à 60 i/s. Cercle centré en (520, 1060), R = 410 (légèrement à gauche, car le bord droit et les 300 px du bas sont masqués par l'interface).
- `generator/render.py` : fond en dégradé avec vignette, anneau néon, halo (couche 1/4 floutée), balles avec reflet, traînées, particules, titre, interface, texte final animé, carton de fin de 1,5 s. Encodage via un tube vers ffmpeg (libx264 veryfast CRF 20, yuv420p, AAC 160k, faststart).
- `generator/audio.py` : rebond, choc, pop, carillon, explosion, arpège de victoire, nappe ; gamme pentatonique avec une tonique aléatoire ; 6 voix max ; normalisation + tanh.
- `generator/text.py` : Poppins Bold (`assets/fonts`, licence OFL ; DejaVu en secours), sprites en cache.
- `generator/mechanics/<nom>.py` : `NAME`, `TITLES`, `DESCRIPTIONS`, `HASHTAGS`, `sample(rng)`, `simulate(seed, params) -> Timeline|None`, `score(tl)`. À enregistrer dans `mechanics/__init__.py`.
- La durée cible est garantie par construction : l'événement final doit tomber entre 27,5 et 54,5 s (puis 1 s de suite + 1,5 s de carton, soit une vidéo de 30 à 57 s).

## Mécaniques
| Nom | Principe | Fins | Astuce durée |
|---|---|---|---|
| grow_ball | grossit de `growth` px à chaque rebond ; variante `escape` avec ouverture tournante | FULL! / ESCAPED! / TOO BIG! | `growth` réajusté jusqu'à 6 fois (escape : 30 sous-graines) |
| color_escape | chaque évasion = +1 point et 2 balles de cette couleur (plafond par couleur pour éviter l'emballement) | « X WINS! » | N (premier à N) choisi après la simulation pour une fin dans la fenêtre, serrée de préférence |
| merge_balls | deux balles de même niveau fusionnent ; rangée de niveaux « ? » | MAX! / FULL! | niveau objectif = le plus haut atteint dans la fenêtre (minimum 5) |

Score qualité : durée proche de 42 s, fin serrée, changements de leader, type de fin.

## Métadonnées
Titre anglais en question ou défi (dans `TITLES`, cohérent avec ce qui est affiché). Description : 1 ligne + hashtags (`#shorts #satisfying #simulation …`).
Metricool : type short, privacy public, category GAMING, isAiGeneratedContent false, madeForKids false.

## Comptes
- Metricool : blogId 7627641, fuseau Asia/Makassar, chaîne YouTube UCMu0izz7oXj-ZFSW08sIyOQ.
- vidIQ : compte connecté, chaîne YouTube non liée au 2026-10-24.
- Média publié = URL raw : `https://raw.githubusercontent.com/Qboli33/shorts-sim/main/<file>` (vérifier HTTP 200 et la taille avant de programmer).

## Ménage du dépôt
Supprimer de `out/` les vidéos publiées depuis plus de 14 jours. Au-delà d'environ 1 Go, réécrire l'historique avec une branche orpheline qui ne garde que l'état courant (`git checkout --orphan fresh && git add -A && git commit && git branch -M main && git push -f origin main`), seulement après accord de l'utilisateur la première fois.

## Décisions
- 2026-10-24 : OpenCV et Poppins n'étaient pas préinstallés. pip fonctionne : `setup.sh` installe OpenCV, et la police est versionnée dans le dépôt.
- 2026-10-24 : les balles évadées disparaissent avec un pop, au lieu de tomber hors du cercle (plus lisible et moins coûteux).
- 2026-10-24 : les 5 Shorts de test sont dans `out/2026-10-24/` (statut `test` dans le log).

## À faire
- Ajouter au moins une 4e mécanique avant le premier lot de 10 (idées : chaque rebond ajoute une balle, anneaux concentriques qui se brisent, couleurs qui se convertissent, labyrinthe qui rétrécit).
