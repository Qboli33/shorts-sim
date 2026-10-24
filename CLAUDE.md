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
- Une mécanique ne dépasse jamais 30 % d'un lot de 10 : `quotas()` refuse le lot s'il n'y a pas assez de mécaniques (il en faut au moins 4).
- `--sfx meow,boing,classic` impose les packs sonores en cycle ; sinon tirage aléatoire pondéré (`audio.PACKS`, miaou majoritaire).

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
- `generator/mechanics/<nom>.py` : `NAME`, `TITLES`, `DESCRIPTIONS`, `HASHTAGS`, `sample(rng)`, `simulate(seed, params) -> Timeline|None`, `score(tl)`, `comment(tl, rng)`. À enregistrer dans `mechanics/__init__.py`.
- La durée cible est garantie par construction : l'événement final doit tomber entre 26 et 53 s de simulation. S'y ajoutent 2,2 s de ralenti, 1 s de suite et 1,5 s de carton, soit une vidéo de 30 à 58 s.
- Suspense : les 1,2 s de simulation avant l'événement final passent au ralenti (x0,35, positions interpolées), avec un assombrissement des bords et des battements de cœur.

## Mécaniques
| Nom | Principe | Fins | Astuce durée |
|---|---|---|---|
| grow_ball | grossit de `growth` px à chaque rebond ; variante `escape` avec ouverture tournante | FULL! / ESCAPED! / TOO BIG! | `growth` réajusté jusqu'à 6 fois (escape : 30 sous-graines) |
| color_escape | chaque évasion = +1 point et 2 balles de cette couleur (plafond par couleur pour éviter l'emballement) | « X WINS! » | N (premier à N) choisi après la simulation pour une fin dans la fenêtre, serrée de préférence |
| merge_balls | deux objets de même niveau fusionnent (planètes de la Lune au Soleil, dessinées par code dans `generator/objects.py`) ; rangée de niveaux « ? » | « JUPITER! » / FULL! | niveau objectif = le plus haut atteint dans la fenêtre (minimum 5) |
| bounce_spawn | chaque rebond de la balle blanche « mère » ajoute une balle | « 128! » | objectif (50…200) choisi après la simulation |

Score qualité : durée proche de 42 s, fin serrée, changements de leader, type de fin. Pour color_escape, on favorise fortement la victoire d'un point et un dernier changement de leader tardif (paramètre `cap` bas pour éviter l'emballement).

## Style (retour utilisateur du 2026-10-24)
- **Texte à l'écran quasi absent** : `overlay` très court ou vide (`OVERLAYS` dans chaque mécanique ; vide une fois sur deux). Interface réduite à un chiffre (%, compteur, scores) ; texte final court (FULL!, OUT!, STUCK!, MAX!, RED!, 128!) ; pas de sous-titre. Le titre YouTube (`title`) garde la question en anglais.
- **Sons rigolos** synthétisés (aucun échantillon) : miaou, boing, coin-coin, couic, de plus en plus graves à l'approche de la fin (jusqu'à -1,6 octave). « classic » reste possible mais rare.
- **Suspense jusqu'au bout** : ralenti final, fins serrées.
- **Fin en boucle** : les 0,45 dernières secondes du carton se fondent dans la première image (`LOOP_FADE`), pour que la relecture enchaîne sans coupure.
- **Premier commentaire automatique** : `comment(tl, rng)` dans chaque mécanique, une question adaptée à la vidéo (champ `first_comment` du log), envoyée comme `firstCommentText` à Metricool. Metricool ne peut pas l'épingler.
- **Image de chaîne** : `python3 -m generator.branding` génère `assets/branding/avatar.png` et `banner.png`. Textes de chaîne dans `assets/branding/channel.md`.
- **Objets plutôt que balles pour la fusion** (demande du 2026-10-24) : série tirée au hasard dans `objects.SETS`, soit des planètes procédurales, soit des chats de dessin animé dessinés par code (du chaton au King Cat, avec un miaou dont la hauteur suit la taille). Dans grow_ball, 35 % des vidéos utilisent un chat. Chute lente (gravité de 650 à 1100).
- Mèmes et sons viraux d'autres créateurs : refusés (droits d'auteur et contenu réutilisé non monétisable). On recrée l'esprit par code. Idées de formats : `state/ideas.md`. D'autres séries d'objets dessinées par code pourront s'ajouter (gemmes, fruits stylisés…), sans aucune marque ni personnage.

## Métadonnées
Titre anglais en question ou défi (dans `TITLES`, cohérent avec ce qui est affiché). Description : 1 ligne + hashtags (`#shorts #satisfying #simulation …`).
Metricool : type short, privacy public, category GAMING, isAiGeneratedContent false, madeForKids false.

## Publication
- `state/settings.json` : `autopublish` false = posts créés avec createScheduledPostForReview (validation par l'utilisateur dans Metricool) ; true = createScheduledPost direct.
- Pousser les vidéos par paquets d'un jour (environ 10 vidéos par commit et par push) pour éviter les refus de push trop gros.
- Depuis le 2026-10-24, le mode auto refuse toute création de post Metricool (en validation, brouillon ou direct). On génère donc `python3 -m generator.metricool_csv --date <date>`, qui écrit `/mnt/project-files/metricool/<date>.csv`, et l'utilisateur l'importe lui-même (calendrier > Import CSV). Ne jamais passer par le navigateur ou une session remote pour contourner ce refus (l'utilisateur l'a demandé, et on a refusé).
- **Mode actuel (choix de l'utilisateur du 2026-10-24)** : publication manuelle dans YouTube Studio (Metricool gratuit plafonne à 20 vidéos par mois). Pour chaque jour du lot, copier les vidéos renommées avec leur titre dans `/mnt/project-files/a-publier/<date>/`, avec une `fiche.md` (heure, description + question du premier commentaire). Guide : `/mnt/project-files/a-publier/guide-youtube-studio.md`.
- 2026-10-24 : Short de test non répertorié programmé (Metricool 405051945). Metricool récupère bien l'URL raw et la recopie sur static.metricool.com.

## Comptes
- Metricool : blogId 7627641, fuseau Asia/Makassar, chaîne YouTube UCMu0izz7oXj-ZFSW08sIyOQ.
- vidIQ : compte connecté, chaîne YouTube non liée au 2026-10-24.
- Média publié = URL raw : `https://raw.githubusercontent.com/Qboli33/shorts-sim/main/<file>` (vérifier HTTP 200 et la taille avant de programmer).

## Tâche planifiée
« Lot hebdo Shorts simulations » : chaque dimanche à 18:47 (Asia/Makassar), elle reprend la session de mise en place et exécute le cycle complet (bilan, mix, 70 Shorts, Metricool en validation, récapitulatif).

## Ménage du dépôt
Supprimer de `out/` les vidéos publiées depuis plus de 14 jours. Au-delà d'environ 1 Go, réécrire l'historique avec une branche orpheline qui ne garde que l'état courant (`git checkout --orphan fresh && git add -A && git commit && git branch -M main && git push -f origin main`), (autorisé par le brief de l'utilisateur).

## Décisions
- 2026-10-24 : OpenCV et Poppins n'étaient pas préinstallés. pip fonctionne : `setup.sh` installe OpenCV, et la police est versionnée dans le dépôt.
- 2026-10-24 : les balles évadées disparaissent avec un pop, au lieu de tomber hors du cercle (plus lisible et moins coûteux).
- 2026-10-24 : les 5 Shorts de test sont dans `out/2026-10-24/` (statut `test` dans le log).
- 2026-10-24 : le mode auto de la session refuse `createScheduledPostForReview` même avec l'accord écrit de l'utilisateur. Règle ajoutée dans `.claude/settings.json` (accord de l'utilisateur), effective dans une nouvelle session. Si c'est encore refusé : écrire la liste des posts dans `/mnt/project-files/metricool/<date>.md` et la donner à l'utilisateur.

## À faire
- bounce_spawn produit des fichiers lourds (environ 40 Mo, avec beaucoup de balles en mouvement). Réduire si le dépôt grossit trop.
- Nouvelles mécaniques (au moins 1 par semaine) : anneaux concentriques qui se brisent, couleurs qui se convertissent, labyrinthe qui rétrécit, etc.
