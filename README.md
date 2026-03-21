# Data Battle — Météorage : Prédiction de fin d'alerte orageuse

## Contexte et objectif

**Météorage** est une entreprise française spécialisée dans la détection de la foudre (filiale de Météo-France). Leur service d'alerte fonctionne ainsi : dès qu'un premier éclair est détecté dans une zone de surveillance autour d'un aéroport, une alerte est déclenchée. Elle est levée **30 minutes après le dernier éclair nuage-sol** détecté dans la zone — c'est la **baseline**.

Le problème : 30 minutes, c'est long. Dans un aéroport, ça signifie 30 minutes d'activité suspendue (pistes, personnel au sol, etc.) après chaque orage.

**L'objectif de ce projet** est de prédire plus tôt la fin réelle de l'orage, afin de lever l'alerte avant la règle des 30 minutes — sans trop sacrifier la sécurité. On formule ça comme un **problème d'analyse de survie** : on cherche à estimer la probabilité que l'alerte soit encore active en fonction des caractéristiques de l'orage observé.

### Métriques clés

| Métrique | Définition | Valeur actuelle |
|---|---|---|
| Gain moyen | Minutes gagnées vs baseline (positif = on lève avant) | +6.6 min |
| Gain médian | Idem, médiane | +17.0 min |
| Faux all-clear | Alertes levées trop tôt (avant la fin réelle de l'orage) | 19% |
| C-index | Capacité du modèle à ordonner les durées (1.0 = parfait) | 0.969 |

---

## Données

Les données ont été fournies par Météorage. Elles couvrent **10 ans d'observations** (2016–2025) dans un rayon de 30 km autour de 6 aéroports :

| Aéroport | Longitude | Latitude |
|---|---|---|
| Ajaccio | 8.8029 | 41.9236 |
| Bastia | 9.4837 | 42.5527 |
| Biarritz | -1.524 | 43.4683 |
| Bron | 4.9389 | 45.7294 |
| Nantes | -1.6107 | 47.1532 |
| Pise | 10.399 | 43.695 |

**Chaque ligne du CSV correspond à un éclair** avec les colonnes suivantes :

| Colonne | Description |
|---|---|
| `date` | Horodatage UTC de l'éclair |
| `lon` / `lat` | Position de l'éclair (degrés décimaux, WGS84) |
| `amplitude` | Polarité et intensité maximale du courant (kA) |
| `maxis` | Erreur de localisation théorique estimée (km) |
| `icloud` | `False` = éclair nuage-sol / `True` = éclair intra-nuage |
| `dist` | Distance à l'aéroport (km) |
| `azimuth` | Direction de l'éclair par rapport à l'aéroport (degrés, 0° = nord) |
| `lightning_id` | Identifiant unique de l'éclair |
| `lightning_airport_id` | Identifiant de l'éclair au sein d'un aéroport |
| `alert_airport_id` | Numéro de l'alerte à laquelle appartient l'éclair |
| `is_last_lightning_cloud_ground` | `True` si c'est le dernier éclair nuage-sol d'une alerte |

> **Note :** `alert_airport_id` et `is_last_lightning_cloud_ground` ne sont renseignés que pour les éclairs à moins de 20 km de l'aéroport.

> **Note :** Les données intra-nuage de Pise pour 2016 sont potentiellement incorrectes — à écarter pour ce type d'analyse.

---

## Structure du projet

```
data_battle/
│
├── data/
│   ├── segment_alerts_all_airports_train.csv   ← fichier brut à placer ici
│   └── features.parquet                        ← généré par features.py
│
├── models/
│   ├── rsf_model.pkl                           ← généré par model.py
│   └── cox_model.pkl                           ← généré par model.py
│
├── outputs/
│   ├── evaluation.png                          ← généré par evaluate.py
│   ├── kaplan_meier.png                        ← généré par model.py
│   ├── prediction_reelle.png                   ← généré par predict.py
│   └── false_allclear_analysis.png             ← généré par analyze_false_allclear.py
│
├── src/
│   ├── features.py                             ← feature engineering
│   ├── model.py                                ← entraînement des modèles
│   ├── predict.py                              ← inférence sur une alerte
│   └── evaluate.py                             ← évaluation complète sur le jeu de test
│
├── analyze_false_allclear.py                   ← analyse des faux all-clear
├── features.md                                 ← documentation des features
├── pyproject.toml
└── README.md
```

---

## Installation

Ce projet utilise **uv** pour la gestion de l'environnement Python.

### 1. Cloner le dépôt

```bash
git clone <url-du-repo>
cd data_battle
```

### 2. Installer les dépendances

```bash
uv sync
```

### 3. Placer les données

Copier le fichier CSV brut fourni par Météorage dans le dossier `data/` :

```
data/segment_alerts_all_airports_train.csv
```

---

## Exécution

Les scripts s'enchaînent dans cet ordre. Chaque étape produit un fichier utilisé par la suivante.

### Étape 1 — Feature engineering

Lit le CSV brut et calcule les features par alerte. Produit `data/features.parquet`.

```bash
python src/features.py data/segment_alerts_all_airports_train.csv
```

### Étape 2 — Entraînement des modèles

Entraîne un modèle Cox PH (interprétable) et un Random Survival Forest (performant). Produit les fichiers `.pkl` dans `models/`.

```bash
python src/model.py data/features.parquet
```

### Étape 3 — Évaluation complète

Évalue le modèle RSF sur le jeu de test (20% des alertes, split stratifié par aéroport). Affiche les métriques et produit `outputs/evaluation.png`.

```bash
python src/evaluate.py data/features.parquet
```

### Étape 4 (optionnelle) — Analyse des faux all-clear

Analyse en détail les alertes pour lesquelles le modèle recommande de lever trop tôt. Produit `outputs/false_allclear_analysis.png`.

```bash
python analyze_false_allclear.py data/features.parquet
```

### Étape 5 (optionnelle) — Inférence sur une alerte

Simule la prédiction minute par minute sur une alerte réelle tirée aléatoirement du dataset.

```bash
python src/predict.py data/features.parquet
```

---

## Description des fichiers source

### `src/features.py`

Calcule les features à partir du CSV brut. Pour chaque alerte (identifiée par `airport` + `airport_alert_id`), on extrait les caractéristiques de la séquence d'éclairs nuage-sol. Le résultat est un DataFrame avec une ligne par alerte, sauvegardé en parquet.

Les features sont divisées en deux groupes :
- **Actives** (17) : utilisées par les modèles
- **Inactives** (6) : calculées mais non utilisées (importance trop faible), conservées pour usage futur

→ Voir `features.md` pour la définition complète de chaque feature.

### `src/model.py`

Entraîne deux modèles de survie sur les features calculées :

- **Kaplan-Meier** : courbe de survie empirique globale, sans features, pour exploration
- **Cox PH** (`lifelines`) : modèle linéaire interprétable, bonne baseline
- **Random Survival Forest** (`scikit-survival`) : modèle non-linéaire, capture les interactions, meilleure performance

Le split train/test est **80/20, stratifié par aéroport** pour garantir la représentativité de chaque site dans les deux ensembles.

### `src/predict.py`

Charge un modèle sauvegardé et prédit la probabilité de fin d'alerte pour une alerte en cours. Prend en entrée un dictionnaire de features et le temps écoulé depuis le dernier éclair CG (`time_since_last_cg`). Retourne la courbe de survie conditionnelle, la recommandation (LEVER / MAINTENIR) et le gain estimé vs baseline.

### `src/evaluate.py`

Évalue le modèle sur l'ensemble du jeu de test. Pour chaque alerte, simule une prédiction à `t=0` (début d'alerte) et mesure le gain vs la baseline 30 min. Produit 4 graphiques :

1. Distribution des gains vs baseline
2. Temps de levée prédit vs durée réelle
3. Distribution du gain par aéroport
4. Courbe trade-off gain / taux de faux all-clear selon le seuil

### `analyze_false_allclear.py`

Analyse les alertes mal classées (levée recommandée avant la fin réelle de l'orage). Identifie les patterns communs : aéroport concerné, durée réelle des alertes, marge d'erreur, et profil des features pour comprendre pourquoi le modèle se trompe.

---

## Résultats actuels

```
Alertes évaluées       : 526
Gain moyen             : +6.6 min
Gain médian            : +17.0 min
Modèle bat baseline    : 347/526 (66%)
Faux all-clear         : 100/526 (19%)
C-index RSF            : 0.969
```

| Aéroport | N | Gain moyen | Gain médian | % gain | % faux all-clear |
|---|---|---|---|---|---|
| Ajaccio | 106 | +9.0 min | +19.5 min | 71% | 15% |
| Bastia | 107 | +4.6 min | +11.0 min | 64% | 20% |
| Biarritz | 118 | +7.5 min | +20.0 min | 66% | 17% |
| Nantes | 41 | +5.5 min | +20.0 min | 61% | 27% |
| Pise | 154 | +6.1 min | +14.0 min | 65% | 21% |

---

## Pistes d'amélioration

- **Nantes** a le taux de faux all-clear le plus élevé (27%) — piste : features spécifiques ou modèle dédié
- Les faux all-clear sont quasi exclusivement des **alertes longues** (durée réelle moyenne : 112 min) — le modèle se trompe sur les orages qui font des pauses avant de reprendre
- Tester une **optimisation du seuil par aéroport** plutôt qu'un seuil global à 0.80
- Explorer des **features de contexte météo** (saison, heure de la journée)



# Comment lancer les modèles(pour l'instant ignorez la version B et C)
# Commandes du pipeline

## Étape 1 — Build dataset
```bash
python src/build_silence_dataset.py
```

## Étape 2 — Variantes B et C
```bash
python src/make_dataset_variants.py
```

## Étape 3 — Entraînement

### Version A (avec time_since_* — fuite)
```bash
python src/2Amodels_roc_comparison.py output/silence_dataset.csv output/baseline_results_A
```

### Version B (sans time_since_*)
```bash
python run_baseline_classifier.py output/silence_dataset_B.csv output/baseline_results_B
```

### Version C (propre — recommandée)
```bash
python run_baseline_classifier.py output/silence_dataset_C.csv output/baseline_results_C
```

## Variables utilisées pour la prédiction

### Features utilisées

- **airport** : Aéroport concerné par l’alerte ; variable catégorielle de contexte.
- **minutes_since_reference_cg** : Nombre de minutes écoulées depuis le CG inner de référence.
- **elapsed_min** : Temps écoulé depuis le début de l’alerte.
- **obs_elapsed_min** : Temps écoulé depuis le début de la fenêtre d’observation.
- **time_since_last_event_min** : Temps depuis le dernier événement observé, tous types et zones confondus.
- **time_since_last_cg_min** : Temps depuis le dernier éclair sol (CG).
- **time_since_last_ic_min** : Temps depuis le dernier éclair intra-nuage (IC).
- **time_since_last_inner_min** : Temps depuis le dernier événement observé dans la zone inner (< 20 km).
- **time_since_last_cg_inner_min** : Temps depuis le dernier éclair sol observé dans la zone inner.
- **last_event_type** : Type du dernier événement observé (`CG` ou `IC`).
- **last_event_zone** : Zone du dernier événement observé (`inner` ou `outer`).
- **last_event_amplitude** : Amplitude du dernier événement observé.
- **last_event_dist** : Distance du dernier événement observé à l’aéroport.
- **n_total** : Nombre total d’événements observés.
- **n_cg** : Nombre total de CG observés.
- **n_ic** : Nombre total de IC observés.
- **n_inner** : Nombre total d’événements dans la zone inner.
- **n_outer** : Nombre total d’événements hors zone inner.
- **n_cg_inner** : Nombre total de CG dans la zone inner.
- **n_cg_outer** : Nombre total de CG hors zone inner.
- **n_ic_inner** : Nombre total de IC dans la zone inner.
- **n_ic_outer** : Nombre total de IC hors zone inner.
- **amp_abs_mean** : Moyenne de l’amplitude absolue des événements observés.
- **amp_abs_max** : Maximum de l’amplitude absolue observée.
- **dist_mean** : Distance moyenne des événements observés.
- **dist_min** : Distance minimale observée.
- **mean_interarrival_min** : Moyenne des temps entre événements successifs.
- **median_interarrival_min** : Médiane des temps entre événements successifs.
- **max_interarrival_min** : Maximum des temps inter-arrivées observés.
- **current_silence_over_mean_interarrival** : Rapport entre le silence actuel et le temps inter-arrivées moyen.
- **current_silence_over_median_interarrival** : Rapport entre le silence actuel et le temps inter-arrivées médian.
- **current_silence_over_max_interarrival** : Rapport entre le silence actuel et le plus grand temps inter-arrivées observé.
- **n_total_last_5m** : Nombre total d’événements sur les 5 dernières minutes.
- **n_cg_inner_last_5m** : Nombre de CG inner sur les 5 dernières minutes.
- **n_cg_outer_last_5m** : Nombre de CG outer sur les 5 dernières minutes.
- **n_ic_inner_last_5m** : Nombre de IC inner sur les 5 dernières minutes.
- **n_ic_outer_last_5m** : Nombre de IC outer sur les 5 dernières minutes.
- **dist_mean_last_5m** : Distance moyenne des événements sur les 5 dernières minutes.
- **dist_min_last_5m** : Distance minimale observée sur les 5 dernières minutes.
- **amp_abs_mean_last_5m** : Amplitude absolue moyenne sur les 5 dernières minutes.
- **n_total_last_10m** : Nombre total d’événements sur les 10 dernières minutes.
- **n_cg_inner_last_10m** : Nombre de CG inner sur les 10 dernières minutes.
- **n_cg_outer_last_10m** : Nombre de CG outer sur les 10 dernières minutes.
- **n_ic_inner_last_10m** : Nombre de IC inner sur les 10 dernières minutes.
- **n_ic_outer_last_10m** : Nombre de IC outer sur les 10 dernières minutes.
- **dist_mean_last_10m** : Distance moyenne des événements sur les 10 dernières minutes.
- **dist_min_last_10m** : Distance minimale observée sur les 10 dernières minutes.
- **amp_abs_mean_last_10m** : Amplitude absolue moyenne sur les 10 dernières minutes.
- **n_total_last_20m** : Nombre total d’événements sur les 20 dernières minutes.
- **n_cg_inner_last_20m** : Nombre de CG inner sur les 20 dernières minutes.
- **n_cg_outer_last_20m** : Nombre de CG outer sur les 20 dernières minutes.
- **n_ic_inner_last_20m** : Nombre de IC inner sur les 20 dernières minutes.
- **n_ic_outer_last_20m** : Nombre de IC outer sur les 20 dernières minutes.
- **dist_mean_last_20m** : Distance moyenne des événements sur les 20 dernières minutes.
- **dist_min_last_20m** : Distance minimale observée sur les 20 dernières minutes.
- **amp_abs_mean_last_20m** : Amplitude absolue moyenne sur les 20 dernières minutes.

### Colonnes non utilisées pour la prédiction

- **airport_alert_id** : Identifiant technique de l’alerte au sein de l’aéroport.
- **alert_group** : Identifiant de groupe utilisé pour les splits train/test et la validation croisée groupée.
- **obs_start** : Début de la fenêtre d’observation.
- **alert_start** : Instant de début de l’alerte.
- **decision_time** : Instant exact où les features sont calculées.
- **cg_reference_index** : Index du CG inner de référence dans la séquence de l’alerte.
- **y** : Variable cible ; vaut 1 s’il n’y a plus de CG inner après l’instant considéré, 0 sinon.


### Features à ajouter

- **n_total_since_last_cg_inner** : Nombre total d’événements observés depuis le dernier CG dans la zone inner jusqu’à l’instant de décision.
- **n_ic_since_last_cg_inner** : Nombre d’éclairs intra-nuage (IC) observés depuis le dernier CG inner.
- **n_cg_outer_since_last_cg_inner** : Nombre de CG observés dans la zone outer depuis le dernier CG inner.
- **n_ic_outer_since_last_cg_inner** : Nombre de IC observés dans la zone outer depuis le dernier CG inner.
- **dist_mean_since_last_cg_inner** : Distance moyenne des événements observés depuis le dernier CG inner.
- **dist_min_since_last_cg_inner** : Distance minimale observée depuis le dernier CG inner.
- **amp_abs_mean_since_last_cg_inner** : Amplitude absolue moyenne des événements observés depuis le dernier CG inner.

- **delta_n_total_5m_vs_prev_5m** : Différence entre le nombre total d’événements sur les 5 dernières minutes et le nombre total d’événements sur les 5 minutes précédentes.
- **delta_n_cg_inner_5m_vs_prev_5m** : Différence entre le nombre de CG inner sur les 5 dernières minutes et celui des 5 minutes précédentes.
- **delta_n_cg_outer_5m_vs_prev_5m** : Différence entre le nombre de CG outer sur les 5 dernières minutes et celui des 5 minutes précédentes.
- **delta_n_ic_outer_5m_vs_prev_5m** : Différence entre le nombre de IC outer sur les 5 dernières minutes et celui des 5 minutes précédentes.
- **delta_dist_mean_5m_vs_prev_5m** : Différence entre la distance moyenne observée sur les 5 dernières minutes et celle observée sur les 5 minutes précédentes.
- **delta_dist_min_5m_vs_prev_5m** : Différence entre la distance minimale observée sur les 5 dernières minutes et celle observée sur les 5 minutes précédentes.

- **n_cg_0_10_last_10m** : Nombre de CG observés à moins de 10 km de l’aéroport sur les 10 dernières minutes.
- **n_cg_10_20_last_10m** : Nombre de CG observés entre 10 et 20 km de l’aéroport sur les 10 dernières minutes.
- **n_cg_20_30_last_10m** : Nombre de CG observés entre 20 et 30 km de l’aéroport sur les 10 dernières minutes.

- **share_inner_last_10m** : Proportion d’événements observés dans la zone inner parmi l’ensemble des événements observés sur les 10 dernières minutes.

### Variables à retirer potentiellement pour limiter la redondance

- **minutes_since_reference_cg** : Nombre de minutes écoulées depuis le CG inner de référence ; variable redondante avec `time_since_last_cg_inner_min` dans le schéma actuel de construction des instants de décision.
- **obs_elapsed_min** : Temps écoulé depuis le début de la fenêtre d’observation ; variable redondante avec `elapsed_min` puisque la fenêtre d’observation commence toujours 10 minutes avant le début de l’alerte.
- **max_interarrival_min** : Plus grand temps inter-arrivées observé ; variable potentiellement bruitée et moins stable que la moyenne ou la médiane des temps inter-arrivées.
- **current_silence_over_max_interarrival** : Rapport entre le silence actuel et le plus grand temps inter-arrivées observé ; variable potentiellement instable car fondée sur une quantité extrême.
