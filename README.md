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




# Commandes du pipeline

## Étape 1 — Build dataset
```bash
python build_silence_dataset.py
```

## Étape 2 — Variantes B et C
```bash
python make_dataset_variants.py
```

## Étape 3 — Entraînement

### Version A (avec time_since_* — fuite)
```bash
python 2Amodels_roc_comparison.py output/silence_dataset.csv output/baseline_results_A
```

### Version B (sans time_since_*)
```bash
python run_baseline_classifier.py output/silence_dataset_B.csv output/baseline_results_B
```

### Version C (propre — recommandée)
```bash
python run_baseline_classifier.py output/silence_dataset_C.csv output/baseline_results_C
```