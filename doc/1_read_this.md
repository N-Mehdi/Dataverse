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

> **Note :** `alert_airport_id` et `is_last_lightning_cloud_ground` ne sont renseignés que pour les éclairs de type Nuage-Sol à moins de 20 km de l'aéroport.

---

## Structure du projet

```
Dataverse/
│
├── data/
│   ├── segment_alerts_all_airports_train.csv   ← fichier brut à placer ici
│   └── features.parquet                        ← généré par features.py
│
├── doc/
│   ├── features.md                             ← définition de toutes les features
│   └── 1_read_this.md                          ← point d'entrée pour comprendre le projet
|
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
|   ├── analyse_false_all_clear.py              ← analyse des faux all-clear
│   ├── features.py                             ← feature engineering
│   ├── model.py                                ← entraînement des modèles
│   ├── predict.py                              ← inférence sur une alerte
│   └── evaluate.py                             ← évaluation complète sur le jeu de test
│
├── pyproject.toml
└── README.md
```

---

## Installation

Ce projet utilise **uv** pour la gestion de l'environnement Python.

### 1. Installer les dépendances

```bash
uv sync
```

### 2. Placer les données

Copier le fichier CSV brut fourni par Météorage dans le dossier `data/` :

```
data/segment_alerts_all_airports_train.csv
```

---

## Analyse descriptive
Avant de commencer la modélisation, regardez d'abord l'nalyse descriptive dans data/exploration.ipynb.  
Pour executer une celule : ctrl + enter dans la cellule (cliquer dans la cellule avant le raccourci)  
Pour executer une cellule et passer à la suivante : ctrl + shift + enter

## Description des fichiers source (src)

### `src/features.py`

Calcule les features à partir du CSV brut. Pour chaque alerte (identifiée par `airport` + `airport_alert_id`), on extrait les caractéristiques de la séquence d'éclairs nuage-sol. Le résultat est un DataFrame avec une ligne par alerte, sauvegardé en parquet.

Les features sont divisées en deux groupes :
- **Actives** (17) : utilisées par les modèles
- **Inactives** (6) : calculées mais non utilisées (importance trop faible), conservées pour usage futur

→ Voir `doc/features.md` pour la définition complète de chaque feature.

### `src/model.py`

Entraîne deux modèles de survie sur les features calculées :

- **Kaplan-Meier** : courbe de survie empirique globale, sans features, pour exploration
- **Cox PH** (`lifelines`) : modèle linéaire interprétable, bonne baseline
- **Random Survival Forest** (`scikit-survival`) : modèle non-linéaire, capture les interactions, meilleure performance

Le split train/test est **80/20, stratifié par aéroport** pour garantir la représentativité de chaque site dans les deux ensembles.  
ref. tout en bas pour expliquer les modèles entraînés.

### `src/predict.py`

Charge un modèle sauvegardé et prédit la probabilité de fin d'alerte pour une alerte en cours. Prend en entrée un dictionnaire de features et le temps écoulé depuis le dernier éclair CG (= Cloud-Ground, Nuage-sol) (`time_since_last_cg`). Retourne la courbe de survie conditionnelle, la recommandation (LEVER / MAINTENIR) et le gain estimé vs baseline.

### `src/evaluate.py`

Évalue le modèle sur l'ensemble du jeu de test. Pour chaque alerte, simule une prédiction à `t=0` (début d'alerte) et mesure le gain vs la baseline 30 min. Produit 4 graphiques :

1. Distribution des gains vs baseline
2. Temps de levée prédit vs durée réelle
3. Distribution du gain par aéroport
4. Courbe trade-off gain / taux de faux all-clear selon le seuil

→ Voir `outputs/evaluation.png` pour les 4 graphiques précédents.

### `analyze_false_allclear.py`

Analyse les alertes mal classées (levée de l'alerte recommandée avant la fin réelle de l'orage). Identifie les patterns communs : aéroport concerné, durée réelle des alertes, marge d'erreur, et profil des features pour comprendre pourquoi le modèle se trompe.

---


## Exécution

Les scripts s'enchaînent dans cet ordre. Chaque étape produit un fichier utilisé par la suivante.  
Veuillez executer les codes suivants un par un si vous souhaitez découvrir les analyses faites pour l'instant.

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

Remarques sur l'entraînement des modèles :  
2627 - 1661 = 966 alertes sont perdues à cause de deux filtres :

duration > 0 — les alertes avec un seul éclair CG ont une durée de 0 (début = fin). D'après les stats de features.py, le 25e percentile de durée est 0, donc une grosse partie des alertes sont des éclairs isolés sans durée mesurable.  
dropna — quelques alertes avec des valeurs manquantes dans les features.  

Le filtre duration > 0 est le plus impactant. Ces alertes à durée nulle correspondent à des orages très courts avec un seul éclair CG — elles ne sont pas vraiment utilisables pour un modèle de survie (pas de séquence temporelle à analyser).  

À voir si on peut créer une feature pour intégrer ce type d'alertes. 

### Étape 3 — Évaluation complète

Évalue le modèle RSF sur le jeu de test (20% des alertes, split stratifié par aéroport). Affiche les métriques et produit `outputs/evaluation.png`.

```bash
python src/evaluate.py data/features.parquet
```

### Étape 4 (optionnelle) — Analyse des faux all-clear

Analyse en détail les alertes pour lesquelles le modèle recommande de lever trop tôt. Produit `outputs/false_allclear_analysis.png`.

```bash
python src/analyse_false_all_clear.py data/features.parquet
```

### Étape 5 (optionnelle) — Inférence sur une alerte

Simule la prédiction minute par minute sur une alerte réelle tirée aléatoirement du dataset.

```bash
python src/predict.py data/features.parquet
```

---
## Modèle de survie

Un modèle de survie est un type de modèle statistique conçu pour prédire le temps avant qu'un événement se produise. Dans notre cas, l'événement c'est la fin de l'orage. Ce qui rend ces modèles spéciaux par rapport à une régression classique, c'est qu'ils gèrent naturellement l'incertitude temporelle : on ne sait pas encore quand l'orage va finir, on veut juste estimer la probabilité qu'il soit encore actif dans X minutes.

## Kaplan-Meier

Kaplan-Meier est le point de départ de toute analyse de survie. Il ne prend aucune feature en entrée — il regarde juste la distribution empirique des durées d'alerte dans les données historiques et construit une courbe qui dit "après X minutes, quelle proportion des orages est encore active ?". C'est utile pour comprendre la forme générale du problème : est-ce que les orages finissent surtout en moins de 20 min ? En moins de 60 min ? C'est une exploration, pas un modèle prédictif.

## Le modèle Cox PH

Cox PH (Proportional Hazards) est le modèle de survie classique en statistique. Il suppose que chaque feature multiplie le "risque de fin d'orage" par un facteur constant dans le temps — c'est l'hypothèse des hasards proportionnels. Par exemple, si "dist_min" a un coefficient négatif, un éclair proche de l'aéroport réduit la probabilité de fin imminente. L'avantage c'est qu'il est très interprétable : on peut lire directement l'effet de chaque feature. L'inconvénient c'est qu'il est linéaire et ne capture pas les interactions entre features.

## Le modèle RSF (Random Survival Forest)

Random Survival Forest est l'extension non-linéaire. C'est une forêt aléatoire adaptée à la survie : chaque arbre apprend à séparer les alertes courtes des alertes longues en combinant plusieurs features à la fois. Il n'a aucune hypothèse sur la forme des relations entre features et durée, ce qui lui permet de détecter des patterns complexes — par exemple "si l'orage a fait plusieurs pauses ET que l'amplitude est élevée, alors il va probablement durer encore longtemps". C'est notre modèle principal.

### Le C-index

Le C-index (Concordance index) mesure la capacité du modèle RSF à bien ordonner les durées d'alerte.  
Concrètement : on prend toutes les paires d'alertes possibles dans le jeu de test, et on vérifie que le modèle prédit bien "l'alerte A va durer plus longtemps que l'alerte B" quand c'est effectivement le cas dans la réalité.

0.5 = le modèle est aussi bon qu'un tirage aléatoire  
1.0 = le modèle ordonne parfaitement toutes les paires  
0.969 = sur 100 paires d'alertes, le modèle se trompe sur l'ordre seulement ~3 fois  

Ce que le C-index ne mesure pas : la précision absolue du temps de levée prédit. Un modèle peut avoir un excellent C-index mais quand même lever trop tôt sur certaines alertes — c'est exactement ce qu'on observe avec nos 19% de faux all-clear. C'est pourquoi on utilise aussi le gain moyen et le taux de faux all-clear comme métriques complémentaires.


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
