# Features — Prédiction de fin d'alerte orageuse

Généré depuis `src/features.py` (RSF v7, 30 features actives).

---

## Données sources

Données Météorage observées sur 10 ans (2016–2025), dans un rayon de 30 km autour de chaque aéroport. Chaque ligne correspond à un éclair.

| Colonne | Description |
|---|---|
| `date` | Horodatage UTC de l'éclair |
| `lon` / `lat` | Position de l'éclair (degrés décimaux, WGS84 EPSG:4326) |
| `amplitude` | Polarité et intensité maximale du courant de décharge (kA) |
| `maxis` | Erreur de localisation théorique estimée (km) |
| `icloud` | `False` = éclair nuage-sol (CG), `True` = éclair intra-nuage |
| `dist` | Distance de l'éclair à l'aéroport (km) |
| `azimuth` | Direction de l'éclair par rapport à l'aéroport (degrés, 0=Nord, 90=Est, 180=Sud, 270=Ouest) |
| `airport_alert_id` | Identifiant de l'alerte au sein d'un aéroport |
| `is_last_lightning_cloud_ground` | `True` si c'est le dernier éclair CG de l'alerte (uniquement pour éclairs < 20 km) |

> **Note Pise** : données intra-nuage de 2016 potentiellement non fiables — à écarter pour les analyses sur `icloud`.

Seuls les éclairs **nuage-sol** (`icloud = False`) sont utilisés pour le calcul des features.

---

## Aéroports

| Aéroport | Latitude | Longitude |
|---|---|---|
| Ajaccio | 41.9236 | 8.8029 |
| Bastia | 42.5527 | 9.4837 |
| Biarritz | 43.4683 | -1.524 |
| Bron | 45.7294 | 4.9389 |
| Nantes | 47.1532 | -1.6107 |
| Pise | 43.695 | 10.399 |

---

## Variable cible

### RSF (analyse de survie)
- `duration` : durée totale de l'alerte en minutes (premier → dernier éclair CG)
- `event` : toujours 1 (alertes non censurées)

### XGBoost snapshots (`features_snapshot.py`)
- `y = 1` : un éclair CG survient dans les **30 prochaines minutes** → maintenir l'alerte
- `y = 0` : aucun éclair CG dans les 30 min → lever possible

---

## Features actives (30)

### Comptage

| Feature | Description |
|---|---|
| `n_cg_total` | Nombre total d'éclairs CG dans l'alerte |
| `activity_trend` | Éclairs en 2ème moitié − éclairs en 1ère moitié (positif = activité croissante) |
| `cg_density` | `n_cg_total / duration` (éclairs/min) |

### Amplitude

| Feature | Description |
|---|---|
| `amp_max` | Intensité maximale absolue sur toute l'alerte (kA) |
| `amp_trend_global` | Pente linéaire de l'amplitude sur toute l'alerte |
| `amp_decay_rate` | `amp_trend_global / amp_mean` (décroissance normalisée) |

### Distance

| Feature | Description |
|---|---|
| `dist_min` | Distance minimale d'un éclair à l'aéroport (km) |
| `dist_min_adjusted` | `dist_min − maxis` au point le plus proche (borne inférieure pessimiste) |
| `dist_trend_global` | Pente linéaire de la distance sur toute l'alerte |

### Localisation

| Feature | Description |
|---|---|
| `maxis_mean` | Erreur de localisation moyenne sur l'alerte (km) |

### Azimuth

| Feature | Description |
|---|---|
| `azimuth_spread` | Écart-type circulaire des directions des éclairs (degrés) |
| `azimuth_trend` | Vitesse de rotation angulaire moyenne (degrés/min) |

### Centroïde

| Feature | Description |
|---|---|
| `centroid_speed` | Vitesse de déplacement du centroïde des éclairs (km/min), calculée entre le centroïde des 5 premiers et des 5 derniers éclairs |
| `spatial_spread` | Dispersion spatiale des éclairs autour de leur centroïde (km), calculée comme l'écart-type de position en lat/lon converti en km |

### Temporel

| Feature | Description |
|---|---|
| `elapsed_time` | Durée totale de l'alerte (min) |
| `inter_time_last3` | Intervalle moyen entre les 3 derniers éclairs CG (min) |

### Structure temporelle

Ces features capturent la dynamique de l'orage dans le temps. Une alerte avec de longues pauses et plusieurs reprises a plus de chances de se prolonger.

| Feature | Description |
|---|---|
| `n_bursts` | Nombre de reprises d'activité après un creux (bins de 5 min) |
| `activity_variance` | Variance du comptage d'éclairs par fenêtre de 5 min |
| `pause_max` | Plus longue pause entre deux éclairs consécutifs (min) |
| `pause_ratio` | `pause_max / elapsed_time` |
| `intensity_persistence` | Proportion du temps avec une activité supérieure à la moyenne |
| `pause_since_peak` | Temps entre le pic d'activité et la fin de l'alerte (min) |
| `resume_risk` | `pause_max × n_bursts` — risque de reprise après une longue pause |
| `long_pause_count` | Nombre de pauses de plus de 5 min |
| `pause_cv` | Coefficient de variation des intervalles inter-éclairs (irrégularité temporelle) |

### Queue d'alerte — signaux de fin d'orage (v7)

Caractérisent explicitement la « mort » de l'orage en analysant le comportement des derniers éclairs.

| Feature | Description | Interprétation |
|---|---|---|
| `last_cg_amp_ratio` | `amp_dernier / amp_max` | Proche de 0 → dernier éclair faible → orage mourant |
| `last3_dist_trend` | Pente de la distance sur les 3 derniers éclairs (km/min) | Positif → éclairs qui s'éloignent |
| `inter_time_acceleration` | Tendance des intervalles inter-éclairs (min/éclair) | Positif → intervalles qui s'allongent → ralentissement |
| `final_activity_ratio` | Densité des 5 dernières min / densité globale | Proche de 0 → activité quasi nulle en fin d'alerte |
| `dying_score` | Score composite [0, 1] — combine les 4 signaux ci-dessus | Proche de 1 → orage clairement mourant |

`dying_score` est calculé comme suit :

```
s_amp   = 1 - min(last_cg_amp_ratio, 1)
s_dist  = clip(last3_dist_trend / 5, 0, 1)
s_accel = clip(inter_time_acceleration / 5, 0, 1)
s_act   = 1 - min(final_activity_ratio, 1)

dying_score = (s_amp + s_dist + s_accel + s_act) / 4
```

---

## Features inactives (importance négative ou nulle)

Calculées mais désactivées après analyse de l'importance par permutation (RSF).

| Feature | Raison |
|---|---|
| `amp_mean`, `amp_last`, `amp_trend_recent`, `amp_recent_mean` | Importance négative |
| `dist_last`, `dist_mean`, `dist_recent_min`, `dist_last_adjusted` | Importance négative |
| `dist_trend_recent`, `dist_trend_last5`, `dist_trend_last2` | Redondantes avec `dist_trend_global` |
| `n_cg_recent`, `n_cg_last5`, `n_cg_last2` | Redondantes avec `n_cg_total` et `cg_density` |
| `maxis_last` | Importance négative |
| `azimuth_last`, `azimuth_last_vs_mean` | Importance quasi nulle |
| `centroid_approach`, `centroid_dist_last`, `centroid_dist_change` | Importance négative |
| `ratio_ic_cg`, `n_ic_total`, `n_ic_recent`, `ratio_ic_recent` | Non informatifs pour la fin d'alerte |
| `month`, `season` | Importance négative |
| `hour_of_day` | Importance négative (v6) |
| `time_since_prev_alert`, `n_prev_alerts_6h` | Importance quasi nulle (v6) |
| `dist_last_vs_mean` | Importance négative |
| `time_since_last_cg` | Toujours 0 dans le RSF — utile uniquement en inférence XGBoost |

---

## Feature spéciale XGBoost : `time_since_last_cg`

Dans le RSF, `time_since_last_cg` est toujours 0 : les features sont calculées à l'instant du dernier éclair.

Dans l'approche XGBoost snapshots, des snapshots sont générés toutes les 2 min après le dernier éclair (jusqu'à 35 min de silence). `time_since_last_cg` devient alors la feature la plus importante (importance = 0.21) : plus elle est grande, plus l'orage est probablement terminé. Le modèle combine ce signal avec les autres features pour distinguer un vrai silence d'une simple pause.