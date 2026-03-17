# features_snapshot.py

Génère les données d'entraînement pour le modèle XGBoost de prédiction de fin d'alerte orageuse.

---

## Concept : le snapshot

Plutôt que de modéliser une alerte entière d'un coup, on photographie l'état de l'orage à des instants réguliers (toutes les 2 minutes). Chaque snapshot devient une ligne du dataset d'entraînement.

```
t_start ──── snapshots pendant l'alerte ──── t_last_cg ──── snapshots silence ──── t_end
               (interval = 2 min)                             (jusqu'à +35 min)
```

**Cible y :**
- `y = 1` si un éclair CG survient dans les 30 prochaines minutes
- `y = 0` sinon → on peut lever l'alerte

---

## Paramètres

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `SNAPSHOT_INTERVAL` | 2 min | Intervalle entre deux snapshots |
| `HORIZON` | 30 min | Fenêtre de prédiction pour y |
| `SILENCE_WINDOW` | 35 min | Durée de la période de silence générée après le dernier éclair |
| `MIN_CG` | 2 | Nombre minimum d'éclairs CG pour générer un snapshot |

---

## Features calculées (32 au total)

### Temporelles

| Feature | Description |
|---|---|
| `time_since_last_cg` | ⭐ Minutes depuis le dernier éclair — feature clé : 0 pendant l'orage, croît pendant le silence |
| `elapsed_time` | Temps écoulé depuis le début de l'alerte jusqu'au snapshot |
| `inter_time_last3` | Intervalle moyen entre les 3 derniers éclairs |
| `inter_time_acceleration` | Accélération/décélération des intervalles entre éclairs |
| `pause_max` | Plus longue pause sans éclair observée jusqu'au snapshot |
| `pause_ratio` | `pause_max / durée_historique` |
| `pause_since_peak` | Temps écoulé depuis le pic d'activité |
| `pause_cv` | Coefficient de variation des pauses (régularité) |

### Amplitude des éclairs

| Feature | Description |
|---|---|
| `amp_max` | Amplitude maximale observée |
| `amp_mean` | Amplitude moyenne |
| `amp_decay_rate` | Taux de décroissance de l'amplitude (normalisé par la moyenne) |
| `amp_trend_global` | Pente globale de l'amplitude dans le temps |
| `last_cg_amp_ratio` | Amplitude du dernier éclair / amplitude max (proche de 0 = orage s'affaiblit) |

### Spatiales

| Feature | Description |
|---|---|
| `dist_min` | Distance minimale d'un éclair à l'aéroport (km) |
| `dist_min_adjusted` | `dist_min` corrigée par le rayon du cluster (`maxis`) |
| `dist_mean` | Distance moyenne |
| `dist_last` | Distance du dernier éclair |
| `dist_trend_global` | Tendance de la distance : négatif = l'orage s'approche |
| `azimuth_spread` | Dispersion angulaire des éclairs (écart-type circulaire en degrés) |
| `centroid_speed` | Vitesse de déplacement du centroïde de l'orage (km/min) |
| `spatial_spread` | Étendue spatiale de l'orage (km) |
| `maxis_mean` | Rayon moyen des clusters d'éclairs |

### Activité globale

| Feature | Description |
|---|---|
| `n_cg_total` | Nombre total d'éclairs CG dans l'historique jusqu'au snapshot |
| `cg_density` | Éclairs par minute (intensité moyenne) |
| `activity_trend` | Plus d'éclairs en 2e moitié qu'en 1re ? (>0 = activité croissante) |
| `activity_variance` | Variabilité de l'activité entre fenêtres de 5 min |
| `intensity_persistence` | Fraction du temps où l'activité est au-dessus de la moyenne |
| `n_bursts` | Nombre de reprises d'activité après une pause |
| `resume_risk` | `pause_max × n_bursts` (risque de nouvelle reprise) |
| `long_pause_count` | Nombre de pauses > 5 min |
| `final_activity_ratio` | Activité des 5 dernières minutes / activité moyenne |

### Score synthétique

| Feature | Description |
|---|---|
| `dying_score` | Score 0→1 combinant 4 signaux de fin d'orage : amplitude décroissante, éloignement, décélération, faible activité récente |

---

## Deux types de snapshots par alerte

### 1. Pendant l'alerte

Snapshots générés toutes les `SNAPSHOT_INTERVAL` minutes, de `t_start + 2×interval` jusqu'au dernier éclair. L'historique `hist` contient uniquement les éclairs survenus **avant** le snapshot — pas de fuite de données.

```python
t_snap = t_start + pd.Timedelta(minutes=interval * 2)
while t_snap <= t_last_cg:
    hist = grp[grp["date"] <= t_snap]  # historique tronqué au snapshot
    ...
```

### 2. Après le dernier éclair (silence)

Snapshots générés toutes les 2 minutes pendant `SILENCE_WINDOW` minutes après le dernier éclair. `time_since_last_cg` croît progressivement — c'est le signal le plus fort de fin d'orage.

```python
t_snap = t_last_cg + pd.Timedelta(minutes=interval)
t_end  = t_last_cg + pd.Timedelta(minutes=silence_window)
```

---

## Calcul de la cible y

La cible est calculée en regardant les éclairs **globaux de l'aéroport** (pas uniquement ceux de l'alerte courante) dans la fenêtre `[t_snapshot, t_snapshot + 30 min]`.

```python
future = all_cg_ap[
    (all_cg_ap["date"] > t_snap) & (all_cg_ap["date"] <= t_horizon)
]
y = 1 if len(future) > 0 else 0
```

---

## Usage

```bash
python features_snapshot.py data/segment_alerts_all_airports_train.csv
```

**Sortie :** `data/snapshots.parquet`

**Format de sortie :**

| Colonne | Type | Description |
|---|---|---|
| 32 features | float | Voir tableau ci-dessus |
| `airport` | str | Nom de l'aéroport |
| `airport_alert_id` | float | Identifiant de l'alerte |
| `t_snapshot` | timestamp | Instant du snapshot |
| `y` | int | Cible : 1 = éclair dans les 30 min |

---

## Notes importantes

- `elapsed_time` dans ce module vaut le temps écoulé **jusqu'au snapshot**, pas la durée totale de l'alerte — il n'y a pas de fuite de données contrairement au module RSF (`features.py`).
- Le split train/test doit se faire **par alerte** (`GroupShuffleSplit`) pour éviter que des snapshots d'une même alerte se retrouvent dans les deux ensembles.
- Les éclairs intra-nuage (`icloud == True`) sont exclus du calcul des features.