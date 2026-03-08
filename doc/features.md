# Features — Modèle de prédiction de fin d'alerte orageuse

## Features actives

| Feature | Définition |
|---|---|
| `n_cg_total` | Nombre total d'éclairs nuage-sol (CG) survenus depuis le début de l'alerte |
| `n_cg_recent` | Nombre d'éclairs CG survenus dans les 10 dernières minutes de l'alerte |
| `activity_trend` | Différence entre le nombre d'éclairs CG de la 2ème moitié et de la 1ère moitié de l'alerte — une valeur négative indique une activité décroissante |
| `amp_max` | Intensité maximale (en valeur absolue) de tous les éclairs CG de l'alerte, exprimée en kA |
| `amp_mean` | Intensité moyenne (en valeur absolue) de tous les éclairs CG de l'alerte, exprimée en kA |
| `amp_trend_global` | Pente de la régression linéaire de l'amplitude en fonction du temps sur toute la durée de l'alerte (kA/min) — valeur négative = éclairs de moins en moins intenses |
| `amp_trend_recent` | Pente de la régression linéaire de l'amplitude en fonction du temps sur les 10 dernières minutes (kA/min) |
| `dist_min` | Distance minimale atteinte par un éclair CG par rapport à l'aéroport, en km |
| `dist_mean` | Distance moyenne de tous les éclairs CG par rapport à l'aéroport, en km |
| `dist_recent_min` | Distance minimale atteinte par un éclair CG dans les 10 dernières minutes, en km |
| `dist_trend_global` | Pente de la régression linéaire de la distance en fonction du temps sur toute la durée de l'alerte (km/min) — valeur positive = orage qui s'éloigne |
| `dist_trend_recent` | Pente de la régression linéaire de la distance en fonction du temps sur les 10 dernières minutes (km/min) |
| `elapsed_time` | Durée écoulée depuis le début de l'alerte jusqu'au dernier éclair CG observé, en minutes |
| `n_bursts` | Nombre de reprises d'activité CG après un creux — une reprise est comptée chaque fois qu'une fenêtre de 5 min sans éclair est suivie d'une fenêtre avec au moins un éclair |
| `activity_variance` | Variance du nombre d'éclairs CG par fenêtre de 5 minutes sur toute la durée de l'alerte — valeur élevée = orage irrégulier en dents de scie |
| `pause_max` | Plus longue pause observée entre deux éclairs CG consécutifs au cours de l'alerte, en minutes |
| `intensity_persistence` | Proportion de fenêtres de 5 minutes où le nombre d'éclairs CG dépasse la moyenne de l'alerte — valeur proche de 1 = activité soutenue et persistante |

## Features inactives

Ces features sont calculées et conservées dans le fichier parquet mais ne sont pas utilisées par le modèle (importance trop faible lors des évaluations initiales).

| Feature | Définition |
|---|---|
| `n_ic_total` | Nombre total d'éclairs intra-nuage (IC) survenus depuis le début de l'alerte |
| `ratio_ic_cg` | Rapport entre le nombre d'éclairs intra-nuage et le nombre d'éclairs CG sur toute l'alerte |
| `n_ic_recent` | Nombre d'éclairs intra-nuage survenus dans les 10 dernières minutes de l'alerte |
| `ratio_ic_recent` | Rapport entre les éclairs intra-nuage récents et les éclairs CG récents (10 dernières minutes) |
| `amp_recent_mean` | Amplitude moyenne (en valeur absolue) des éclairs CG des 10 dernières minutes, en kA |
| `time_since_last_cg` | Temps écoulé depuis le dernier éclair CG observé, en minutes — vaut toujours 0 dans le jeu d'entraînement, utilisé uniquement en inférence temps réel |