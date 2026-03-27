# src/build_silence_dataset.py

Génère les données d'entraînement pour les modèles de prédiction.

# README - Description des features du `silence_dataset`

## 1. Objectif du dataset

Ce dataset a pour but de modéliser la **levée d’alerte orage** à partir d’**instants de décision** construits pendant les périodes de silence qui suivent un éclair **CG inner**.

À chaque instant de décision `t`, on cherche à prédire la variable cible :

- `y = 1` s’il n’y a **plus aucun CG inner après t** dans l’alerte ;
- `y = 0` sinon.

Autrement dit, le problème consiste à déterminer si le silence observé correspond à une **vraie fin d’activité dangereuse** ou seulement à une **accalmie temporaire**.

---

## 2. Construction générale des instants de décision

Pour chaque alerte :

1. on isole les éclairs **CG inner** ;
2. après chaque CG inner, on construit une grille d’instants de décision toutes les `3` minutes ;
3. cette grille s’étend :
   - jusqu’au prochain CG inner s’il arrive avant,
   - ou au maximum jusqu’à `30` minutes après le CG inner de référence ;
4. à chaque instant `t`, on calcule des variables décrivant :
   - la récence,
   - l’activité récente,
   - la proximité,
   - la structure de l’orage,
   - les signaux de type LRE.

---

## 3. Variables d’identification et de contexte

Ces variables servent à identifier l’alerte, l’aéroport et l’instant de décision.  
Elles ne sont pas toutes destinées à être utilisées comme variables prédictives.

### Variables :
- `airport` : aéroport concerné ;
- `airport_alert_id` : identifiant de l’alerte dans l’aéroport ;
- `alert_group` : identifiant unique de l’alerte, de la forme `airport__alert_id` ;
- `obs_start` : début de la fenêtre d’observation ;
- `alert_start` : début de l’alerte, défini comme la date du premier `CG inner` ;
- `decision_time` : instant auquel on prend la décision ;
- `cg_reference_index` : indice du `CG inner` ayant servi de référence pour construire l’instant de silence ;
- `minutes_since_reference_cg` : temps écoulé entre `decision_time` et le `CG inner` de référence.

### Rôle :
Ces variables servent principalement :
- au suivi des alertes,
- au découpage train/test,
- à l’interprétation métier,
- à la reconstruction temporelle des décisions.

---

## 4. Variable cible

### `y`
Indique si l’alerte peut être considérée comme terminée à l’instant `t`.

- `y = 1` : plus aucun `CG inner` ne survient après `t` ;
- `y = 0` : au moins un `CG inner` survient encore après `t`.

### Rôle :
C’est la variable à prédire.

---

## 5. Catégorie 1 - Variables de récence

Ces variables mesurent **depuis combien de temps aucun événement n’a été observé**, globalement ou par sous-type.  
Elles sont au cœur du problème, car la levée d’alerte dépend d’abord de la **durée du silence**.

### Variables :
- `elapsed_min` : temps écoulé depuis le début de l’alerte ;
- `obs_elapsed_min` : temps écoulé depuis le début de la fenêtre d’observation ;
- `time_since_last_event_min` : temps depuis le dernier éclair, tous types confondus ;
- `time_since_last_cg_min` : temps depuis le dernier éclair `CG` ;
- `time_since_last_ic_min` : temps depuis le dernier éclair `IC` ;
- `time_since_last_inner_min` : temps depuis le dernier éclair en zone `inner` ;
- `time_since_last_cg_inner_min` : temps depuis le dernier `CG inner`.

### Intuition métier :
Plus le dernier événement dangereux remonte dans le temps, plus la probabilité de fin d’alerte augmente.

---

## 6. Catégorie 2 - Variables de comptage cumulées

Ces variables décrivent le **volume d’activité observé depuis le début de la fenêtre d’observation** jusqu’à l’instant `t`.

### Variables :
- `n_total` : nombre total d’éclairs observés ;
- `n_cg` : nombre total de `CG` ;
- `n_ic` : nombre total de `IC` ;
- `n_inner` : nombre total d’éclairs en `inner` ;
- `n_outer` : nombre total d’éclairs en `outer` ;
- `n_cg_inner` : nombre total de `CG inner` ;
- `n_cg_outer` : nombre total de `CG outer` ;
- `n_ic_inner` : nombre total de `IC inner` ;
- `n_ic_outer` : nombre total de `IC outer` ;
- `n_lre` : nombre total d’éclairs situés à moins de `3 km` de l’aéroport.

### Intuition métier :
Ces variables décrivent l’historique global de l’activité orageuse et permettent de distinguer :
- une alerte issue d’un épisode intense,
- une alerte issue d’un épisode faible ou peu durable.

---

## 7. Catégorie 3 - Variables de fenêtres glissantes

Ces variables décrivent l’activité **très récente**, dans des fenêtres de `5`, `10` et `20` minutes avant `decision_time`.

### Variables de comptage :
Pour chaque fenêtre `w ∈ {5, 10, 20}` :
- `n_total_last_{w}m`
- `n_cg_inner_last_{w}m`
- `n_cg_outer_last_{w}m`
- `n_ic_inner_last_{w}m`
- `n_ic_outer_last_{w}m`
- `n_lre_last_{w}m`

### Intuition métier :
Ces variables capturent la **persistance récente du danger**.  
Elles sont souvent plus informatives que les comptes cumulés, car la levée dépend surtout de ce qui s’est passé **dans les dernières minutes**.

---

## 8. Catégorie 4 - Variables de distance et de proximité

Ces variables décrivent à quelle distance de l’aéroport se situe l’activité électrique.

### Variables globales :
- `dist_mean` : distance moyenne des éclairs observés ;
- `dist_min` : distance minimale observée ;
- `last_event_dist` : distance du dernier événement.

### Variables sur fenêtres glissantes :
Pour chaque fenêtre `w ∈ {5, 10, 20}` :
- `dist_mean_last_{w}m`
- `dist_min_last_{w}m`

### Variables de dynamique spatiale :
- `delta_dist_min_20_5` : différence entre la distance minimale sur 20 minutes et la distance minimale sur 5 minutes ;
- `delta_dist_mean_20_5` : différence entre la distance moyenne sur 20 minutes et celle sur 5 minutes.

### Intuition métier :
Ces variables permettent de savoir si l’activité :
- reste proche de l’aéroport,
- s’en éloigne,
- ou au contraire se rapproche à nouveau.

Les variables `delta_*` visent à capter une **approche progressive** ou un **éloignement progressif** du phénomène.

---

## 9. Catégorie 5 - Variables d’amplitude

Ces variables décrivent l’intensité électrique des éclairs.

### Variables globales :
- `last_event_amplitude` : amplitude du dernier événement ;
- `amp_abs_mean` : amplitude absolue moyenne ;
- `amp_abs_max` : amplitude absolue maximale.

### Variables sur fenêtres glissantes :
Pour chaque fenêtre `w ∈ {5, 10, 20}` :
- `amp_abs_mean_last_{w}m`

### Intuition métier :
L’amplitude peut apporter de l’information sur la vigueur du phénomène, mais elle est généralement considérée comme un signal plus secondaire que :
- la récence,
- la proximité,
- et la persistance de l’activité en `inner`.

---

## 10. Catégorie 6 - Variables d’inter-arrivées et de normalisation du silence

Ces variables décrivent le rythme temporel des éclairs observés.

### Variables :
- `mean_interarrival_min` : inter-arrivée moyenne entre éclairs ;
- `median_interarrival_min` : inter-arrivée médiane ;
- `max_interarrival_min` : inter-arrivée maximale.

### Ratios de silence :
- `current_silence_over_mean_interarrival`
- `current_silence_over_median_interarrival`
- `current_silence_over_max_interarrival`

### Intuition métier :
Le silence courant n’a pas la même signification selon le rythme habituel de l’orage.

Exemple :
- un silence de 6 minutes peut être long dans un orage très dense ;
- mais banal dans un orage naturellement espacé.

Ces ratios permettent donc de **normaliser le silence observé** par rapport au comportement passé de l’orage.

---

## 11. Catégorie 7 - Variables LRE (Low Radius Events / proximité extrême)

Ici, un **LRE** désigne un éclair situé à moins de `3 km` de l’aéroport.

### Variables :
- `n_lre` : nombre total de LRE observés ;
- `n_lre_last_5m`
- `n_lre_last_10m`
- `n_lre_last_20m`
- `has_lre_before` : indicateur binaire valant `1` s’il y a déjà eu au moins un LRE avant `t` ;
- `time_since_last_lre_min` : temps depuis le dernier LRE ;
- `n_lt_3km_last_10m` : nombre d’éclairs à moins de 3 km dans les 10 dernières minutes ;
- `n_lt_3km_last_20m` : nombre d’éclairs à moins de 3 km dans les 20 dernières minutes.

### Intuition métier :
Les LRE représentent une **proximité critique** vis-à-vis de l’aéroport.  
Même si ces événements sont plus rares, ils sont importants opérationnellement car ils signalent un danger très proche.

---

## 12. Catégorie 8 - Variables sur le dernier événement observé

Ces variables décrivent la nature du dernier éclair connu au moment de la décision.

### Variables :
- `last_event_type` : type du dernier événement (`CG` ou `IC`) ;
- `last_event_zone` : zone du dernier événement (`inner` ou `outer`) ;
- `last_event_amplitude` : amplitude du dernier événement ;
- `last_event_dist` : distance du dernier événement.

### Intuition métier :
Le dernier événement observé donne une information synthétique et immédiate sur l’état courant du phénomène :
- nature de l’éclair,
- proximité,
- intensité.

---

## 13. Lecture globale du jeu de features

Le jeu de features est construit autour de quatre idées principales :

### a. La récence
La question centrale est :  
**depuis combien de temps le phénomène dangereux est-il silencieux ?**

### b. La persistance récente
Le modèle doit savoir si l’activité récente, notamment en `inner`, est encore dense.

### c. La proximité
Un orage encore actif près de l’aéroport n’a pas la même signification qu’un orage qui s’éloigne.

### d. Les signaux critiques LRE
La présence récente d’événements très proches peut indiquer un danger résiduel plus fort.

---

## 14. Résumé des catégories de features

### Variables d’identification
- `airport`
- `airport_alert_id`
- `alert_group`
- `obs_start`
- `alert_start`
- `decision_time`
- `cg_reference_index`
- `minutes_since_reference_cg`

### Variable cible
- `y`

### Variables de récence
- `elapsed_min`
- `obs_elapsed_min`
- `time_since_last_event_min`
- `time_since_last_cg_min`
- `time_since_last_ic_min`
- `time_since_last_inner_min`
- `time_since_last_cg_inner_min`

### Variables de comptage cumulées
- `n_total`
- `n_cg`
- `n_ic`
- `n_inner`
- `n_outer`
- `n_cg_inner`
- `n_cg_outer`
- `n_ic_inner`
- `n_ic_outer`
- `n_lre`

### Variables sur fenêtres glissantes
- `n_total_last_5m`, `n_total_last_10m`, `n_total_last_20m`
- `n_cg_inner_last_5m`, `n_cg_inner_last_10m`, `n_cg_inner_last_20m`
- `n_cg_outer_last_5m`, `n_cg_outer_last_10m`, `n_cg_outer_last_20m`
- `n_ic_inner_last_5m`, `n_ic_inner_last_10m`, `n_ic_inner_last_20m`
- `n_ic_outer_last_5m`, `n_ic_outer_last_10m`, `n_ic_outer_last_20m`
- `n_lre_last_5m`, `n_lre_last_10m`, `n_lre_last_20m`

### Variables de distance
- `dist_mean`
- `dist_min`
- `last_event_dist`
- `dist_mean_last_5m`, `dist_mean_last_10m`, `dist_mean_last_20m`
- `dist_min_last_5m`, `dist_min_last_10m`, `dist_min_last_20m`
- `delta_dist_min_20_5`
- `delta_dist_mean_20_5`

### Variables d’amplitude
- `last_event_amplitude`
- `amp_abs_mean`
- `amp_abs_max`
- `amp_abs_mean_last_5m`
- `amp_abs_mean_last_10m`
- `amp_abs_mean_last_20m`

### Variables d’inter-arrivées
- `mean_interarrival_min`
- `median_interarrival_min`
- `max_interarrival_min`
- `current_silence_over_mean_interarrival`
- `current_silence_over_median_interarrival`
- `current_silence_over_max_interarrival`

### Variables LRE
- `n_lre`
- `n_lre_last_5m`
- `n_lre_last_10m`
- `n_lre_last_20m`
- `has_lre_before`
- `time_since_last_lre_min`
- `n_lt_3km_last_10m`
- `n_lt_3km_last_20m`

### Variables sur le dernier événement
- `last_event_type`
- `last_event_zone`
- `last_event_amplitude`
- `last_event_dist`

---

## 15. Philosophie générale du feature engineering

La construction des variables ne repose pas sur une accumulation arbitraire de descripteurs.  
Elle suit une logique métier liée à la levée d’alerte :

- mesurer si le silence est long ou non ;
- mesurer si une activité récente persiste près de l’aéroport ;
- mesurer si l’orage s’éloigne ou reste proche ;
- détecter des événements de proximité extrême via les LRE.

En ce sens, les features les plus importantes ne sont pas nécessairement les plus nombreuses, mais celles qui traduisent le mieux :
- la récence,
- la persistance de l’activité en `inner`,
- et la proximité dangereuse résiduelle.

---