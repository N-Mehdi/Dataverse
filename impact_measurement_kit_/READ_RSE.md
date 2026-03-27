# Kit de mesure d'impacts — Projet Météorage / Data Battle 2026

Ce kit sert à produire des **mesures directes** puis des **estimations ACV** pour votre solution.

## Ce que le kit mesure directement
- durée de traitement (s)
- mémoire max (GB)
- charge CPU moyenne et max
- lectures / écritures du processus
- nombre de GPU détectés
- énergie / émissions via CodeCarbon

## Ce que le kit estime ensuite à partir de facteurs documentés
- coût électrique
- eau liée à l'électricité
- déplétion abiotique liée à l'électricité
- énergie primaire
- part imputable des impacts matériels si vous fournissez un facteur ACV matériel

## Structure
- `measure_utils.py` : instrumentation de base
- `impact_campaign.py` : mesure build / train / predict
- `benchmark_inference.py` : mesure inférence unitaire et batch
- `aggregate_results.py` : consolidation des runs
- `estimate_lca_impacts.py` : conversion avec facteurs ACV documentés
- `capacity_cost_template.py` : premier scénario de capacité / coût d'adoption
- `hardware_inventory_template.yaml` : fiche à compléter pour la machine
- `lca_factors_example.yaml` : facteurs à remplacer par vos valeurs sourcées

---

## 1) Installation minimale

Depuis la racine du projet (`Dataverse/`) avec le venv activé :

```bash
source .venv/bin/activate
uv sync
```

Les dépendances `codecarbon` et `psutil` sont déjà dans `pyproject.toml`.

---

## 2) Mesurer le pipeline réel

Depuis la racine du projet (`Dataverse/`) :

```bash
python impact_measurement_kit_/impact_campaign.py \
  --project-root . \
  --build-cmd "python -m src.build_silence_dataset data/segment_alerts_all_airports_train.csv" \
  --train-cmd "python src/XGboost/XGboost_On_All_Data.py" \
  --predict-cmd "python -m src.predict data/segment_alerts_all_airports_train.parquet" \
  --output-dir impact_measurement_kit_/impact_runs
```

Le dossier `impact_runs/` contiendra un JSON par run et un CSV CodeCarbon par run.

---

## 3) Mesurer l'inférence

### Cas A — modèle sauvegardé disponible

```bash
python impact_measurement_kit_/benchmark_inference.py \
  --model-path output/model_comparison_with_xgboost/model_xgboost.pkl \
  --parquet-path output/silence_dataset.parquet \
  --batch-size 1000 \
  --output-dir impact_measurement_kit_/impact_runs
```

### Cas B — sans modèle chargé

```bash
python impact_measurement_kit_/benchmark_inference.py \
  --model-path output/model_comparison_with_xgboost/model_xgboost.pkl \
  --parquet-path output/silence_dataset.parquet \
  --batch-size 1000 \
  --output-dir impact_measurement_kit_/impact_runs
```

---

## 4) Consolider les mesures

```bash
python impact_measurement_kit_/aggregate_results.py \
  --input-dir impact_measurement_kit_/impact_runs \
  --output-csv impact_measurement_kit_/impact_summary.csv
```

Cela produit `impact_summary.csv`.

---

## 5) Ajouter les facteurs ACV documentés

Complétez :
- `impact_measurement_kit_/hardware_inventory_template.yaml`
- `impact_measurement_kit_/lca_factors_example.yaml`

Puis lancez :

```bash
python impact_measurement_kit_/estimate_lca_impacts.py \
  --summary-csv impact_measurement_kit_/impact_summary.csv \
  --config impact_measurement_kit_/lca_factors_example.yaml \
  --output-csv impact_measurement_kit_/impact_enriched.csv
```

---

## 6) Scénario de déploiement / coût

Exemple avec un pic de 120 alertes/heure :

```bash
python impact_measurement_kit_/capacity_cost_template.py \
  --impact-csv impact_measurement_kit_/impact_enriched.csv \
  --peak-alerts-per-hour 120 \
  --utilization-target 0.7 \
  --cloud-instance-hourly-eur 0.20 \
  --annual-maintenance-eur 1500
```

---

## 7) Ce que vous pourrez écrire dans le rapport

### Mesures directes
- électricité consommée pour build / train / inférence
- durée totale d'une campagne expérimentale
- nombre de CPU/GPU utilisés
- mémoire et volumes d'I/O

### Estimations ACV
- eau d'usage = kWh × facteur eau local
- ADPe d'usage = kWh × facteur ADPe du mix électrique
- part matérielle = impact machine × (heures projet / durée de vie imputée)

---

## 8) Règles de prudence

1. Ne jamais appeler "mesure directe" un chiffre eau ou ADPe.
2. Toujours séparer **usage** et **matériel**.
3. Toujours citer la source des facteurs dans le rapport.
4. Faire au moins 3 runs par étape et rapporter la moyenne.
5. Distinguer clairement **prototype**, **production**, et **scénario annuel**.