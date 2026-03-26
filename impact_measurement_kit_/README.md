# Kit de mesure d'impacts pour le projet Météorage

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

## 1) Installation minimale
```bash
pip install codecarbon psutil pandas pyarrow pyyaml
```

## 2) Mesurer votre pipeline réel
Adaptez les commandes si vos scripts sont dans `src/`.

```bash
python impact_campaign.py \
  --project-root /chemin/vers/votre/projet \
  --build-cmd "python build_silence_dataset.py" \
  --train-cmd "python global_roc_comparison.py" \
  --predict-cmd "python predict.py" \
  --output-dir impact_runs
```

Le dossier `impact_runs/` contiendra un JSON par run et un CSV CodeCarbon par run.

## 3) Mesurer l'inférence
### Cas A — vous pouvez charger votre modèle sauvegardé
```bash
python benchmark_inference.py \
  --model-path /chemin/vers/model_xgboost.pkl \
  --parquet-path /chemin/vers/output/silence_dataset.parquet \
  --batch-size 1000 \
  --output-dir impact_runs
```

### Cas B — votre fonction d'inférence n'est pas encore câblée
```bash
python benchmark_inference.py --dummy --batch-size 1000 --output-dir impact_runs
```

## 4) Consolider les mesures
```bash
python aggregate_results.py
```

Cela produit `impact_summary.csv`.

## 5) Ajouter les facteurs ACV documentés
Complétez :
- `hardware_inventory_template.yaml`
- `lca_factors_example.yaml`

Puis lancez :
```bash
python estimate_lca_impacts.py \
  --summary-csv impact_summary.csv \
  --config lca_factors_example.yaml \
  --output-csv impact_enriched.csv
```

## 6) Faire un scénario de déploiement / coût
Exemple avec un pic de 120 alertes/heure :
```bash
python capacity_cost_template.py \
  --impact-csv impact_enriched.csv \
  --peak-alerts-per-hour 120 \
  --utilization-target 0.7 \
  --cloud-instance-hourly-eur 0.20 \
  --annual-maintenance-eur 1500
```

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

## 8) Règles de prudence
1. Ne jamais appeler "mesure directe" un chiffre eau ou ADPe.
2. Toujours séparer **usage** et **matériel**.
3. Toujours citer la source des facteurs dans le rapport.
4. Faire au moins 3 runs par étape et rapporter la moyenne.
5. Distinguer clairement **prototype**, **production**, et **scénario annuel**.
