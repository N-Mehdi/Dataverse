#!/bin/bash
# compare_rse_models.sh - Mesure RSE des 3 modèles + comparaison
# Usage : bash impact_measurement_kit_/compare_rse_models.sh

set -e
source .venv/bin/activate

PARQUET="output/silence_dataset.parquet"
BATCH=1000

declare -A MODELS
MODELS["xgboost"]="output/model_full_with_xgboost/model_xgboost_full.pkl"
MODELS["random_forest"]="output/model_full_with_random_forest/model_random_forest_full.pkl"
MODELS["logistic"]="output/model_full_with_logistic/model_logistic_full.pkl"

for MODEL_NAME in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$MODEL_NAME]}"
    OUT_DIR="impact_measurement_kit_/impact_runs_${MODEL_NAME}"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "[SKIP] $MODEL_NAME - fichier introuvable : $MODEL_PATH"
        continue
    fi

    echo ""
    echo "========================================"
    echo "Mesure : $MODEL_NAME"
    echo "========================================"

    python impact_measurement_kit_/benchmark_inference.py \
        --model-path "$MODEL_PATH" \
        --parquet-path "$PARQUET" \
        --batch-size $BATCH \
        --output-dir "$OUT_DIR"
done

echo ""
echo "========================================"
echo "Comparaison RSE des modèles"
echo "========================================"

python - <<'EOF'
import json
import pandas as pd
from pathlib import Path

models = ["xgboost", "random_forest", "logistic"]
rows = []

for model in models:
    run_dir = Path(f"impact_measurement_kit_/impact_runs_{model}")
    if not run_dir.exists():
        continue
    for meta_path in sorted(run_dir.glob("*_meta.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        rows.append({
            "modele":          model,
            "run":             meta["run_name"],
            "duree_s":         meta.get("elapsed_seconds"),
            "ram_max_gb":      meta.get("max_rss_gb"),
            "cpu_moy_%":       meta.get("avg_cpu_percent"),
            "cpu_max_%":       meta.get("max_cpu_percent"),
            "emissions_kgco2": meta.get("emissions_kg_co2eq"),
        })

df = pd.DataFrame(rows).sort_values(["run", "modele"]).reset_index(drop=True)

print("\n── Résultats complets ──────────────────────────────────────────")
print(df.to_string(index=False))

# Comparaison sur infer_1000 uniquement (le plus représentatif)
batch = df[df["run"] == "infer_1000_alerts"].copy()

if not batch.empty:
    # Score RSE composite : normalise chaque métrique et fait la moyenne
    metrics = ["duree_s", "ram_max_gb", "cpu_moy_%", "emissions_kgco2"]
    for m in metrics:
        col_min = batch[m].min()
        col_max = batch[m].max()
        if col_max > col_min:
            batch[f"_score_{m}"] = (batch[m] - col_min) / (col_max - col_min)
        else:
            batch[f"_score_{m}"] = 0.0
    score_cols = [f"_score_{m}" for m in metrics]
    batch["score_rse"] = batch[score_cols].mean(axis=1).round(4)

    best = batch.loc[batch["score_rse"].idxmin(), "modele"]
    worst = batch.loc[batch["score_rse"].idxmax(), "modele"]

    print("\n── Score RSE (batch 1000 alertes - plus bas = plus sobre) ─────")
    print(batch[["modele", "duree_s", "ram_max_gb", "cpu_moy_%", "emissions_kgco2", "score_rse"]].to_string(index=False))

    print(f"\n✅  Modèle le plus sobre  : {best.upper()}")
    print(f"❌  Modèle le moins sobre : {worst.upper()}")

out = "impact_measurement_kit_/rse_comparison.csv"
df.to_csv(out, index=False)
print(f"\nSauvegardé : {out}")
EOF

echo ""
echo "Terminé."