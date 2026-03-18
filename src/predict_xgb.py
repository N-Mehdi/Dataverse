"""
predict_xgb.py — Prédiction sur snapshots + export graphique PNG

Entrée :
    - parquet de snapshots

Sorties :
    - data/predictions.parquet
    - outputs/predictions_hist.png
"""

import os
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from features import FEATURES


def predict_on_snapshots(
    snapshots_path: str = "data/features_classification.parquet",
    model_path: str = "models/xgb_model.pkl",
    decision_path: str = "models/decision_config.json",
    out_parquet: str = "data/predictions.parquet",
    out_png: str = "outputs/predictions_hist.png",
):
    # Chargement modèle
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(decision_path, "r", encoding="utf-8") as f:
        decision_config = json.load(f)

    decision_threshold = float(decision_config["decision_threshold"])

    # Chargement snapshots
    df = pd.read_parquet(snapshots_path)

    # Prédictions
    X = df[FEATURES]
    df = df.copy()
    df["y_proba"] = model.predict_proba(X)[:, 1]
    df["y_pred"] = (df["y_proba"] >= decision_threshold).astype(int)

    # Sauvegarde parquet
    os.makedirs("data", exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"Prédictions sauvegardées dans {out_parquet}")

    # Graphique PNG
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.hist(df["y_proba"], bins=30)
    plt.axvline(
        decision_threshold,
        linestyle="--",
        linewidth=2,
        label=f"Seuil = {decision_threshold:.2f}"
    )
    plt.xlabel("Probabilité prédite de lever l'alerte")
    plt.ylabel("Nombre de snapshots")
    plt.title("Distribution des probabilités prédites")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"Graphique sauvegardé dans {out_png}")

    return df


if __name__ == "__main__":
    predict_on_snapshots()