"""
evaluate_xgb.py — Évaluation du modèle XGBoost pour la fin d'alerte

Évalue :
    - AUC
    - Matrice de confusion au seuil retenu
    - Rapport de classification
    - Faux all-clear globaux
    - Faux all-clear par aéroport
    - Importance des features
"""

import json
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from features import FEATURES 

# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_artifacts(
    model_path="models/xgb_model.pkl",
    test_path="models/test_snapshots.parquet",
    decision_path="models/decision_config.json",
):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df_test = pd.read_parquet(test_path)

    with open(decision_path, "r", encoding="utf-8") as f:
        decision_config = json.load(f)

    return model, df_test, decision_config

# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def evaluate():
    model, df_test, decision_config = load_artifacts()

    decision_threshold = float(decision_config["decision_threshold"])

    X_test = df_test[FEATURES]
    y_test = df_test["y"].astype(int).to_numpy()

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= decision_threshold).astype(int)

    auc = roc_auc_score(y_test, y_proba)

    print(f"AUC : {auc:.4f}")
    print(f"Seuil de décision retenu : {decision_threshold:.2f}")

    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    print("\nRapport classification :")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Maintenir (0)", "Lever (1)"],
            zero_division=0,
        )
    )

    # ── Faux all-clear globaux ───────────────────────────────────────────────
    false_all_clear = int(((y_test == 0) & (y_pred == 1)).sum())
    n_true_maintain = int((y_test == 0).sum())
    fac_rate = false_all_clear / max(n_true_maintain, 1)

    true_all_clear = int(((y_test == 1) & (y_pred == 1)).sum())
    n_true_lift = int((y_test == 1).sum())
    lift_recall = true_all_clear / max(n_true_lift, 1)

    print("\nIndicateurs métier :")
    print(f"Faux all-clear : {false_all_clear}/{n_true_maintain} ({100 * fac_rate:.2f}%)")
    print(f"Levées correctement autorisées : {true_all_clear}/{n_true_lift} ({100 * lift_recall:.2f}%)")

    # ── Faux all-clear par aéroport ──────────────────────────────────────────
    print("\nFaux all-clear par aéroport :")
    df_eval = df_test.copy()
    df_eval["y_proba"] = y_proba
    df_eval["y_pred"] = y_pred

    rows = []
    for airport, grp in df_eval.groupby("airport"):
        n_fac = int(((grp["y"] == 0) & (grp["y_pred"] == 1)).sum())
        n_danger = int((grp["y"] == 0).sum())
        fac_rate_airport = n_fac / max(n_danger, 1)

        n_true_lift_airport = int((grp["y"] == 1).sum())
        n_good_lift = int(((grp["y"] == 1) & (grp["y_pred"] == 1)).sum())
        lift_recall_airport = n_good_lift / max(n_true_lift_airport, 1)

        rows.append({
            "airport": airport,
            "false_all_clear": n_fac,
            "n_true_maintain": n_danger,
            "false_all_clear_rate": fac_rate_airport,
            "true_all_clear": n_good_lift,
            "n_true_lift": n_true_lift_airport,
            "lift_recall": lift_recall_airport,
        })

    airport_report = pd.DataFrame(rows).sort_values("false_all_clear_rate", ascending=False)
    print(airport_report.round(4).to_string(index=False))

    # ── Importance des features ──────────────────────────────────────────────
    print("\nTop 10 features :")
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    print(importances.sort_values(ascending=False).head(10).round(4).to_string())

    # ── Courbe ROC ───────────────────────────────────────────────────────────
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_df = pd.DataFrame({
        "threshold": thresholds,
        "fpr": fpr,
        "tpr": tpr,
    })
    roc_df.to_csv("models/roc_points.csv", index=False)
    print("\nPoints ROC sauvegardés dans models/roc_points.csv")


if __name__ == "__main__":
    evaluate()