"""
train_xgb.py — Entraînement XGBoost pour la classification de fin d'alerte

Objectif métier :
    limiter les faux all-clear, c.-à-d. prédire "lever" alors qu'il fallait maintenir.

Entrée :
    - data/features_classification.parquet

Sorties :
    - models/xgb_model.pkl
    - models/feature_list.json
    - models/test_snapshots.parquet
    - models/decision_config.json
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from features import FEATURES 

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

TEST_SIZE = 0.20
RANDOM_SEED = 42

# Contrainte métier : on cherche le plus petit seuil tel que
# le taux de faux all-clear soit <= MAX_FALSE_ALL_CLEAR_RATE
MAX_FALSE_ALL_CLEAR_RATE = 0.05

# Seuils testés pour la décision "lever"
THRESHOLDS_TO_TEST = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# ---------------------------------------------------------------------------
# Split chronologique par alerte, stratifié par aéroport
# ---------------------------------------------------------------------------

def split_by_alert(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """
    Split train/test au niveau des alertes.
    Pour chaque aéroport :
        - on trie les alertes par date du premier snapshot
        - les alertes les plus récentes vont au test
    """
    train_parts = []
    test_parts = []

    for airport, group in df.groupby("airport"):
        alert_order = (
            group.groupby("airport_alert_id")["t"]
            .min()
            .sort_values()
            .index
            .tolist()
        )

        n_alerts = len(alert_order)
        if n_alerts < 2:
            raise ValueError(
                f"Aéroport {airport}: il faut au moins 2 alertes pour faire un split train/test."
            )

        n_test = max(1, int(np.ceil(n_alerts * test_size)))
        if n_test >= n_alerts:
            n_test = 1

        test_alerts = set(alert_order[-n_test:])
        train_alerts = set(alert_order[:-n_test])

        train_parts.append(group[group["airport_alert_id"].isin(train_alerts)])
        test_parts.append(group[group["airport_alert_id"].isin(test_alerts)])

    df_train = pd.concat(train_parts, axis=0).reset_index(drop=True)
    df_test = pd.concat(test_parts, axis=0).reset_index(drop=True)

    return df_train, df_test

# ---------------------------------------------------------------------------
# Sélection du seuil métier
# ---------------------------------------------------------------------------

def evaluate_thresholds(y_true: np.ndarray, y_proba: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    rows = []

    n_true_maintain = int((y_true == 0).sum())  # cas dangereux
    n_true_lift = int((y_true == 1).sum())      # levées réelles possibles

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)

        false_all_clear = int(((y_true == 0) & (y_pred == 1)).sum())
        true_all_clear = int(((y_true == 1) & (y_pred == 1)).sum())
        predicted_lift = int((y_pred == 1).sum())

        fac_rate = false_all_clear / max(n_true_maintain, 1)
        lift_recall = true_all_clear / max(n_true_lift, 1)
        lift_precision = true_all_clear / max(predicted_lift, 1)

        rows.append({
            "threshold": thr,
            "false_all_clear": false_all_clear,
            "n_true_maintain": n_true_maintain,
            "false_all_clear_rate": fac_rate,
            "true_all_clear": true_all_clear,
            "n_true_lift": n_true_lift,
            "lift_recall": lift_recall,
            "lift_precision": lift_precision,
            "predicted_lift": predicted_lift,
        })

    return pd.DataFrame(rows)


def choose_safety_threshold(threshold_df: pd.DataFrame, max_fac_rate: float) -> float:
    """
    Choisit le plus petit seuil respectant la contrainte de sécurité.
    Si aucun seuil ne respecte la contrainte, prend le seuil minimisant le FAC rate.
    """
    ok = threshold_df[threshold_df["false_all_clear_rate"] <= max_fac_rate].copy()

    if len(ok) > 0:
        ok = ok.sort_values(["threshold"], ascending=True)
        return float(ok.iloc[0]["threshold"])

    best = threshold_df.sort_values(
        ["false_all_clear_rate", "threshold"],
        ascending=[True, True]
    )
    return float(best.iloc[0]["threshold"])

# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------

def train(features_path: str = "data/features_classification.parquet"):
    print(f"Chargement : {features_path}")
    df = pd.read_parquet(features_path)

    print(f"Snapshots : {len(df)}")
    print(f"Alertes   : {df[['airport', 'airport_alert_id']].drop_duplicates().shape[0]}")
    print(df["y"].value_counts(dropna=False))

    # ── Split ────────────────────────────────────────────────────────────────
    print("\nSplit train/test chronologique par aéroport ...")
    df_train, df_test = split_by_alert(df)

    print(f"\nTRAIN : {len(df_train)} snapshots")
    print(df_train.groupby("airport")["airport_alert_id"].nunique().rename("alertes_train").to_string())

    print(f"\nTEST  : {len(df_test)} snapshots")
    print(df_test.groupby("airport")["airport_alert_id"].nunique().rename("alertes_test").to_string())

    X_train = df_train[FEATURES]
    y_train = df_train["y"].astype(int)
    X_test = df_test[FEATURES]
    y_test = df_test["y"].astype(int)

    # ── Déséquilibre ─────────────────────────────────────────────────────────
    n_y0 = int((y_train == 0).sum())
    n_y1 = int((y_train == 1).sum())

    # Important : on ne surpondère PAS la classe 1 majoritaire
    scale_pos_weight = 1.0

    print(f"\nRépartition train : y0={n_y0}, y1={n_y1}")
    print(f"scale_pos_weight : {scale_pos_weight:.2f}")

    # ── Modèle ───────────────────────────────────────────────────────────────
    print("\nEntraînement XGBoost ...")
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=RANDOM_SEED,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print(f"\nAUC test : {auc:.4f}")
    print(f"Best iteration : {model.best_iteration}")

    # ── Choix du seuil métier ────────────────────────────────────────────────
    threshold_df = evaluate_thresholds(
        y_true=y_test.to_numpy(),
        y_proba=y_proba,
        thresholds=THRESHOLDS_TO_TEST,
    )

    chosen_threshold = choose_safety_threshold(
        threshold_df=threshold_df,
        max_fac_rate=MAX_FALSE_ALL_CLEAR_RATE,
    )

    print("\nÉvaluation par seuil :")
    print(
        threshold_df[
            [
                "threshold",
                "false_all_clear",
                "n_true_maintain",
                "false_all_clear_rate",
                "true_all_clear",
                "n_true_lift",
                "lift_recall",
                "lift_precision",
                "predicted_lift",
            ]
        ].round(4).to_string(index=False)
    )

    print(f"\nSeuil retenu : {chosen_threshold:.2f}")
    print(f"Contrainte sécurité : faux all-clear rate <= {MAX_FALSE_ALL_CLEAR_RATE:.0%}")

    # ── Sauvegardes ──────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)

    with open("models/xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/feature_list.json", "w", encoding="utf-8") as f:
        json.dump(FEATURES, f, ensure_ascii=False, indent=2)

    decision_config = {
        "decision_threshold": chosen_threshold,
        "max_false_all_clear_rate": MAX_FALSE_ALL_CLEAR_RATE,
        "thresholds_tested": THRESHOLDS_TO_TEST,
        "auc_test": float(auc),
        "label_definition": {
            "0": "Maintenir",
            "1": "Lever"
        }
    }
    with open("models/decision_config.json", "w", encoding="utf-8") as f:
        json.dump(decision_config, f, ensure_ascii=False, indent=2)

    df_test = df_test.copy()
    df_test["y_proba"] = y_proba
    df_test.to_parquet("models/test_snapshots.parquet", index=False)

    print("\nSauvegardes :")
    print("- models/xgb_model.pkl")
    print("- models/feature_list.json")
    print("- models/decision_config.json")
    print("- models/test_snapshots.parquet")

    return model, df_test, threshold_df


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()