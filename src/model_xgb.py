"""
model_xgb.py
------------
Entraîne un classifieur XGBoost sur les snapshots temporels.

Cible : y=1 si éclair CG dans les 30 prochaines minutes, y=0 sinon.
Split train/test par alerte (pas par snapshot) pour éviter la fuite de données.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import sys

from features_snapshot import FEATURE_COLS

TEST_SIZE = 0.20
RANDOM_SEED = 42


def load_snapshots(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def split_by_alert(df: pd.DataFrame):
    """Split train/test par alerte pour éviter la fuite de données temporelles."""
    alerts = df[["airport", "airport_alert_id"]].drop_duplicates()
    alerts["group_id"] = (
        alerts["airport"] + "_" + alerts["airport_alert_id"].astype(str)
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    idx_train, idx_test = next(gss.split(alerts, groups=alerts["group_id"]))

    train_alerts = set(alerts.iloc[idx_train]["group_id"])
    test_alerts = set(alerts.iloc[idx_test]["group_id"])

    df["group_id"] = df["airport"] + "_" + df["airport_alert_id"].astype(str)
    train = df[df["group_id"].isin(train_alerts)].copy()
    test = df[df["group_id"].isin(test_alerts)].copy()

    print(f"Train : {len(train)} snapshots ({train['group_id'].nunique()} alertes)")
    print(f"Test  : {len(test)} snapshots ({test['group_id'].nunique()} alertes)")
    return train, test


def train_xgb(train: pd.DataFrame) -> xgb.XGBClassifier:
    X = train[FEATURE_COLS]
    y = train["y"]

    # Gestion du déséquilibre de classes
    pos = y.sum()
    neg = (y == 0).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    print(f"scale_pos_weight : {scale_pos_weight:.2f} (pos={pos}, neg={neg})")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        early_stopping_rounds=30,
    )

    X_tr = train[FEATURE_COLS]
    y_tr = train["y"]

    # Pas de validation séparée ici — on utilise tout le train
    # Pour early stopping, on utilise 10% du train comme val interne
    from sklearn.model_selection import train_test_split

    X_t, X_v, y_t, y_v = train_test_split(
        X_tr, y_tr, test_size=0.1, random_state=RANDOM_SEED
    )
    model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=50)
    return model


def evaluate(model, test: pd.DataFrame):
    X_test = test[FEATURE_COLS]
    y_test = test["y"]

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"\nAUC-ROC (test) : {auc:.4f}")

    # Importance des features
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    imp = imp.sort_values(ascending=False)
    print("\nImportance des features (gain) :")
    print(imp.to_string())

    return auc, imp


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/snapshots.parquet"
    print(f"Chargement de {path}...")
    df = load_snapshots(path)

    print(f"\nDataset : {len(df)} snapshots")
    train, test = split_by_alert(df)

    print("\n── XGBoost ──────────────────────────────────")
    model = train_xgb(train)

    auc, imp = evaluate(model, test)

    joblib.dump(
        {"model": model, "train_alerts": train["group_id"].unique()},
        "models/xgb_model.pkl",
    )
    print("\nModèle sauvegardé : models/xgb_model.pkl")
