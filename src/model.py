"""
model.py
--------
Entraînement et évaluation des modèles de survie pour la prédiction
de fin d'alerte orageuse.

Modèles implémentés :
  1. Kaplan-Meier           → exploration, pas de features
  2. Cox PH                 → baseline interprétable (lifelines)
  3. Random Survival Forest → modèle non-linéaire (scikit-survival)

Métriques :
  - Concordance index (C-index) : capacité à bien ordonner les durées
  - Brier Score                 : qualité de la calibration probabiliste

Usage :
    python src/model.py data/features.parquet
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Importer les listes de features depuis features.py
import sys

sys.path.append(str(Path(__file__).parent))
from features import ACTIVE_FEATURES

TARGET_DURATION = "duration"
TARGET_EVENT = "event"

# Features utilisées par RSF (toutes les actives)
FEATURE_COLS = ACTIVE_FEATURES

# Features utilisées par Cox (sous-ensemble sans variance nulle)
COX_FEATURE_COLS = [
    "n_cg_total",
    "amp_max",
    "amp_mean",
    "dist_min",
    "dist_mean",
    "n_cg_recent",
    "dist_recent_min",
    "activity_trend",
]


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_DURATION, TARGET_EVENT])
    df = df[df[TARGET_DURATION] > 0]
    print(f"Dataset chargé : {len(df)} alertes")
    return df


# ---------------------------------------------------------------------------
# Split train / test
# ---------------------------------------------------------------------------


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["airport"]
    )
    print(f"Train : {len(train)} alertes | Test : {len(test)} alertes")
    return train, test


# ---------------------------------------------------------------------------
# 1. Kaplan-Meier (exploration)
# ---------------------------------------------------------------------------


def fit_kaplan_meier(df: pd.DataFrame, save_fig: bool = True):
    """Courbe de survie globale — pas de features, distribution empirique."""
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=df[TARGET_DURATION],
        event_observed=df[TARGET_EVENT],
        label="Toutes alertes",
    )

    print("\n── Kaplan-Meier ──────────────────────────────")
    print(f"Durée médiane d'alerte : {kmf.median_survival_time_:.1f} minutes")

    if save_fig:
        fig, ax = plt.subplots(figsize=(8, 5))
        kmf.plot_survival_function(ax=ax)
        ax.axvline(x=30, color="red", linestyle="--", label="Règle 30 min (baseline)")
        ax.set_xlabel("Temps depuis début d'alerte (minutes)")
        ax.set_ylabel("P(alerte encore active)")
        ax.set_title("Courbe de survie Kaplan-Meier")
        ax.legend()
        Path("outputs").mkdir(exist_ok=True)
        fig.savefig("outputs/kaplan_meier.png", dpi=150, bbox_inches="tight")
        print("Figure sauvegardée : outputs/kaplan_meier.png")

    return kmf


# ---------------------------------------------------------------------------
# 2. Cox Proportional Hazards (baseline interprétable)
# ---------------------------------------------------------------------------


def fit_cox(train: pd.DataFrame, test: pd.DataFrame):
    """
    Modèle de Cox — interprétable, rapide, bonne baseline.
    Utilise COX_FEATURE_COLS (sous-ensemble sans variance nulle).
    """
    cols = COX_FEATURE_COLS + [TARGET_DURATION, TARGET_EVENT]
    train_cox = train[cols].copy()
    test_cox = test[cols].copy()

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_cox, duration_col=TARGET_DURATION, event_col=TARGET_EVENT)

    print("\n── Cox PH ────────────────────────────────────")
    cph.print_summary(decimals=3)

    c_index = concordance_index(
        test_cox[TARGET_DURATION],
        -cph.predict_partial_hazard(test_cox),
        test_cox[TARGET_EVENT],
    )
    print(f"\nC-index (test) : {c_index:.4f}  (0.5 = aléatoire, 1.0 = parfait)")

    return cph


# ---------------------------------------------------------------------------
# 3. Random Survival Forest
# ---------------------------------------------------------------------------


def fit_rsf(train: pd.DataFrame, test: pd.DataFrame, n_estimators: int = 100):
    """
    Random Survival Forest — non-linéaire, capture les interactions,
    pas d'hypothèse proportionnelle.
    """
    X_train = train[FEATURE_COLS].values
    X_test = test[FEATURE_COLS].values

    y_train = Surv.from_arrays(
        event=train[TARGET_EVENT].astype(bool), time=train[TARGET_DURATION]
    )
    y_test = Surv.from_arrays(
        event=test[TARGET_EVENT].astype(bool), time=test[TARGET_DURATION]
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    rsf.fit(X_train, y_train)

    c_stat, _, _, _, _ = concordance_index_censored(
        test[TARGET_EVENT].astype(bool), test[TARGET_DURATION], rsf.predict(X_test)
    )
    print("\n── Random Survival Forest ────────────────────")
    print(f"C-index (test) : {c_stat:.4f}")

    # Importance par permutation (feature_importances_ non dispo dans sksurv)
    perm = permutation_importance(
        rsf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1
    )
    importances = pd.Series(perm.importances_mean, index=FEATURE_COLS)
    print("\nImportance des features (permutation) :")
    print(importances.sort_values(ascending=False).to_string())

    return rsf, scaler


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------


def save_model(model, path: str, scaler=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, path)
    print(f"\nModèle sauvegardé : {path}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/features.parquet"

    df = load_features(path)
    train, test = split_data(df)

    kmf = fit_kaplan_meier(df)

    cph = fit_cox(train, test)
    save_model(cph, "models/cox_model.pkl")

    rsf, scaler = fit_rsf(train, test, n_estimators=100)
    save_model(rsf, "models/rsf_model.pkl", scaler=scaler)
