import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import joblib


TARGET_COL = "y"
GROUP_COL = "alert_group"
AIRPORT_COL = "airport"

NON_FEATURE_COLS = {
    "airport_alert_id",
    "alert_group",
    "obs_start",
    "alert_start",
    "decision_time",
    "cg_reference_index",
    "minutes_since_reference_cg",
    "y",
}

BEST_SPECS = {
    "random_forest": {
        "n_estimators": None,       # à remplir après recherche hyperparamètres
        "max_depth": None,          # à remplir après recherche hyperparamètres
        "min_samples_split": None,  # à remplir après recherche hyperparamètres
        "min_samples_leaf": None,   # à remplir après recherche hyperparamètres
        "max_features": None,       # à remplir après recherche hyperparamètres
        "class_weight": None,       # à remplir après recherche hyperparamètres
    }
}


def load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["obs_start", "alert_start", "decision_time"])


def build_feature_lists(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return feature_cols, numeric_cols, categorical_cols


def make_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def build_model_pipeline(model_name, params, numeric_cols, categorical_cols):
    preproc = make_preprocessor(numeric_cols, categorical_cols)

    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    return Pipeline([
        ("preprocess", preproc),
        ("model", model),
    ])


def fit_final_models(df, best_specs, out_dir):
    feature_cols, numeric_cols, categorical_cols = build_feature_lists(df)

    X = df[feature_cols]
    y = df[TARGET_COL].astype(int).values

    trained_models = {}

    for model_name, params in best_specs.items():
        pipe = build_model_pipeline(model_name, params, numeric_cols, categorical_cols)
        pipe.fit(X, y)
        joblib.dump(pipe, out_dir / f"model_{model_name}_full.pkl")
        trained_models[model_name] = pipe

    return trained_models, feature_cols


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "output/silence_dataset.parquet"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output/model_full_with_random_forest"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chargement : {input_path}")
    df = load_dataset(input_path)

    print(f"Nb lignes : {len(df)}")
    print(f"Nb alertes : {df[GROUP_COL].nunique()}")
    print(f"Nb aéroports : {df[AIRPORT_COL].nunique()}")

    if "has_lre_before" in df.columns:
        alert_lre = (
            df.groupby(GROUP_COL)["has_lre_before"].max().fillna(0).astype(int)
        )
        print("\nRépartition alertes LRE :")
        print(alert_lre.value_counts(normalize=True).sort_index())

    print("\nRépartition y globale :")
    print(df[TARGET_COL].value_counts(normalize=True).sort_index())

    best_specs = BEST_SPECS
    print(f"\nHyperparamètres utilisés : {best_specs}")

    print("\nEntraînement final sur 100% des données...")
    trained_models, feature_cols = fit_final_models(df, best_specs, out_dir)

    feature_df = pd.DataFrame({"feature": feature_cols})
    feature_df.to_csv(out_dir / "feature_columns.csv", index=False)

    print(f"\nFichiers sauvegardés dans : {out_dir}")
    for model_name in trained_models:
        print(f"- model_{model_name}_full.pkl")
    print("- feature_columns.csv")


if __name__ == "__main__":
    main()