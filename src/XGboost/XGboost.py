import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

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
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
    }
}


def load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["obs_start", "alert_start", "decision_time"])


def train_test_split_by_alert(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    train_parts = []
    test_parts = []

    for airport, sub in df.groupby(AIRPORT_COL):
        alert_level = (
            sub.groupby(GROUP_COL, as_index=False)
            .agg(
                airport=(AIRPORT_COL, "first"),
                airport_alert_id=("airport_alert_id", "first"),
                has_lre_alert=("has_lre_before", "max"),
            )
            .copy()
        )

        alert_level["has_lre_alert"] = (
            alert_level["has_lre_alert"].fillna(0).astype(int)
        )

        class_counts = alert_level["has_lre_alert"].value_counts()
        can_stratify = len(class_counts) >= 2 and class_counts.min() >= 2

        if can_stratify:
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            idx_train, idx_test = next(
                sss.split(alert_level, alert_level["has_lre_alert"])
            )
        else:
            n_alerts = len(alert_level)
            rng = np.random.RandomState(random_state)
            perm = rng.permutation(n_alerts)
            n_test = max(1, int(round(test_size * n_alerts)))
            idx_test = perm[:n_test]
            idx_train = perm[n_test:]

        train_groups = set(alert_level.iloc[idx_train][GROUP_COL])
        test_groups = set(alert_level.iloc[idx_test][GROUP_COL])

        train_parts.append(sub[sub[GROUP_COL].isin(train_groups)].copy())
        test_parts.append(sub[sub[GROUP_COL].isin(test_groups)].copy())

    train_df = (
        pd.concat(train_parts, axis=0)
        .sort_values([AIRPORT_COL, GROUP_COL, "decision_time"])
        .reset_index(drop=True)
    )

    test_df = (
        pd.concat(test_parts, axis=0)
        .sort_values([AIRPORT_COL, GROUP_COL, "decision_time"])
        .reset_index(drop=True)
    )

    return train_df, test_df


def build_feature_lists(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return feature_cols, numeric_cols, categorical_cols


def make_preprocessor(numeric_cols, categorical_cols, scale_numeric=True):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    return preprocessor


def build_model_pipeline(model_name, params, numeric_cols, categorical_cols):
    preproc = make_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)

    if model_name == "xgboost":
        model = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            reg_lambda=params["reg_lambda"],
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    return Pipeline(
        [
            ("preprocess", preproc),
            ("model", model),
        ]
    )


def get_scores(fitted_model, X):
    if hasattr(fitted_model, "predict_proba"):
        return fitted_model.predict_proba(X)[:, 1]
    if hasattr(fitted_model, "decision_function"):
        return fitted_model.decision_function(X)
    raise ValueError("Le modèle ne fournit ni predict_proba ni decision_function.")


def tpr_at_fpr(y_true, y_score, fpr_target=0.05):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid = np.where(fpr <= fpr_target)[0]
    if len(valid) == 0:
        return 0.0, None
    idx = valid[-1]
    return float(tpr[idx]), float(thresholds[idx])


def evaluate_predictions(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_score)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "auc": float(auc),
        "threshold": float(threshold),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def fit_and_evaluate_best_models(train_df, test_df, best_specs, out_dir):
    feature_cols, numeric_cols, categorical_cols = build_feature_lists(train_df)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int).values
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].astype(int).values

    metrics_rows = []
    roc_rows = []
    pred_rows = []

    for model_name, params in best_specs.items():
        pipe = build_model_pipeline(model_name, params, numeric_cols, categorical_cols)
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, out_dir / f"model_{model_name}.pkl")
        test_score = get_scores(pipe, X_test)

        m05 = evaluate_predictions(y_test, test_score, threshold=0.5)
        tpr5, thr5 = tpr_at_fpr(y_test, test_score, fpr_target=0.05)
        tpr1, thr1 = tpr_at_fpr(y_test, test_score, fpr_target=0.01)

        metrics_rows.append(
            {
                "model": model_name,
                "best_params": str(params),
                "auc": m05["auc"],
                "fpr_at_threshold_0.5": m05["fpr"],
                "tpr_at_threshold_0.5": m05["tpr"],
                "tpr_at_fpr_5pct": tpr5,
                "threshold_at_fpr_5pct": thr5,
                "tpr_at_fpr_1pct": tpr1,
                "threshold_at_fpr_1pct": thr1,
                "tp_at_0.5": m05["tp"],
                "fp_at_0.5": m05["fp"],
                "tn_at_0.5": m05["tn"],
                "fn_at_0.5": m05["fn"],
            }
        )

        fpr, tpr, thresholds = roc_curve(y_test, test_score)
        roc_model_df = pd.DataFrame(
            {
                "model": model_name,
                "fpr": fpr,
                "tpr": tpr,
                "threshold": thresholds,
            }
        )
        roc_rows.append(roc_model_df)

        pred_df = test_df[
            [AIRPORT_COL, "airport_alert_id", GROUP_COL, "decision_time", TARGET_COL]
        ].copy()
        pred_df["model"] = model_name
        pred_df["score"] = test_score
        pred_rows.append(pred_df)

    metrics_df = (
        pd.DataFrame(metrics_rows)
        .sort_values("auc", ascending=False)
        .reset_index(drop=True)
    )
    roc_df = pd.concat(roc_rows, axis=0).reset_index(drop=True)
    pred_df = pd.concat(pred_rows, axis=0).reset_index(drop=True)

    return metrics_df, roc_df, pred_df


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "output/silence_dataset.parquet"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output/model_comparison_with_xgboost"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chargement : {input_path}")
    df = load_dataset(input_path)

    print("Split train/test groupé par alerte et stratifié sur les LRE...")
    train_df, test_df = train_test_split_by_alert(df, test_size=0.2, random_state=42)

    print(f"Train lignes : {len(train_df)} | alertes : {train_df[GROUP_COL].nunique()}")
    print(f"Test  lignes : {len(test_df)} | alertes : {test_df[GROUP_COL].nunique()}")

    train_alert_lre = (
        train_df.groupby(GROUP_COL)["has_lre_before"].max().fillna(0).astype(int)
    )
    test_alert_lre = (
        test_df.groupby(GROUP_COL)["has_lre_before"].max().fillna(0).astype(int)
    )

    print("\nRépartition alertes LRE train :")
    print(train_alert_lre.value_counts(normalize=True).sort_index())

    print("\nRépartition alertes LRE test :")
    print(test_alert_lre.value_counts(normalize=True).sort_index())

    print("\nRépartition y train :")
    print(train_df[TARGET_COL].value_counts(normalize=True).sort_index())
    print("\nRépartition y test :")
    print(test_df[TARGET_COL].value_counts(normalize=True).sort_index())

    best_specs = BEST_SPECS
    print(f"\nHyperparamètres utilisés : {best_specs}")

    print("\nEntraînement final des meilleurs modèles et évaluation test...")
    metrics_df, roc_df, pred_df = fit_and_evaluate_best_models(
        train_df, test_df, best_specs, out_dir
    )

    metrics_df.to_csv(out_dir / "test_metrics.csv", index=False)
    roc_df.to_csv(out_dir / "roc_points.csv", index=False)
    pred_df.to_csv(out_dir / "test_predictions_long.csv", index=False)

    print("\nMétriques test :")
    print(metrics_df)

    print(f"\nFichiers sauvegardés dans : {out_dir}")
    print("- test_metrics.csv")
    print("- roc_points.csv")
    print("- test_predictions_long.csv")


if __name__ == "__main__":
    main()