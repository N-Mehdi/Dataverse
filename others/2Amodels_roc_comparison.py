import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["obs_start", "alert_start", "decision_time"])


def train_test_split_by_alert(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    train_parts = []
    test_parts = []

    for airport, sub in df.groupby(AIRPORT_COL):
        alert_level = sub[[GROUP_COL]].drop_duplicates().copy()
        groups = alert_level[GROUP_COL].values

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        idx_train, idx_test = next(gss.split(alert_level, groups=groups))

        train_groups = set(alert_level.iloc[idx_train][GROUP_COL])
        test_groups = set(alert_level.iloc[idx_test][GROUP_COL])

        train_parts.append(sub[sub[GROUP_COL].isin(train_groups)].copy())
        test_parts.append(sub[sub[GROUP_COL].isin(test_groups)].copy())

    train_df = pd.concat(train_parts, axis=0).sort_values(
        [AIRPORT_COL, GROUP_COL, "decision_time"]
    ).reset_index(drop=True)

    test_df = pd.concat(test_parts, axis=0).sort_values(
        [AIRPORT_COL, GROUP_COL, "decision_time"]
    ).reset_index(drop=True)

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

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])

    return preprocessor


def build_models(numeric_cols, categorical_cols):
    models = {}

    # Modèles nécessitant une normalisation raisonnable
    preproc_scaled = make_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)

    models["logistic"] = Pipeline([
        ("preprocess", preproc_scaled),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )),
    ])

    models["knn"] = Pipeline([
        ("preprocess", preproc_scaled),
        ("model", KNeighborsClassifier(
            n_neighbors=25,
            weights="distance",
            p=2,
        )),
    ])

    # Arbre et forêt : la standardisation n'est pas nécessaire
    preproc_tree = make_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)

    models["cart"] = Pipeline([
        ("preprocess", preproc_tree),
        ("model", DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    models["random_forest"] = Pipeline([
        ("preprocess", preproc_tree),
        ("model", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    return models


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


def cross_validate_grouped(train_df: pd.DataFrame, models: dict, n_splits: int = 5):
    feature_cols, _, _ = build_feature_lists(train_df)

    X = train_df[feature_cols]
    y = train_df[TARGET_COL].astype(int).values
    groups = train_df[GROUP_COL].values

    gkf = GroupKFold(n_splits=n_splits)
    rows = []

    for model_name, pipe in models.items():
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            fitted = clone(pipe)
            fitted.fit(X_tr, y_tr)
            va_score = fitted.predict_proba(X_va)[:, 1]

            auc = roc_auc_score(y_va, va_score)
            tpr05, thr05 = tpr_at_fpr(y_va, va_score, fpr_target=0.05)
            tpr01, thr01 = tpr_at_fpr(y_va, va_score, fpr_target=0.01)

            rows.append({
                "model": model_name,
                "fold": fold,
                "auc": float(auc),
                "tpr_at_fpr_5pct": tpr05,
                "thr_at_fpr_5pct": thr05,
                "tpr_at_fpr_1pct": tpr01,
                "thr_at_fpr_1pct": thr01,
            })

    return pd.DataFrame(rows)


def fit_and_evaluate_models(train_df: pd.DataFrame, test_df: pd.DataFrame, models: dict):
    feature_cols, _, _ = build_feature_lists(train_df)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int).values
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].astype(int).values

    metrics_rows = []
    roc_rows = []
    pred_rows = []

    fitted_models = {}

    for model_name, pipe in models.items():
        fitted = clone(pipe)
        fitted.fit(X_train, y_train)
        fitted_models[model_name] = fitted

        test_score = fitted.predict_proba(X_test)[:, 1]

        m05 = evaluate_predictions(y_test, test_score, threshold=0.5)
        tpr5, thr5 = tpr_at_fpr(y_test, test_score, fpr_target=0.05)
        tpr1, thr1 = tpr_at_fpr(y_test, test_score, fpr_target=0.01)

        metrics_rows.append({
            "model": model_name,
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
        })

        fpr, tpr, thresholds = roc_curve(y_test, test_score)
        roc_model_df = pd.DataFrame({
            "model": model_name,
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
        })
        roc_rows.append(roc_model_df)

        pred_df = test_df[[AIRPORT_COL, "airport_alert_id", GROUP_COL, "decision_time", TARGET_COL]].copy()
        pred_df["model"] = model_name
        pred_df["score"] = test_score
        pred_rows.append(pred_df)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("auc", ascending=False).reset_index(drop=True)
    roc_df = pd.concat(roc_rows, axis=0).reset_index(drop=True)
    pred_df = pd.concat(pred_rows, axis=0).reset_index(drop=True)

    return metrics_df, roc_df, pred_df, fitted_models


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "output/silence_dataset_B.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output/model_comparison"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chargement : {input_path}")
    df = load_dataset(input_path)

    print("Split train/test groupé par alerte...")
    train_df, test_df = train_test_split_by_alert(df, test_size=0.2, random_state=42)

    print(f"Train lignes : {len(train_df)} | alertes : {train_df[GROUP_COL].nunique()}")
    print(f"Test  lignes : {len(test_df)} | alertes : {test_df[GROUP_COL].nunique()}")

    print("\nRépartition y train :")
    print(train_df[TARGET_COL].value_counts(normalize=True).sort_index())
    print("\nRépartition y test :")
    print(test_df[TARGET_COL].value_counts(normalize=True).sort_index())

    feature_cols, numeric_cols, categorical_cols = build_feature_lists(train_df)
    models = build_models(numeric_cols, categorical_cols)

    print("\nValidation croisée groupée...")
    cv_df = cross_validate_grouped(train_df, models=models, n_splits=5)
    cv_df.to_csv(out_dir / "cv_results.csv", index=False)

    cv_summary = (
        cv_df.groupby("model", as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            tpr_at_fpr_5pct_mean=("tpr_at_fpr_5pct", "mean"),
            tpr_at_fpr_1pct_mean=("tpr_at_fpr_1pct", "mean"),
        )
        .sort_values("auc_mean", ascending=False)
        .reset_index(drop=True)
    )
    cv_summary.to_csv(out_dir / "cv_summary.csv", index=False)

    print("\nRésumé CV :")
    print(cv_summary)

    print("\nEntraînement final et évaluation test...")
    metrics_df, roc_df, pred_df, _ = fit_and_evaluate_models(train_df, test_df, models=models)

    metrics_df.to_csv(out_dir / "test_metrics.csv", index=False)
    roc_df.to_csv(out_dir / "roc_points.csv", index=False)
    pred_df.to_csv(out_dir / "test_predictions_long.csv", index=False)

    print("\nMétriques test :")
    print(metrics_df)

    print(f"\nFichiers sauvegardés dans : {out_dir}")
    print("- cv_results.csv")
    print("- cv_summary.csv")
    print("- test_metrics.csv")
    print("- roc_points.csv")
    print("- test_predictions_long.csv")


if __name__ == "__main__":
    main()