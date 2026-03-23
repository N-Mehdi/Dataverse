# run_baseline_classifier.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
    df = pd.read_csv(path, parse_dates=["obs_start", "alert_start", "decision_time"])
    return df


def train_test_split_by_alert(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split groupé par alerte, séparément dans chaque aéroport,
    pour conserver grossièrement la structure par aéroport.
    """
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

    train_df = pd.concat(train_parts, axis=0).sort_values([AIRPORT_COL, GROUP_COL, "decision_time"]).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).sort_values([AIRPORT_COL, GROUP_COL, "decision_time"]).reset_index(drop=True)
    return train_df, test_df


def build_feature_lists(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return feature_cols, numeric_cols, categorical_cols


def build_pipeline(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])
    return pipe


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
        "auc": auc,
        "threshold": threshold,
        "fpr": fpr,
        "tpr": tpr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def cross_validate_grouped(train_df: pd.DataFrame, n_splits: int = 5):
    feature_cols, numeric_cols, categorical_cols = build_feature_lists(train_df)

    X = train_df[feature_cols]
    y = train_df[TARGET_COL].astype(int).values
    groups = train_df[GROUP_COL].values

    gkf = GroupKFold(n_splits=n_splits)
    rows = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        pipe = build_pipeline(numeric_cols, categorical_cols)
        pipe.fit(X_tr, y_tr)
        va_score = pipe.predict_proba(X_va)[:, 1]

        auc = roc_auc_score(y_va, va_score)
        tpr05, thr05 = tpr_at_fpr(y_va, va_score, fpr_target=0.05)
        tpr01, thr01 = tpr_at_fpr(y_va, va_score, fpr_target=0.01)

        rows.append({
            "fold": fold,
            "auc": auc,
            "tpr_at_fpr_5pct": tpr05,
            "thr_at_fpr_5pct": thr05,
            "tpr_at_fpr_1pct": tpr01,
            "thr_at_fpr_1pct": thr01,
        })

    return pd.DataFrame(rows)


def save_coefficients(pipe: Pipeline, feature_cols, numeric_cols, categorical_cols, out_path: str):
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    feature_names = preprocess.get_feature_names_out()
    coef = model.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
    }).sort_values("abs_coefficient", ascending=False)

    coef_df.to_csv(out_path, index=False)


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "output/silence_dataset_B.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output/baseline_results"

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

    print("\nValidation croisée groupée...")
    cv_df = cross_validate_grouped(train_df, n_splits=5)
    cv_path = out_dir / "cv_results.csv"
    cv_df.to_csv(cv_path, index=False)
    print(cv_df)
    print("\nMoyennes CV :")
    print(cv_df.mean(numeric_only=True))

    feature_cols, numeric_cols, categorical_cols = build_feature_lists(train_df)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int).values
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].astype(int).values

    print("\nEntraînement final sur train complet...")
    pipe = build_pipeline(numeric_cols, categorical_cols)
    pipe.fit(X_train, y_train)

    test_score = pipe.predict_proba(X_test)[:, 1]

    test_metrics_05 = evaluate_predictions(y_test, test_score, threshold=0.5)
    tpr5, thr5 = tpr_at_fpr(y_test, test_score, fpr_target=0.05)
    tpr1, thr1 = tpr_at_fpr(y_test, test_score, fpr_target=0.01)

    metrics_df = pd.DataFrame([
        {"metric": "auc", "value": test_metrics_05["auc"]},
        {"metric": "fpr_at_threshold_0.5", "value": test_metrics_05["fpr"]},
        {"metric": "tpr_at_threshold_0.5", "value": test_metrics_05["tpr"]},
        {"metric": "tpr_at_fpr_5pct", "value": tpr5},
        {"metric": "threshold_at_fpr_5pct", "value": thr5},
        {"metric": "tpr_at_fpr_1pct", "value": tpr1},
        {"metric": "threshold_at_fpr_1pct", "value": thr1},
    ])
    metrics_path = out_dir / "test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    pred_df = test_df[[AIRPORT_COL, "airport_alert_id", GROUP_COL, "decision_time", TARGET_COL]].copy()
    pred_df["score"] = test_score
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    save_coefficients(
        pipe=pipe,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        out_path=str(out_dir / "logistic_coefficients.csv"),
    )

    print("\nMétriques test :")
    print(metrics_df)
    print(f"\nFichiers sauvegardés dans : {out_dir}")


if __name__ == "__main__":
    main()