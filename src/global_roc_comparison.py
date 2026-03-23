import sys
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier

from xgboost import XGBClassifier


# ============================================================
# Configuration
# ============================================================

TARGET_COL = "y_h20"   # "y_h20", "y_h30" ou "y_terminal"
GROUP_COL = "alert_group"
AIRPORT_COL = "airport"
USE_AIRPORT_AS_FEATURE = True

ALL_TARGET_COLS = {"y_terminal", "y_h20", "y_h30"}

NON_FEATURE_COLS = {
    "airport_alert_id",
    "alert_group",
    "obs_start",
    "alert_start",
    "decision_time",
    "cg_reference_index",
    # On garde minutes_since_reference_cg pour la baseline silence_rule
    # et pour tester son apport réel dans les modèles
    "use_airport_context",
    *ALL_TARGET_COLS,
}


# ============================================================
# Chargement
# ============================================================

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["obs_start", "alert_start", "decision_time"])


# ============================================================
# Split
# ============================================================

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


def leave_one_airport_out_splits(df: pd.DataFrame):
    airports = sorted(df[AIRPORT_COL].dropna().unique())
    for airport in airports:
        train_df = df[df[AIRPORT_COL] != airport].copy()
        test_df = df[df[AIRPORT_COL] == airport].copy()
        if len(train_df) > 0 and len(test_df) > 0:
            yield airport, train_df, test_df


# ============================================================
# Features / preprocessing
# ============================================================

def build_feature_lists(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    if not USE_AIRPORT_AS_FEATURE and AIRPORT_COL in feature_cols:
        feature_cols.remove(AIRPORT_COL)

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


# ============================================================
# Modèles
# ============================================================

def build_model_pipeline(model_name, params, numeric_cols, categorical_cols):
    if model_name in {"logistic", "knn"}:
        preproc = make_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)
    else:
        preproc = make_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)

    if model_name == "logistic":
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver=params["solver"],
            penalty=params["penalty"],
            C=params["C"],
            random_state=42,
        )

    elif model_name == "knn":
        model = KNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            p=params["p"],
        )

    elif model_name == "cart":
        model = DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced",
            random_state=42,
        )

    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

    elif model_name == "bagging_tree":
        base_tree = DecisionTreeClassifier(
            max_depth=params["base_max_depth"],
            min_samples_leaf=params["base_min_samples_leaf"],
            random_state=42,
        )
        model = BaggingClassifier(
            estimator=base_tree,
            n_estimators=params["n_estimators"],
            max_samples=params["max_samples"],
            max_features=params["max_features"],
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )

    elif model_name == "adaboost_tree":
        base_tree = DecisionTreeClassifier(
            max_depth=params["base_max_depth"],
            min_samples_leaf=params["base_min_samples_leaf"],
            random_state=42,
        )
        model = AdaBoostClassifier(
            estimator=base_tree,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=42,
        )

    elif model_name == "xgboost":
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

    return Pipeline([
        ("preprocess", preproc),
        ("model", model),
    ])


def parameter_grid():
    return {
        "logistic": [
            {"solver": "liblinear", "penalty": "l1", "C": c}
            for c in [0.01, 0.1, 1.0, 10.0]
        ] + [
            {"solver": "liblinear", "penalty": "l2", "C": c}
            for c in [0.01, 0.1, 1.0, 10.0]
        ],

        "knn": [
            {"n_neighbors": k, "weights": w, "p": p}
            for k, w, p in itertools.product(
                [5, 15, 25, 50],
                ["uniform", "distance"],
                [1, 2],
            )
        ],

        "cart": [
            {"max_depth": d, "min_samples_leaf": leaf}
            for d, leaf in itertools.product(
                [3, 5, 7, 10, None],
                [5, 10, 20, 50],
            )
        ],

        "random_forest": [
            {
                "n_estimators": n_est,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "max_features": mf,
            }
            for n_est, depth, leaf, mf in itertools.product(
                [200, 400],
                [None, 10],
                [5, 10, 20],
                ["sqrt", 0.5],
            )
        ],

        "bagging_tree": [
            {
                "base_max_depth": depth,
                "base_min_samples_leaf": leaf,
                "n_estimators": n_est,
                "max_samples": ms,
                "max_features": mf,
            }
            for depth, leaf, n_est, ms, mf in itertools.product(
                [None, 10],
                [5, 10, 20],
                [100, 300],
                [0.8, 1.0],
                [0.8, 1.0],
            )
        ],

        "adaboost_tree": [
            {
                "base_max_depth": depth,
                "base_min_samples_leaf": leaf,
                "n_estimators": n_est,
                "learning_rate": lr,
            }
            for depth, leaf, n_est, lr in itertools.product(
                [1, 2, 3],
                [5, 10, 20],
                [100, 300],
                [0.01, 0.05, 0.1],
            )
        ],

        "xgboost": [
            {
                "n_estimators": n_est,
                "max_depth": depth,
                "learning_rate": lr,
                "subsample": subs,
                "colsample_bytree": colsub,
                "min_child_weight": mcw,
                "reg_lambda": reg_lambda,
            }
            for n_est, depth, lr, subs, colsub, mcw, reg_lambda in itertools.product(
                [100, 300],
                [3, 5, 7],
                [0.03, 0.1],
                [0.8, 1.0],
                [0.8, 1.0],
                [1, 5],
                [1.0, 5.0],
            )
        ],
    }


# ============================================================
# Scores / métriques
# ============================================================

def get_scores(fitted_model, X):
    if hasattr(fitted_model, "predict_proba"):
        return fitted_model.predict_proba(X)[:, 1]
    if hasattr(fitted_model, "decision_function"):
        return fitted_model.decision_function(X)
    raise ValueError("Le modèle ne fournit ni predict_proba ni decision_function.")


def simple_silence_score(df: pd.DataFrame) -> np.ndarray:
    x = df["minutes_since_reference_cg"].astype(float).values
    max_x = np.nanmax(x) if np.isfinite(np.nanmax(x)) else 1.0
    max_x = max(max_x, 1.0)
    return x / max_x


def tpr_at_fpr(y_true, y_score, fpr_target=0.05):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid = np.where(fpr <= fpr_target)[0]
    if len(valid) == 0:
        return 0.0, None
    idx = valid[-1]
    return float(tpr[idx]), float(thresholds[idx])


def find_threshold_for_target_fpr(y_true, y_score, target_fpr=0.05):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 1.0
    idx = valid[-1]
    return float(thresholds[idx])


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


def evaluate_alert_level(pred_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []

    grouped = pred_df.sort_values(["model", GROUP_COL, "decision_time"]).groupby(
        ["model", GROUP_COL], sort=False
    )

    for (model, alert_group), g in grouped:
        g = g.sort_values("decision_time").reset_index(drop=True)

        proposed = g[g["score"] >= threshold]
        first_raise_time = proposed["decision_time"].iloc[0] if len(proposed) > 0 else pd.NaT

        true_positive_zone = g[g[TARGET_COL] == 1]
        first_true_time = true_positive_zone["decision_time"].iloc[0] if len(true_positive_zone) > 0 else pd.NaT

        if pd.notna(first_raise_time) and pd.notna(first_true_time):
            advance_min = (first_true_time - first_raise_time).total_seconds() / 60.0
        else:
            advance_min = np.nan

        rows.append({
            "model": model,
            "alert_group": alert_group,
            "airport": g[AIRPORT_COL].iloc[0],
            "first_raise_time": first_raise_time,
            "first_true_time": first_true_time,
            "alert_has_raise": int(pd.notna(first_raise_time)),
            "alert_has_true_positive_zone": int(pd.notna(first_true_time)),
            "advance_min": advance_min,
            "false_early_raise": int(
                pd.notna(first_raise_time)
                and pd.notna(first_true_time)
                and first_raise_time < first_true_time
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# CV / sélection
# ============================================================

def cross_val_score_params(train_df, model_name, params, n_splits=5):
    feature_cols, numeric_cols, categorical_cols = build_feature_lists(train_df)

    X = train_df[feature_cols]
    y = train_df[TARGET_COL].astype(int).values
    groups = train_df[GROUP_COL].values

    gkf = GroupKFold(n_splits=n_splits)
    rows = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        pipe = build_model_pipeline(model_name, params, numeric_cols, categorical_cols)
        pipe.fit(X_tr, y_tr)
        va_score = get_scores(pipe, X_va)

        auc = roc_auc_score(y_va, va_score)
        tpr05, thr05 = tpr_at_fpr(y_va, va_score, fpr_target=0.05)
        tpr01, thr01 = tpr_at_fpr(y_va, va_score, fpr_target=0.01)

        rows.append({
            "model": model_name,
            "params": str(params),
            "fold": fold,
            "auc": float(auc),
            "tpr_at_fpr_5pct": tpr05,
            "thr_at_fpr_5pct": thr05,
            "tpr_at_fpr_1pct": tpr01,
            "thr_at_fpr_1pct": thr01,
        })

    return pd.DataFrame(rows)


def select_best_models(train_df, n_splits=5):
    grids = parameter_grid()
    detailed_rows = []
    best_rows = []
    best_specs = {}

    for model_name, grid in grids.items():
        print(f"\nRecherche hyperparamètres pour {model_name}...")
        best_primary = -np.inf
        best_secondary = -np.inf
        best_params = None
        best_cv_df = None

        for i, params in enumerate(grid, start=1):
            cv_df = cross_val_score_params(train_df, model_name, params, n_splits=n_splits)
            mean_auc = cv_df["auc"].mean()
            mean_tpr5 = cv_df["tpr_at_fpr_5pct"].mean()

            detailed_rows.append(cv_df)

            if (mean_tpr5 > best_primary) or (
                np.isclose(mean_tpr5, best_primary) and mean_auc > best_secondary
            ):
                best_primary = mean_tpr5
                best_secondary = mean_auc
                best_params = params
                best_cv_df = cv_df

            if i % 5 == 0 or i == len(grid):
                print(f"  {i}/{len(grid)} combinaisons testées")

        best_rows.append({
            "model": model_name,
            "best_params": str(best_params),
            "selection_metric": "cv_tpr_at_fpr_5pct_mean",
            "cv_auc_mean": float(best_cv_df["auc"].mean()),
            "cv_auc_std": float(best_cv_df["auc"].std()),
            "cv_tpr_at_fpr_5pct_mean": float(best_cv_df["tpr_at_fpr_5pct"].mean()),
            "cv_tpr_at_fpr_1pct_mean": float(best_cv_df["tpr_at_fpr_1pct"].mean()),
        })

        best_specs[model_name] = best_params

    detailed_df = pd.concat(detailed_rows, axis=0).reset_index(drop=True)
    best_df = pd.DataFrame(best_rows).sort_values(
        ["cv_tpr_at_fpr_5pct_mean", "cv_auc_mean"], ascending=False
    ).reset_index(drop=True)

    return best_specs, detailed_df, best_df


# ============================================================
# Entraînement final / test
# ============================================================

def fit_and_evaluate_best_models(train_df, test_df, best_specs):
    feature_cols, numeric_cols, categorical_cols = build_feature_lists(train_df)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int).values
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].astype(int).values

    metrics_rows = []
    roc_rows = []
    pred_rows = []
    alert_eval_rows = []

    # Baseline silence_rule
    silence_score = simple_silence_score(test_df)
    silence_m05 = evaluate_predictions(y_test, silence_score, threshold=0.5)
    silence_tpr5, silence_thr5 = tpr_at_fpr(y_test, silence_score, fpr_target=0.05)
    silence_tpr1, silence_thr1 = tpr_at_fpr(y_test, silence_score, fpr_target=0.01)

    metrics_rows.append({
        "model": "silence_rule",
        "best_params": "{}",
        "auc": silence_m05["auc"],
        "fpr_at_threshold_0.5": silence_m05["fpr"],
        "tpr_at_threshold_0.5": silence_m05["tpr"],
        "tpr_at_fpr_5pct": silence_tpr5,
        "threshold_at_fpr_5pct": silence_thr5,
        "tpr_at_fpr_1pct": silence_tpr1,
        "threshold_at_fpr_1pct": silence_thr1,
        "tp_at_0.5": silence_m05["tp"],
        "fp_at_0.5": silence_m05["fp"],
        "tn_at_0.5": silence_m05["tn"],
        "fn_at_0.5": silence_m05["fn"],
    })

    fpr, tpr, thresholds = roc_curve(y_test, silence_score)
    roc_rows.append(pd.DataFrame({
        "model": "silence_rule",
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
    }))

    pred_silence_df = test_df[[AIRPORT_COL, "airport_alert_id", GROUP_COL, "decision_time", TARGET_COL]].copy()
    pred_silence_df["model"] = "silence_rule"
    pred_silence_df["score"] = silence_score
    pred_rows.append(pred_silence_df)

    if silence_thr5 is not None:
        alert_eval_df = evaluate_alert_level(pred_silence_df, threshold=silence_thr5)
        alert_eval_df["threshold_used"] = silence_thr5
        alert_eval_rows.append(alert_eval_df)

    # Modèles ML
    for model_name, params in best_specs.items():
        pipe = build_model_pipeline(model_name, params, numeric_cols, categorical_cols)
        pipe.fit(X_train, y_train)
        test_score = get_scores(pipe, X_test)

        m05 = evaluate_predictions(y_test, test_score, threshold=0.5)
        tpr5, thr5 = tpr_at_fpr(y_test, test_score, fpr_target=0.05)
        tpr1, thr1 = tpr_at_fpr(y_test, test_score, fpr_target=0.01)

        metrics_rows.append({
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

        if thr5 is not None:
            alert_eval_df = evaluate_alert_level(pred_df, threshold=thr5)
            alert_eval_df["threshold_used"] = thr5
            alert_eval_rows.append(alert_eval_df)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["tpr_at_fpr_5pct", "auc"], ascending=False
    ).reset_index(drop=True)
    roc_df = pd.concat(roc_rows, axis=0).reset_index(drop=True)
    pred_df = pd.concat(pred_rows, axis=0).reset_index(drop=True)
    alert_eval_df = pd.concat(alert_eval_rows, axis=0).reset_index(drop=True)

    return metrics_df, roc_df, pred_df, alert_eval_df


# ============================================================
# Main
# ============================================================

def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "output/silence_dataset.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output/model_comparison_with_xgboost"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chargement : {input_path}")
    print(f"Cible utilisée : {TARGET_COL}")
    print(f"Airport utilisé comme feature : {USE_AIRPORT_AS_FEATURE}")
    df = load_dataset(input_path)

    print("Split train/test groupé par alerte...")
    train_df, test_df = train_test_split_by_alert(df, test_size=0.2, random_state=42)

    print(f"Train lignes : {len(train_df)} | alertes : {train_df[GROUP_COL].nunique()}")
    print(f"Test  lignes : {len(test_df)} | alertes : {test_df[GROUP_COL].nunique()}")

    print("\nRépartition y train :")
    print(train_df[TARGET_COL].value_counts(normalize=True).sort_index())
    print("\nRépartition y test :")
    print(test_df[TARGET_COL].value_counts(normalize=True).sort_index())

    print("\nSélection des hyperparamètres par CV groupée...")
    best_specs, detailed_cv_df, best_df = select_best_models(train_df, n_splits=5)

    detailed_cv_df.to_csv(out_dir / "cv_results_detailed.csv", index=False)
    best_df.to_csv(out_dir / "cv_summary_best.csv", index=False)

    best_params_df = pd.DataFrame(
        [{"model": k, "best_params": str(v)} for k, v in best_specs.items()]
    ).sort_values("model").reset_index(drop=True)
    best_params_df.to_csv(out_dir / "best_params.csv", index=False)

    print("\nMeilleurs hyperparamètres :")
    print(best_params_df)

    print("\nRésumé CV des meilleurs modèles :")
    print(best_df)

    print("\nEntraînement final des meilleurs modèles et évaluation test...")
    metrics_df, roc_df, pred_df, alert_eval_df = fit_and_evaluate_best_models(train_df, test_df, best_specs)

    metrics_df.to_csv(out_dir / "test_metrics.csv", index=False)
    roc_df.to_csv(out_dir / "roc_points.csv", index=False)
    pred_df.to_csv(out_dir / "test_predictions_long.csv", index=False)
    alert_eval_df.to_csv(out_dir / "alert_level_evaluation.csv", index=False)

    print("\nMétriques test :")
    print(metrics_df)

    print(f"\nFichiers sauvegardés dans : {out_dir}")
    print("- best_params.csv")
    print("- cv_results_detailed.csv")
    print("- cv_summary_best.csv")
    print("- test_metrics.csv")
    print("- roc_points.csv")
    print("- test_predictions_long.csv")
    print("- alert_level_evaluation.csv")


if __name__ == "__main__":
    main()