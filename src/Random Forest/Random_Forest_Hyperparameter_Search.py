import sys
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold
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


def make_preprocessor(numeric_cols, categorical_cols):
    # Random Forest n'a pas besoin de scaling, mais on impute les valeurs manquantes
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


def build_model_pipeline(params, numeric_cols, categorical_cols):
    preproc = make_preprocessor(numeric_cols, categorical_cols)

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

    return Pipeline([
        ("preprocess", preproc),
        ("model", model),
    ])


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


def parameter_grid():
    """
    Grille d'hyperparamètres pour le Random Forest.

    - n_estimators     : nombre d'arbres dans la forêt
    - max_depth        : profondeur maximale de chaque arbre (None = illimité)
    - min_samples_split: nb min d'échantillons pour splitter un nœud interne
    - min_samples_leaf : nb min d'échantillons requis dans une feuille
    - max_features     : nb de features considérées à chaque split
                         ("sqrt" = racine carrée, "log2" = log base 2)
    - class_weight     : gestion du déséquilibre de classes
    """
    return [
        {
            "n_estimators": n_est,
            "max_depth": depth,
            "min_samples_split": mss,
            "min_samples_leaf": msl,
            "max_features": mf,
            "class_weight": cw,
        }
        for n_est, depth, mss, msl, mf, cw in itertools.product(
            [100, 300, 500],       # n_estimators
            [None, 10, 20],        # max_depth
            [2, 5, 10],            # min_samples_split
            [1, 2, 4],             # min_samples_leaf
            ["sqrt", "log2"],      # max_features
            [None, "balanced"],    # class_weight
        )
    ]


def cross_val_score_params(train_df, params, n_splits=5):
    feature_cols, numeric_cols, categorical_cols = build_feature_lists(train_df)

    X = train_df[feature_cols]
    y = train_df[TARGET_COL].astype(int).values
    groups = train_df[GROUP_COL].values

    gkf = GroupKFold(n_splits=n_splits)
    rows = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        pipe = build_model_pipeline(params, numeric_cols, categorical_cols)
        pipe.fit(X_tr, y_tr)
        va_score = get_scores(pipe, X_va)

        auc = roc_auc_score(y_va, va_score)
        tpr05, thr05 = tpr_at_fpr(y_va, va_score, fpr_target=0.05)
        tpr01, thr01 = tpr_at_fpr(y_va, va_score, fpr_target=0.01)

        rows.append({
            "model": "random_forest",
            "params": str(params),
            "fold": fold,
            "auc": float(auc),
            "tpr_at_fpr_5pct": tpr05,
            "thr_at_fpr_5pct": thr05,
            "tpr_at_fpr_1pct": tpr01,
            "thr_at_fpr_1pct": thr01,
        })

    return pd.DataFrame(rows)


def select_best_model(train_df, n_splits=5):
    grid = parameter_grid()
    detailed_rows = []

    best_auc = -np.inf
    best_params = None
    best_cv_df = None

    print(f"\nRecherche hyperparamètres pour random forest ({len(grid)} combinaisons)...")
    for i, params in enumerate(grid, start=1):
        try:
            cv_df = cross_val_score_params(train_df, params, n_splits=n_splits)
            mean_auc = cv_df["auc"].mean()
            detailed_rows.append(cv_df)

            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = params
                best_cv_df = cv_df

        except Exception as e:
            print(f"  [SKIP] params={params} -> {e}")

        if i % 10 == 0 or i == len(grid):
            print(f"  {i}/{len(grid)} combinaisons testées | meilleur AUC : {best_auc:.4f}")

    detailed_df = pd.concat(detailed_rows, axis=0).reset_index(drop=True)

    best_df = pd.DataFrame([{
        "model": "random_forest",
        "best_params": str(best_params),
        "cv_auc_mean": float(best_cv_df["auc"].mean()),
        "cv_auc_std": float(best_cv_df["auc"].std()),
        "cv_tpr_at_fpr_5pct_mean": float(best_cv_df["tpr_at_fpr_5pct"].mean()),
        "cv_tpr_at_fpr_1pct_mean": float(best_cv_df["tpr_at_fpr_1pct"].mean()),
    }])

    best_specs = {"random_forest": best_params}

    return best_specs, detailed_df, best_df


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "output/silence_dataset.parquet"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output/random_forest_hyper"

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

    best_specs, detailed_cv_df, best_df = select_best_model(train_df, n_splits=5)

    detailed_cv_df.to_csv(out_dir / "cv_results_detailed.csv", index=False)
    best_df.to_csv(out_dir / "cv_summary_best.csv", index=False)

    best_params_df = (
        pd.DataFrame([{"model": k, "best_params": str(v)} for k, v in best_specs.items()])
        .sort_values("model")
        .reset_index(drop=True)
    )
    best_params_df.to_csv(out_dir / "best_params.csv", index=False)

    print("\nMeilleurs hyperparamètres :")
    print(best_params_df)
    print("\nRésumé CV du meilleur modèle :")
    print(best_df)
    print(f"\nFichiers sauvegardés dans : {out_dir}")
    print("- cv_results_detailed.csv")
    print("- cv_summary_best.csv")
    print("- best_params.csv")


if __name__ == "__main__":
    main()