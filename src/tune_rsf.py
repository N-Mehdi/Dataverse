"""
tune_rsf.py
-----------
Grid search sur les hyperparamètres du Random Survival Forest.
Objectif : réduire les faux all-clear tout en maintenant le gain.

Score = gain_moyen - PENALITE * faux_allclear
On cherche le meilleur compromis gain/sécurité.

Usage :
    python tune_rsf.py data/features.parquet
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

sys.path.append("src")
from features import ACTIVE_FEATURES

# ── Paramètres ───────────────────────────────────────────────────────────────
TEST_SIZE = 0.20
RANDOM_SEED = 42
TIMES_ABS = np.arange(0, 91, 1, dtype=float)
PENALITE = 3.0  # 1% faux all-clear = -3 min de gain dans le score

SEUILS_PAR_AIRPORT = {
    "Ajaccio": 0.81,
    "Bastia": 0.76,
    "Biarritz": 0.79,
    "Nantes": 0.83,
    "Pise": 0.76,
}

# Grille d'hyperparamètres à explorer
PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "min_samples_leaf": [5, 10, 20, 30],
    "min_samples_split": [10, 20],
    "max_depth": [None, 5, 8],
}


def evaluate_config(
    rsf, scaler, X_test, durations, airports, seuils=SEUILS_PAR_AIRPORT
):
    """Évalue un modèle RSF : retourne gain moyen, faux all-clear, c-index."""
    X_scaled = scaler.transform(X_test)

    y_test = Surv.from_arrays(event=np.ones(len(durations), dtype=bool), time=durations)
    c_stat, _, _, _, _ = concordance_index_censored(
        y_test["event"], y_test["time"], rsf.predict(X_scaled)
    )

    sf_funcs = rsf.predict_survival_function(X_scaled)
    S_all = np.array(
        [np.interp(TIMES_ABS, sf.x, sf(sf.x), right=0.0) for sf in sf_funcs]
    )

    gains, faux = [], []
    for i, (dur, airport) in enumerate(zip(durations, airports)):
        seuil = seuils.get(airport, 0.80)
        prob_end = 1.0 - S_all[i]
        idx = np.argmax(prob_end >= seuil)
        if prob_end[-1] < seuil:
            t_lever = 30.0  # fallback = baseline, gain = 0
            gains.append(0.0)
            faux.append(False)
            continue
        t_lever = TIMES_ABS[idx]
        gains.append(30.0 - t_lever)
        faux.append(t_lever < dur)

    gain_mean = np.mean(gains)
    faux_rate = np.mean(faux)
    score = gain_mean - PENALITE * faux_rate * 100

    return {
        "c_index": round(c_stat, 4),
        "gain_mean": round(gain_mean, 2),
        "faux_rate": round(faux_rate, 4),
        "score": round(score, 3),
    }


def tune(features_path: str):
    # ── Chargement & split ───────────────────────────────────────────────────
    df = pd.read_parquet(features_path)
    df = df.dropna(subset=ACTIVE_FEATURES + ["duration", "event"])
    df = df[df["duration"] > 0]

    train, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df["airport"]
    )

    X_train = train[ACTIVE_FEATURES].values
    X_test = test[ACTIVE_FEATURES].values
    durations = test["duration"].values
    airports = test["airport"].values

    y_train = Surv.from_arrays(
        event=train["event"].astype(bool), time=train["duration"]
    )

    # Baseline actuelle
    print("── Baseline (config actuelle) ──────────────────────────────────")
    scaler_base = StandardScaler()
    X_train_s = scaler_base.fit_transform(X_train)
    rsf_base = RandomSurvivalForest(
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    rsf_base.fit(X_train_s, y_train)
    base_metrics = evaluate_config(rsf_base, scaler_base, X_test, durations, airports)
    print(
        f"  C-index={base_metrics['c_index']} | gain={base_metrics['gain_mean']:+.2f} min "
        f"| faux={base_metrics['faux_rate']:.1%} | score={base_metrics['score']:.2f}"
    )

    # ── Grid search ──────────────────────────────────────────────────────────
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    configs = list(product(*values))
    total = len(configs)

    print(f"\n── Grid search : {total} configurations ────────────────────────")
    results = []

    for i, config in enumerate(configs):
        params = dict(zip(keys, config))
        print(f"  [{i + 1:3d}/{total}] {params} ", end="", flush=True)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)

        rsf = RandomSurvivalForest(n_jobs=-1, random_state=RANDOM_SEED, **params)
        rsf.fit(X_tr_s, y_train)

        metrics = evaluate_config(rsf, scaler, X_test, durations, airports)
        results.append({**params, **metrics})
        print(
            f"→ gain={metrics['gain_mean']:+.2f} | faux={metrics['faux_rate']:.1%} "
            f"| score={metrics['score']:.2f}"
        )

    # ── Résultats ─────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values("score", ascending=False)

    print(f"\n{'═' * 70}")
    print("  TOP 5 CONFIGURATIONS (score = gain - pénalité × faux_all_clear)")
    print(f"{'═' * 70}")
    print(results_df.head(5).to_string(index=False))

    print(f"\n{'─' * 70}")
    print("  MEILLEURE CONFIG")
    print(f"{'─' * 70}")
    best = results_df.iloc[0]
    print(f"  n_estimators     : {int(best['n_estimators'])}")
    print(f"  min_samples_leaf : {int(best['min_samples_leaf'])}")
    print(f"  min_samples_split: {int(best['min_samples_split'])}")
    print(f"  max_depth        : {best['max_depth']}")
    print(f"  → C-index : {best['c_index']}")
    print(f"  → Gain    : {best['gain_mean']:+.2f} min")
    print(f"  → Faux    : {best['faux_rate']:.1%}")
    print(f"  → Score   : {best['score']:.2f}")

    # Sauvegarde
    out = "outputs/tune_rsf_results.csv"
    results_df.to_csv(out, index=False)
    print(f"\nRésultats complets sauvegardés : {out}")

    return results_df


if __name__ == "__main__":
    features_path = sys.argv[1] if len(sys.argv) > 1 else "data/features.parquet"
    tune(features_path)
