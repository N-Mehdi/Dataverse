"""
test_baseline_tslcg.py
----------------------
Teste si time_since_last_cg seule suffit à expliquer les résultats XGBoost.
Compare 3 approches :
  1. Baseline fixe : lever à 30 min après dernier éclair
  2. Seuil simple  : lever dès que time_since_last_cg > T
  3. XGBoost       : lever quand P(éclair) < seuil

Si l'approche 2 donne les mêmes résultats que le XGBoost,
le modèle n'apporte rien de plus qu'un simple timer.
"""

import pandas as pd
import numpy as np
import joblib

from features_snapshot import FEATURE_COLS
from model_xgb import split_by_alert
from optimize_threshold_xgb import BASELINE_MIN

SEUILS_XGB = {
    "Ajaccio": 0.18,
    "Bastia": 0.10,
    "Biarritz": 0.10,
    "Nantes": 0.10,
    "Pise": 0.11,
}


def evaluate_simple_threshold(test: pd.DataFrame, t_seuil: float):
    """Lève l'alerte dès que time_since_last_cg > t_seuil minutes."""
    gains, faux = [], []

    for (airport, alert_id), grp in test.groupby(["airport", "airport_alert_id"]):
        silence = grp[grp["time_since_last_cg"] > 0].sort_values("time_since_last_cg")

        t_lever = BASELINE_MIN  # fallback
        is_faux = 0
        for _, row in silence.iterrows():
            if row["time_since_last_cg"] > t_seuil:
                t_lever = float(row["time_since_last_cg"])
                is_faux = int(row["y"])
                break

        gains.append(BASELINE_MIN - t_lever)
        faux.append(is_faux)

    return float(np.mean(gains)), float(np.mean(faux)) * 100


if __name__ == "__main__":
    print("Chargement...")
    df = pd.read_parquet("data/snapshots.parquet")
    bundle = joblib.load("models/xgb_model.pkl")
    model = bundle["model"]

    _, test = split_by_alert(df)
    probas_all = model.predict_proba(test[FEATURE_COLS])[:, 1]

    print("\n" + "═" * 60)
    print("  COMPARAISON : timer simple vs XGBoost")
    print("═" * 60)
    print(f"{'Approche':<35s} {'Gain moyen':>10s}  {'Faux all-clear':>14s}")
    print("─" * 60)

    # Baseline fixe
    print(f"{'Baseline 30 min (fixe)':<35s} {'0.0':>10s}  {'?':>14s}")

    # Seuils simples sur time_since_last_cg
    for t in [2, 4, 6, 8, 10, 15, 20]:
        g, f = evaluate_simple_threshold(test, t_seuil=t)
        print(f"  Timer > {t:2d} min{'':<20s} {g:>+10.1f}  {f:>13.0f}%")

    # XGBoost avec seuils optimaux
    from optimize_threshold_xgb import predict_lever

    gains_xgb, faux_xgb = [], []
    for (airport, alert_id), grp in test.groupby(["airport", "airport_alert_id"]):
        seuil = SEUILS_XGB.get(airport, 0.50)
        grp_sorted = grp.sort_values("time_since_last_cg").reset_index(drop=True)
        idx = grp.sort_values("time_since_last_cg").index
        p = probas_all[test.index.get_indexer(idx)]
        t_lever, is_faux = predict_lever(grp_sorted, p, seuil)
        gains_xgb.append(BASELINE_MIN - t_lever)
        faux_xgb.append(is_faux)

    print(
        f"{'XGBoost (seuils optimaux)':<35s} {np.mean(gains_xgb):>+10.1f}  {100 * np.mean(faux_xgb):>13.0f}%"
    )
    print("─" * 60)

    print("\nConclusion :")
    best_t, best_g, best_f = None, -np.inf, 100
    for t in [2, 4, 6, 8, 10, 15, 20]:
        g, f = evaluate_simple_threshold(test, t_seuil=t)
        if g - 100 * f / 100 > best_g - 100 * best_f / 100:
            best_t, best_g, best_f = t, g, f

    xgb_score = np.mean(gains_xgb) - 100 * np.mean(faux_xgb)
    timer_score = best_g - 100 * best_f / 100
    delta = xgb_score - timer_score

    print(
        f"  Meilleur timer simple : > {best_t} min → gain {best_g:+.1f} min, faux {best_f:.0f}%"
    )
    print(
        f"  XGBoost               :          → gain {np.mean(gains_xgb):+.1f} min, faux {100 * np.mean(faux_xgb):.0f}%"
    )
    print(f"  Score XGBoost - Timer : {delta:+.1f} min  ", end="")
    if delta > 2:
        print("→ XGBoost apporte une valeur ajoutée réelle ✓")
    elif delta > 0:
        print("→ XGBoost légèrement meilleur, valeur ajoutée marginale")
    else:
        print("→ Timer simple suffit, XGBoost n'apporte rien de plus ✗")
