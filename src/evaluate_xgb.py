"""
evaluate_xgb.py
---------------
Évalue le modèle XGBoost avec les seuils optimaux par aéroport.

Même logique que optimize_threshold_xgb.py :
  - t_lever mesuré depuis le dernier éclair CG
  - Gain = 30 - time_since_last_cg au moment de la levée
  - Faux all-clear = y=1 au snapshot de levée

Met à jour SEUILS_PAR_AIRPORT après optimize_threshold_xgb.py.
"""

import pandas as pd
import numpy as np
import joblib
import sys
import matplotlib.pyplot as plt

from features_snapshot import FEATURE_COLS
from model_xgb import split_by_alert
from optimize_threshold_xgb import predict_lever, evaluate_threshold, BASELINE_MIN

# ── Mettre à jour après optimize_threshold_xgb.py ────────────────────────────
SEUILS_PAR_AIRPORT = {
    "Ajaccio": 0.18,
    "Bastia": 0.10,
    "Biarritz": 0.10,
    "Nantes": 0.10,
    "Pise": 0.11,
}
SEUIL_DEFAUT = 0.12


if __name__ == "__main__":
    snap_path = sys.argv[1] if len(sys.argv) > 1 else "data/snapshots.parquet"
    model_path = "models/xgb_model.pkl"

    print(f"Chargement snapshots : {snap_path}")
    df = pd.read_parquet(snap_path)

    print(f"Chargement modèle : {model_path}")
    bundle = joblib.load(model_path)
    model = bundle["model"]

    _, test = split_by_alert(df)
    n_alertes = test["airport_alert_id"].nunique()
    print(f"Jeu de test : {len(test)} snapshots ({n_alertes} alertes)")

    X_test = test[FEATURE_COLS]
    probas_all = model.predict_proba(X_test)[:, 1]

    rows = []
    for (airport, alert_id), grp in test.groupby(["airport", "airport_alert_id"]):
        seuil = SEUILS_PAR_AIRPORT.get(airport, SEUIL_DEFAUT)
        grp_sorted = grp.sort_values("time_since_last_cg").reset_index(drop=True)
        idx = grp.sort_values("time_since_last_cg").index
        p = probas_all[test.index.get_indexer(idx)]

        t_lever, is_faux = predict_lever(grp_sorted, p, seuil)
        gain = BASELINE_MIN - t_lever

        rows.append(
            {
                "airport": airport,
                "alert_id": alert_id,
                "t_lever": t_lever,
                "gain": gain,
                "faux": is_faux,
            }
        )

    results = pd.DataFrame(rows)
    n = len(results)
    n_pos = (results["gain"] > 0).sum()
    n_faux = results["faux"].sum()

    print("\n" + "═" * 60)
    print("  RÉSULTATS GLOBAUX")
    print("═" * 60)
    print(f"  Alertes évaluées               : {n}")
    print(f"  Gain moyen                     : {results['gain'].mean():+.1f} min")
    print(f"  Gain médian                    : {results['gain'].median():+.1f} min")
    print(f"  Modèle bat baseline (gain > 0) : {n_pos}/{n}  ({100 * n_pos / n:.0f}%)")
    print(f"  Faux all-clear                 : {n_faux}/{n}  ({100 * n_faux / n:.0f}%)")
    print(
        f"  t_lever moyen (modèle)         : {results['t_lever'].mean():.1f} min après dernier éclair"
    )
    print(
        f"  Baseline                       : {BASELINE_MIN:.1f} min après dernier éclair (fixe)"
    )

    print("\n" + "─" * 60)
    print("  SEUILS UTILISÉS PAR AÉROPORT")
    print("─" * 60)
    for ap, s in sorted(SEUILS_PAR_AIRPORT.items()):
        print(f"  {ap:<12s} : {s * 100:.0f}%")

    print("\n" + "─" * 60)
    print("  PAR AÉROPORT")
    print("─" * 60)
    per_ap = results.groupby("airport").agg(
        n=("gain", "count"),
        gain_mean=("gain", lambda x: f"{x.mean():+.1f}"),
        gain_med=("gain", lambda x: f"{x.median():+.1f}"),
        pct_gain=("gain", lambda x: f"{100 * (x > 0).mean():.0f}%"),
        pct_faux=("faux", lambda x: f"{100 * x.mean():.0f}%"),
    )
    print(per_ap.to_string())

    # Graphique trade-off global
    seuils_range = np.arange(0.10, 0.96, 0.01)
    gains_curve, faux_curve = [], []
    for s in seuils_range:
        g, f = evaluate_threshold(test, probas_all, s)
        gains_curve.append(g)
        faux_curve.append(f)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(faux_curve, gains_curve, "o-", markersize=3, color="steelblue")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(
        100 * n_faux / n,
        color="green",
        linestyle=":",
        linewidth=1.2,
        label=f"Seuils optimisés ({100 * n_faux / n:.0f}% faux)",
    )
    ax.set_xlabel("Faux all-clear (%)")
    ax.set_ylabel("Gain moyen (min)")
    ax.set_title("Trade-off gain / faux all-clear — XGBoost snapshots")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/evaluation_xgb.png", dpi=150)
    print("\nGraphique sauvegardé : outputs/evaluation_xgb.png")
