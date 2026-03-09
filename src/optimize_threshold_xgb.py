"""
optimize_threshold_xgb.py
--------------------------
Optimise le seuil de décision XGBoost par aéroport.

Logique de décision recalibrée :
  - t_lever est mesuré DEPUIS LE DERNIER ÉCLAIR CG (pas depuis le début)
  - La baseline est "30 min après le dernier éclair" (règle actuelle)
  - On cherche le premier snapshot où P(éclair dans 30 min) < seuil
  - Gain = 30 - time_since_last_cg au moment où on lève l'alerte

Faux all-clear : on lève l'alerte à t_lever mais y=1 à ce snapshot
  (= un éclair CG survient dans les 30 min suivant t_lever)
"""

import pandas as pd
import numpy as np
import joblib
import sys
import matplotlib.pyplot as plt

from features_snapshot import FEATURE_COLS, HORIZON
from model_xgb import split_by_alert

PENALITE = 100  # minutes pénalisées par 1% de faux all-clear
SEUIL_MIN = 0.10
SEUIL_MAX = 0.95
SEUIL_STEP = 0.01
BASELINE_MIN = 30  # minutes après le dernier éclair — règle fixe actuelle


def predict_lever(
    alert_snapshots: pd.DataFrame,
    probas: np.ndarray,
    seuil: float,
) -> tuple:
    """
    Retourne (time_since_last_cg_at_lever, y_at_lever) :
      - time_since_last_cg_at_lever : minutes depuis le dernier éclair quand on lève
      - y_at_lever : 1 si faux all-clear (éclair dans les 30 min suivantes), 0 sinon

    On cherche le PREMIER snapshot après le dernier éclair (time_since_last_cg > 0)
    où P(éclair) < seuil.

    Si aucun snapshot ne passe sous le seuil → on lève à la baseline (30 min).
    """
    # On ne considère que les snapshots APRÈS le dernier éclair
    silence = alert_snapshots[alert_snapshots["time_since_last_cg"] > 0].copy()
    silence_probas = probas[alert_snapshots["time_since_last_cg"] > 0]

    if len(silence) == 0:
        # Pas de snapshot de silence → baseline
        return BASELINE_MIN, 0

    silence = silence.reset_index(drop=True)

    for i in range(len(silence)):
        if silence_probas[i] < seuil:
            t = float(silence.iloc[i]["time_since_last_cg"])
            y = int(silence.iloc[i]["y"])
            return t, y

    # Seuil jamais atteint → on lève à la baseline
    return BASELINE_MIN, 0


def evaluate_threshold(test: pd.DataFrame, probas: np.ndarray, seuil: float):
    """
    Évalue le gain moyen et le taux de faux all-clear pour un seuil donné.

    Gain = BASELINE_MIN - time_since_last_cg_at_lever
      > 0 : on lève avant la baseline → bon
      < 0 : on lève après la baseline → mauvais (trop conservateur)

    Faux all-clear : y=1 au snapshot où on lève l'alerte
      = un éclair survient dans les 30 min suivant la levée
    """
    gains = []
    faux = []

    for (airport, alert_id), grp in test.groupby(["airport", "airport_alert_id"]):
        # Trier par time_since_last_cg pour avoir l'ordre chronologique de silence
        grp_sorted = grp.sort_values("time_since_last_cg").reset_index(drop=True)
        idx = grp.sort_values("time_since_last_cg").index
        p = probas[test.index.get_indexer(idx)]

        t_lever, is_faux = predict_lever(grp_sorted, p, seuil)
        gain = BASELINE_MIN - t_lever

        gains.append(gain)
        faux.append(is_faux)

    return float(np.mean(gains)), float(np.mean(faux)) * 100


def optimize_airport(airport: str, test: pd.DataFrame, probas: np.ndarray) -> tuple:
    test_ap = test[test["airport"] == airport]
    if len(test_ap) == 0:
        return 0.5, 0.0, 0.0

    seuils = np.arange(SEUIL_MIN, SEUIL_MAX + SEUIL_STEP, SEUIL_STEP)
    best_score = -np.inf
    best_seuil = 0.5
    best_gain = 0.0
    best_faux = 0.0

    for s in seuils:
        gain, faux = evaluate_threshold(test_ap, probas, s)
        score = gain - PENALITE * faux / 100
        if score > best_score:
            best_score = score
            best_seuil = s
            best_gain = gain
            best_faux = faux

    return best_seuil, best_gain, best_faux


if __name__ == "__main__":
    snap_path = sys.argv[1] if len(sys.argv) > 1 else "data/snapshots.parquet"
    model_path = "models/xgb_model.pkl"

    print(f"Chargement snapshots : {snap_path}")
    df = pd.read_parquet(snap_path)

    print(f"Chargement modèle : {model_path}")
    bundle = joblib.load(model_path)
    model = bundle["model"]

    _, test = split_by_alert(df)

    X_test = test[FEATURE_COLS]
    probas_all = model.predict_proba(X_test)[:, 1]

    airports = sorted(test["airport"].unique())

    print("\n" + "─" * 65)
    print(f"  Optimisation (pénalité faux all-clear : {PENALITE} min / 1%)")
    print("─" * 65)

    seuils_par_airport = {}

    # Référence : seuil global 50%
    gain_g50, faux_g50 = evaluate_threshold(test, probas_all, 0.50)
    print(f"\n  Référence seuil 50% → gain {gain_g50:+.1f} min | faux {faux_g50:.0f}%")

    for airport in airports:
        seuil_opt, gain_opt, faux_opt = optimize_airport(airport, test, probas_all)
        seuils_par_airport[airport] = seuil_opt

        gain_ref, faux_ref = evaluate_threshold(
            test[test["airport"] == airport], probas_all, 0.50
        )

        print(f"\n  {airport}")
        print(f"    Seuil ref   50% → gain {gain_ref:+.1f} min | faux {faux_ref:.0f}%")
        print(
            f"    Seuil optimal {seuil_opt * 100:.0f}% → gain {gain_opt:+.1f} min | faux {faux_opt:.0f}%"
        )
        print(
            f"    Δ gain : {gain_opt - gain_ref:+.1f} min | Δ faux : {faux_opt - faux_ref:+.1f}%"
        )

    # Résultats globaux avec seuils optimisés
    gains_opt, faux_opt_list = [], []
    for (airport, alert_id), grp in test.groupby(["airport", "airport_alert_id"]):
        seuil = seuils_par_airport.get(airport, 0.5)
        grp_sorted = grp.sort_values("time_since_last_cg").reset_index(drop=True)
        idx = grp.sort_values("time_since_last_cg").index
        p = probas_all[test.index.get_indexer(idx)]
        t_lever, is_faux = predict_lever(grp_sorted, p, seuil)
        gains_opt.append(BASELINE_MIN - t_lever)
        faux_opt_list.append(is_faux)

    gains_opt = np.array(gains_opt)
    faux_opt_arr = np.array(faux_opt_list)

    print("\n" + "═" * 65)
    print("  RÉSULTATS GLOBAUX — seuils optimisés vs référence 50%")
    print("═" * 65)
    print(f"{'':30s} {'Réf 50%':>10s}   {'Seuils optimisés':>16s}")
    print(
        f"  Gain moyen          {gain_g50:>+10.1f}              {gains_opt.mean():>+.1f} min"
    )
    print(
        f"  Gain médian         {'':>10s}              {np.median(gains_opt):>+.1f} min"
    )
    print(
        f"  % gain > 0          {0:>10d}%              {100 * np.mean(gains_opt > 0):>4.0f}%"
    )
    print(
        f"  Faux all-clear      {faux_g50:>10.0f}%              {100 * faux_opt_arr.mean():>4.0f}%"
    )

    print("\n  Seuils optimaux par aéroport :")
    for airport, seuil in sorted(seuils_par_airport.items()):
        print(f"    {airport:<12s} : {seuil * 100:.0f}%")

    # Graphique trade-off par aéroport
    fig, axes = plt.subplots(1, len(airports), figsize=(4 * len(airports), 4))
    if len(airports) == 1:
        axes = [axes]
    for ax, airport in zip(axes, airports):
        seuils_range = np.arange(SEUIL_MIN, SEUIL_MAX + SEUIL_STEP, SEUIL_STEP)
        gains_s, faux_s = [], []
        test_ap = test[test["airport"] == airport]
        for s in seuils_range:
            g, f = evaluate_threshold(test_ap, probas_all, s)
            gains_s.append(g)
            faux_s.append(f)
        ax.plot(seuils_range, gains_s, color="steelblue", label="Gain moyen")
        ax2 = ax.twinx()
        ax2.plot(
            seuils_range,
            faux_s,
            color="tomato",
            linestyle="--",
            label="Faux all-clear %",
        )
        ax.axvline(
            seuils_par_airport[airport], color="green", linestyle=":", linewidth=1.5
        )
        ax.set_title(airport)
        ax.set_xlabel("Seuil")
        ax.set_ylabel("Gain (min)")
        ax2.set_ylabel("Faux all-clear (%)")
    plt.tight_layout()
    plt.savefig("outputs/threshold_optimization_xgb.png", dpi=150)
    print("\nGraphique sauvegardé : outputs/threshold_optimization_xgb.png")
