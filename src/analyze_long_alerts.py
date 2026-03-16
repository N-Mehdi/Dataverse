"""
analyze_long_alerts.py
----------------------
Pour les orages longs (>= LONG_ALERT_MIN), calcule le taux de faux all-clear
si on réduit la baseline de 30 min à X min après le dernier éclair CG.

Question : "Si on lève l'alerte à X min au lieu de 30 min après le dernier éclair,
            combien d'orages longs reprennent quand même ?"

Un faux all-clear ici = il y a eu un éclair CG dans les (30 - X) minutes suivantes.

Usage :
    python src/analyze_long_alerts.py
    python src/analyze_long_alerts.py data/snapshots.parquet
"""

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

from model_xgb import split_by_alert
from optimize_threshold_xgb import LONG_ALERT_MIN, BASELINE_MIN

THRESHOLDS = list(range(1, 30))  # on teste de 1 à 29 min après dernier éclair


def compute_faux_at_threshold(test: pd.DataFrame, t_lever: int) -> dict:
    """
    Pour chaque alerte longue, détermine si lever l'alerte à t_lever min
    après le dernier éclair CG est un faux all-clear.

    Faux all-clear = il existe un snapshot avec :
      - time_since_last_cg entre t_lever et BASELINE_MIN
      - y = 1 (un éclair CG survient dans les 30 min suivantes)

    Autrement dit : l'orage a repris dans la fenêtre [t_lever, 30 min].
    """
    rows = []
    for (airport, alert_id), grp in test.groupby(["airport", "airport_alert_id"]):
        elapsed_max = float(grp["elapsed_time"].max())
        is_long = elapsed_max >= LONG_ALERT_MIN
        if not is_long:
            continue

        # Snapshots de silence dans la fenêtre [t_lever, BASELINE_MIN]
        silence = grp[
            (grp["time_since_last_cg"] >= t_lever)
            & (grp["time_since_last_cg"] <= BASELINE_MIN)
        ]

        # Faux all-clear si au moins un de ces snapshots a y=1
        is_faux = int(silence["y"].max() == 1) if len(silence) > 0 else 0

        rows.append(
            {
                "airport": airport,
                "alert_id": alert_id,
                "is_faux": is_faux,
                "n_cg": int(grp["n_cg_total"].max()),
                "elapsed_max": elapsed_max,
            }
        )

    df = pd.DataFrame(rows)
    n = len(df)
    n_faux = df["is_faux"].sum()
    return {
        "t_lever": t_lever,
        "n_alertes": n,
        "n_faux": int(n_faux),
        "pct_faux": 100 * n_faux / n if n > 0 else 0.0,
        "gain": BASELINE_MIN - t_lever,
        "df": df,
    }


if __name__ == "__main__":
    snap_path = sys.argv[1] if len(sys.argv) > 1 else "data/snapshots.parquet"

    print(f"Chargement snapshots : {snap_path}")
    df = pd.read_parquet(snap_path)

    # On utilise TOUT le dataset (train + test) pour avoir plus de statistiques
    # sur les orages longs — qui sont rares
    print(f"Dataset complet : {df['airport_alert_id'].nunique()} alertes")

    long_alerts = df[df["elapsed_time"] >= LONG_ALERT_MIN]
    n_long = long_alerts["airport_alert_id"].nunique()
    print(f"Orages longs (>= {LONG_ALERT_MIN} min) : {n_long} alertes\n")

    _, test = split_by_alert(df)

    # ── Calcul pour chaque seuil ──────────────────────────────────────────────
    print("─" * 65)
    print(f"  {'t_lever':>8s} | {'gain':>6s} | {'faux':>10s} | {'% faux':>8s}")
    print("─" * 65)

    results = []
    for t in THRESHOLDS:
        r = compute_faux_at_threshold(test, t)
        results.append(r)
        print(
            f"  {t:>6d} min | {r['gain']:>+5d} min | {r['n_faux']:>4d}/{r['n_alertes']:<4d}  | {r['pct_faux']:>6.1f}%"
        )

    print("─" * 65)

    # ── Trouver le t_lever max avec 0% de faux all-clear ─────────────────────
    zero_faux = [r for r in results if r["pct_faux"] == 0.0]
    if zero_faux:
        best = max(zero_faux, key=lambda r: r["gain"])
        print(f"\n  Meilleur t_lever avec 0% faux all-clear : {best['t_lever']} min")
        print(
            f"  → Gain possible : +{best['gain']} min par rapport à la baseline de {BASELINE_MIN} min"
        )
    else:
        print("\n  Aucun seuil ne donne 0% de faux all-clear sur ce jeu de test.")
        min_faux = min(results, key=lambda r: r["pct_faux"])
        print(
            f"  Minimum observé : {min_faux['pct_faux']:.1f}% à t={min_faux['t_lever']} min"
        )

    # ── Breakdown par aéroport au meilleur seuil ─────────────────────────────
    if zero_faux:
        best_r = compute_faux_at_threshold(test, best["t_lever"])
        print(f"\n  Détail par aéroport à t_lever = {best['t_lever']} min :")
        print("  " + "─" * 45)
        for airport, grp in best_r["df"].groupby("airport"):
            n = len(grp)
            nf = grp["is_faux"].sum()
            print(f"  {airport:<12s} : {nf}/{n} faux ({100 * nf / n:.0f}%)")

    # ── Graphique ─────────────────────────────────────────────────────────────
    t_vals = [r["t_lever"] for r in results]
    faux_vals = [r["pct_faux"] for r in results]
    gain_vals = [r["gain"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(t_vals, faux_vals, "o-", color="tomato", label="Faux all-clear (%)")
    ax2.plot(t_vals, gain_vals, "s--", color="steelblue", label="Gain (min)")

    ax1.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    if zero_faux:
        ax1.axvline(
            best["t_lever"],
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"Meilleur t_lever 0% faux = {best['t_lever']} min",
        )

    ax1.set_xlabel("t_lever (min après dernier éclair CG)")
    ax1.set_ylabel("Faux all-clear (%)", color="tomato")
    ax2.set_ylabel("Gain vs baseline 30 min (min)", color="steelblue")
    ax1.set_title(
        f"Orages longs (>= {LONG_ALERT_MIN} min) — faux all-clear vs gain selon t_lever"
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig("outputs/analyze_long_alerts.png", dpi=150)
    print("\nGraphique sauvegardé : outputs/analyze_long_alerts.png")
