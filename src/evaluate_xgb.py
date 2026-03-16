"""
evaluate_xgb.py
---------------
Évalue le modèle XGBoost avec les seuils optimaux par aéroport.

Compare 3 stratégies pour les orages longs (>= LONG_ALERT_MIN) :
  - Baseline forcée : on ne lève jamais avant 30 min (0% faux all-clear, 0 gain)
  - Option A        : seuil bas non nul (ex: 5%) → quelques faux, gain possible
  - Option B        : lever seulement si silence >= SILENCE_MIN_LONG min
                      ET proba < seuil normal
"""

import pandas as pd
import numpy as np
import joblib
import sys
import matplotlib.pyplot as plt

from features_snapshot import FEATURE_COLS
from model_xgb import split_by_alert
from optimize_threshold_xgb import evaluate_threshold, BASELINE_MIN, LONG_ALERT_MIN

# ── Mettre à jour après optimize_threshold_xgb.py ────────────────────────────
SEUILS_PAR_AIRPORT = {
    "Ajaccio": 0.18,
    "Bastia": 0.10,
    "Biarritz": 0.10,
    "Nantes": 0.10,
    "Pise": 0.11,
}
SEUIL_DEFAUT = 0.12

# ── Paramètres des stratégies orages longs ────────────────────────────────────
SEUIL_LONG_A = 0.05  # Option A : seuil bas non nul
SILENCE_MIN_LONG = 20  # Option B : silence minimum (min) avant de lever


def predict_lever_strategie(
    alert_snapshots: pd.DataFrame,
    probas: np.ndarray,
    seuil: float,
    strategie: str,
) -> tuple:
    """
    Retourne (t_lever, is_faux) selon la stratégie choisie pour les orages longs.

    Stratégies pour les orages longs (elapsed_time >= LONG_ALERT_MIN) :
      - "baseline" : jamais de levée anticipée → retourne (BASELINE_MIN, 0)
      - "A"        : seuil abaissé à SEUIL_LONG_A
      - "B"        : lever seulement si time_since_last_cg >= SILENCE_MIN_LONG
                     ET proba < seuil normal

    Pour les orages courts, les 3 stratégies sont identiques (seuil normal).
    """
    mask_silence = alert_snapshots["time_since_last_cg"] > 0
    silence = alert_snapshots[mask_silence].copy().reset_index(drop=True)
    silence_probas = probas[mask_silence]

    if len(silence) == 0:
        return BASELINE_MIN, 0

    is_long = float(silence["elapsed_time"].max()) >= LONG_ALERT_MIN

    for i in range(len(silence)):
        p = silence_probas[i]
        t_silence = float(silence.iloc[i]["time_since_last_cg"])

        if is_long:
            if strategie == "baseline":
                return BASELINE_MIN, 0
            elif strategie == "A":
                if p < SEUIL_LONG_A:
                    return t_silence, int(silence.iloc[i]["y"])
            elif strategie == "B":
                if t_silence >= SILENCE_MIN_LONG and p < seuil:
                    return t_silence, int(silence.iloc[i]["y"])
        else:
            if p < seuil:
                return t_silence, int(silence.iloc[i]["y"])

    return BASELINE_MIN, 0


def run_strategie(
    test: pd.DataFrame, probas_all: np.ndarray, strategie: str
) -> pd.DataFrame:
    rows = []
    for (airport, alert_id), grp in test.groupby(
        ["airport", "airport_alert_id"], dropna=False
    ):
        seuil = SEUILS_PAR_AIRPORT.get(airport, SEUIL_DEFAUT)
        grp_sorted = grp.sort_values("time_since_last_cg").reset_index(drop=True)
        idx = grp.sort_values("time_since_last_cg").index
        p = probas_all[test.index.get_indexer(idx)]
        t_lever, is_faux = predict_lever_strategie(grp_sorted, p, seuil, strategie)
        rows.append(
            {
                "airport": airport,
                "alert_id": alert_id,
                "t_lever": t_lever,
                "gain": BASELINE_MIN - t_lever,
                "faux": is_faux,
                "n_cg": int(grp["n_cg_total"].max()),
                "elapsed_max": float(grp["elapsed_time"].max()),
                "is_long": float(grp["elapsed_time"].max()) >= LONG_ALERT_MIN,
            }
        )
    return pd.DataFrame(rows)


def print_stats(label: str, df: pd.DataFrame):
    n = len(df)
    if n == 0:
        print(f"  {label} : aucune alerte")
        return
    n_pos = (df["gain"] > 0).sum()
    n_faux = df["faux"].sum()
    print(
        f"  {label:<40s} | gain moy {df['gain'].mean():+5.1f} min"
        f" | gain med {df['gain'].median():+5.1f} min"
        f" | bat baseline {100 * n_pos / n:4.0f}%"
        f" | faux {n_faux:3d}/{n} ({100 * n_faux / n:4.0f}%)"
        f" | t_lever moy {df['t_lever'].mean():4.1f} min"
    )


def print_segment_stats(label: str, seg: pd.DataFrame):
    n = len(seg)
    if n == 0:
        print(f"  {label:<30s} : aucune alerte")
        return
    n_pos = (seg["gain"] > 0).sum()
    n_faux = seg["faux"].sum()
    print(f"\n  ── {label} (n={n}) ──")
    print(f"    Gain moyen          : {seg['gain'].mean():+.1f} min")
    print(f"    Gain médian         : {seg['gain'].median():+.1f} min")
    print(f"    Modèle bat baseline : {n_pos}/{n}  ({100 * n_pos / n:.0f}%)")
    print(f"    Faux all-clear      : {n_faux}/{n}  ({100 * n_faux / n:.0f}%)")
    print(
        f"    t_lever moyen       : {seg['t_lever'].mean():.1f} min après dernier éclair"
    )


if __name__ == "__main__":
    snap_path = sys.argv[1] if len(sys.argv) > 1 else "data/snapshots.parquet"
    model_path = "models/xgb_model.pkl"

    print(f"Chargement snapshots : {snap_path}")
    df = pd.read_parquet(snap_path)

    print(f"Chargement modèle : {model_path}")
    bundle = joblib.load(model_path)
    model = bundle["model"]

    _, test = split_by_alert(df)
    print(
        f"Jeu de test : {len(test)} snapshots ({test.groupby(['airport', 'airport_alert_id'], dropna=False).ngroups} alertes)"
    )

    X_test = test[FEATURE_COLS]
    probas_all = model.predict_proba(X_test)[:, 1]

    # ── Distribution des alertes ──────────────────────────────────────────────
    cg_par_alerte = (
        test.groupby(["airport", "airport_alert_id"])["n_cg_total"]
        .max()
        .reset_index()
        .rename(columns={"n_cg_total": "n_cg_alerte"})
    )
    CG_SEUIL_PETIT = int(np.percentile(cg_par_alerte["n_cg_alerte"], 33))
    CG_SEUIL_GROS = int(np.percentile(cg_par_alerte["n_cg_alerte"], 66))

    print("\n" + "─" * 60)
    print("  DISTRIBUTION DES ALERTES PAR NOMBRE DE CG")
    print("─" * 60)
    print(f"  Min    : {int(cg_par_alerte['n_cg_alerte'].min())} CG")
    print(f"  P25    : {int(np.percentile(cg_par_alerte['n_cg_alerte'], 25))} CG")
    print(f"  P33    : {CG_SEUIL_PETIT} CG  <- seuil petit/moyen")
    print(f"  Median : {int(cg_par_alerte['n_cg_alerte'].median())} CG")
    print(f"  P66    : {CG_SEUIL_GROS} CG  <- seuil moyen/gros")
    print(f"  P75    : {int(np.percentile(cg_par_alerte['n_cg_alerte'], 75))} CG")
    print(f"  Max    : {int(cg_par_alerte['n_cg_alerte'].max())} CG")

    # ── Évaluation des 3 stratégies ───────────────────────────────────────────
    strats = {
        "Baseline forcée (actuel)": "baseline",
        f"Option A (seuil {SEUIL_LONG_A:.0%})": "A",
        f"Option B (silence >= {SILENCE_MIN_LONG} min)": "B",
    }
    results_par_strat = {
        label: run_strategie(test, probas_all, code) for label, code in strats.items()
    }

    # ── Comparaison globale ───────────────────────────────────────────────────
    sep = "=" * 105
    print(f"\n{sep}")
    print("  COMPARAISON DES STRATÉGIES — GLOBAL")
    print(sep)
    for label, res in results_par_strat.items():
        print_stats(label, res)

    print(f"\n{sep}")
    print(f"  COMPARAISON DES STRATÉGIES — ORAGES LONGS (>= {LONG_ALERT_MIN} min)")
    print(sep)
    for label, res in results_par_strat.items():
        print_stats(label, res[res["is_long"]])

    print(f"\n{sep}")
    print(f"  COMPARAISON DES STRATÉGIES — ORAGES COURTS (< {LONG_ALERT_MIN} min)")
    print(sep)
    for label, res in results_par_strat.items():
        print_stats(label, res[~res["is_long"]])

    # ── Résultats détaillés baseline (référence) ──────────────────────────────
    results = results_par_strat["Baseline forcée (actuel)"]
    results["categorie"] = pd.cut(
        results["n_cg"],
        bins=[0, CG_SEUIL_PETIT, CG_SEUIL_GROS, np.inf],
        labels=[
            f"Petit  (< {CG_SEUIL_PETIT} CG)",
            f"Moyen  ({CG_SEUIL_PETIT}-{CG_SEUIL_GROS} CG)",
            f"Gros   (>= {CG_SEUIL_GROS} CG)",
        ],
    )

    print("\n" + "=" * 60)
    print("  BREAKDOWN PAR INTENSITÉ — baseline (référence)")
    print(
        f"  Seuils : petit < {CG_SEUIL_PETIT} | moyen {CG_SEUIL_PETIT}-{CG_SEUIL_GROS} | gros >= {CG_SEUIL_GROS}"
    )
    print("=" * 60)
    for cat in results["categorie"].cat.categories:
        print_segment_stats(str(cat), results[results["categorie"] == cat])

    print("\n" + "-" * 60)
    print("  PAR AÉROPORT (baseline)")
    print("-" * 60)
    per_ap = results.groupby("airport").agg(
        n=("gain", "count"),
        gain_mean=("gain", lambda x: f"{x.mean():+.1f}"),
        gain_med=("gain", lambda x: f"{x.median():+.1f}"),
        pct_gain=("gain", lambda x: f"{100 * (x > 0).mean():.0f}%"),
        pct_faux=("faux", lambda x: f"{100 * x.mean():.0f}%"),
    )
    print(per_ap.to_string())

    # ── Graphiques ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    width = 0.35

    # Fig 1 : trade-off global
    ax = axes[0]
    seuils_range = np.arange(0.10, 0.96, 0.01)
    gains_curve, faux_curve = [], []
    for s in seuils_range:
        g, f = evaluate_threshold(test, probas_all, s)
        gains_curve.append(g)
        faux_curve.append(f)
    ax.plot(faux_curve, gains_curve, "o-", markersize=3, color="steelblue")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Faux all-clear (%)")
    ax.set_ylabel("Gain moyen (min)")
    ax.set_title("Trade-off global (orages courts)")

    # Fig 2 : comparaison 3 stratégies sur orages longs
    ax2 = axes[1]
    labels_plot = list(strats.keys())
    x = np.arange(len(labels_plot))
    faux_vals = [
        100 * results_par_strat[l][results_par_strat[l]["is_long"]]["faux"].mean()
        for l in labels_plot
    ]
    gain_vals = [
        results_par_strat[l][results_par_strat[l]["is_long"]]["gain"].mean()
        for l in labels_plot
    ]
    ax2b = ax2.twinx()
    ax2.bar(
        x - width / 2,
        faux_vals,
        width,
        color="tomato",
        alpha=0.8,
        label="Faux all-clear (%)",
    )
    ax2b.bar(
        x + width / 2,
        gain_vals,
        width,
        color="steelblue",
        alpha=0.8,
        label="Gain moyen (min)",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.split("(")[0].strip() for l in labels_plot], fontsize=8)
    ax2.set_ylabel("Faux all-clear (%)", color="tomato")
    ax2b.set_ylabel("Gain moyen (min)", color="steelblue")
    ax2.set_title(f"Orages longs — 3 stratégies")

    # Fig 3 : performance par intensité (baseline)
    ax3 = axes[2]
    cats = [str(c) for c in results["categorie"].cat.categories]
    faux_par_cat = [
        100 * results[results["categorie"] == c]["faux"].mean() for c in cats
    ]
    gain_par_cat = [results[results["categorie"] == c]["gain"].mean() for c in cats]
    n_par_cat = [int((results["categorie"] == c).sum()) for c in cats]
    x3 = np.arange(len(cats))
    ax3b = ax3.twinx()
    ax3.bar(x3 - width / 2, faux_par_cat, width, color="tomato", alpha=0.8)
    ax3b.bar(x3 + width / 2, gain_par_cat, width, color="steelblue", alpha=0.8)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(
        [f"{c}\n(n={n_par_cat[i]})" for i, c in enumerate(cats)], fontsize=7
    )
    ax3.set_ylabel("Faux all-clear (%)", color="tomato")
    ax3b.set_ylabel("Gain moyen (min)", color="steelblue")
    ax3.set_title("Performance par intensité (baseline)")

    plt.tight_layout()
    plt.savefig("outputs/evaluation_xgb.png", dpi=150)
    print("\nGraphique sauvegardé : outputs/evaluation_xgb.png")
