"""
analyze_false_allclear.py
--------------------------
Analyse détaillée des faux all-clear du modèle RSF.

Un faux all-clear = le modèle recommande de lever l'alerte
AVANT que l'orage soit réellement terminé (t_lever_abs < duration réelle).

Questions auxquelles on répond :
  1. Distribution par aéroport
  2. Distribution des durées réelles (courtes vs longues alertes)
  3. Marge d'erreur (de combien le modèle lève trop tôt)
  4. Profil des features (faux all-clear vs vrais positifs)
  5. Heure de la journée / mois (patterns temporels)

Usage :
    python analyze_false_allclear.py data/features.parquet
"""

import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append("src")
from predict import Predictor
from features import ACTIVE_FEATURES

SEUIL = 0.80
TEST_SIZE = 0.20
RANDOM_SEED = 42
MODEL_PATH = "models/rsf_model.pkl"


def run_analysis(features_path: str):

    # ── Chargement & split identique à evaluate.py ──────────────────────────
    df = pd.read_parquet(features_path)
    _, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df["airport"]
    )
    print(f"Jeu de test : {len(df_test)} alertes")

    predictor = Predictor(MODEL_PATH, seuil=SEUIL)

    # ── Calcul des prédictions ───────────────────────────────────────────────
    records = []
    for _, row in df_test.iterrows():
        fd = {col: row[col] for col in ACTIVE_FEATURES}
        r = predictor.predict(fd, time_since_last_cg=0)
        records.append(
            {
                "airport": row["airport"],
                "duration": row["duration"],
                "t_lever_abs": r["t_lever_abs"],
                "gain": r["gain_vs_baseline"],
                "faux_allclear": r["t_lever_abs"] < row["duration"],
                "early_by": row["duration"] - r["t_lever_abs"],  # >0 = trop tôt
                **{col: row[col] for col in ACTIVE_FEATURES},
            }
        )

    results = pd.DataFrame(records)
    faux = results[results["faux_allclear"]].copy()
    ok = results[~results["faux_allclear"]].copy()

    print(
        f"\nFaux all-clear : {len(faux)} / {len(results)} ({len(faux) / len(results):.0%})"
    )

    # ═══════════════════════════════════════════════════════════════════════
    # STATS TEXTE
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "═" * 60)
    print("  1. RÉPARTITION PAR AÉROPORT")
    print("═" * 60)
    by_airport = results.groupby("airport").agg(
        total=("faux_allclear", "count"),
        n_faux=("faux_allclear", "sum"),
    )
    by_airport["taux"] = by_airport["n_faux"] / by_airport["total"]
    by_airport["early_by_mean"] = faux.groupby("airport")["early_by"].mean().round(1)
    print(by_airport.to_string())

    print("\n" + "═" * 60)
    print("  2. DURÉE RÉELLE DES ALERTES CONCERNÉES")
    print("═" * 60)
    print(f"  Faux all-clear — durée réelle (min) :")
    print(faux["duration"].describe().round(1).to_string())
    print(f"\n  OK — durée réelle (min) :")
    print(ok["duration"].describe().round(1).to_string())

    print("\n" + "═" * 60)
    print("  3. MARGE D'ERREUR (de combien le modèle lève trop tôt)")
    print("═" * 60)
    print(f"  Moyenne  : {faux['early_by'].mean():.1f} min trop tôt")
    print(f"  Médiane  : {faux['early_by'].median():.1f} min trop tôt")
    print(f"  Max      : {faux['early_by'].max():.1f} min trop tôt")
    print(
        f"  < 5 min  : {(faux['early_by'] < 5).sum()} cas ({(faux['early_by'] < 5).mean():.0%})"
    )
    print(
        f"  < 10 min : {(faux['early_by'] < 10).sum()} cas ({(faux['early_by'] < 10).mean():.0%})"
    )

    print("\n" + "═" * 60)
    print("  4. PROFIL DES FEATURES (faux all-clear vs OK)")
    print("═" * 60)
    comparison = pd.DataFrame(
        {
            "faux_allclear": faux[ACTIVE_FEATURES].mean(),
            "ok": ok[ACTIVE_FEATURES].mean(),
        }
    ).round(3)
    comparison["diff_%"] = (
        (comparison["faux_allclear"] - comparison["ok"]) / comparison["ok"].abs() * 100
    ).round(1)
    print(comparison.to_string())

    # ═══════════════════════════════════════════════════════════════════════
    # GRAPHIQUES
    # ═══════════════════════════════════════════════════════════════════════

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

    # ── 1. Taux de faux all-clear par aéroport ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    airports = by_airport.index.tolist()
    taux = by_airport["taux"].values
    colors = [
        "salmon" if t > results["faux_allclear"].mean() else "steelblue" for t in taux
    ]
    bars = ax1.bar(airports, taux * 100, color=colors, alpha=0.8)
    ax1.axhline(
        y=results["faux_allclear"].mean() * 100,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Moyenne {results['faux_allclear'].mean():.0%}",
    )
    ax1.set_title("Taux de faux all-clear\npar aéroport")
    ax1.set_ylabel("Taux (%)")
    ax1.tick_params(axis="x", rotation=15)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, axis="y")
    for bar, t in zip(bars, taux):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{t:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # ── 2. Distribution durée réelle : faux vs ok ───────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, min(results["duration"].max(), 120), 25)
    ax2.hist(
        ok["duration"], bins=bins, color="steelblue", alpha=0.6, label=f"OK ({len(ok)})"
    )
    ax2.hist(
        faux["duration"],
        bins=bins,
        color="salmon",
        alpha=0.7,
        label=f"Faux all-clear ({len(faux)})",
    )
    ax2.set_title("Durée réelle des alertes\n(faux all-clear vs OK)")
    ax2.set_xlabel("Durée réelle (min)")
    ax2.set_ylabel("Nombre d'alertes")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── 3. Marge d'erreur (combien trop tôt) ───────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(faux["early_by"], bins=20, color="salmon", alpha=0.8, edgecolor="white")
    ax3.axvline(
        x=faux["early_by"].median(),
        color="red",
        linestyle="--",
        label=f"Médiane {faux['early_by'].median():.1f} min",
    )
    ax3.axvline(
        x=faux["early_by"].mean(),
        color="darkred",
        linestyle="-",
        label=f"Moyenne {faux['early_by'].mean():.1f} min",
    )
    ax3.set_title(
        "Marge d'erreur des faux all-clear\n(de combien le modèle lève trop tôt)"
    )
    ax3.set_xlabel("Minutes trop tôt")
    ax3.set_ylabel("Nombre de cas")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # ── 4-9. Top 6 features les plus discriminantes ─────────────────────────
    # Calcul du ratio |diff| / std pour trouver les features les plus discriminantes
    diff_norm = {}
    for col in ACTIVE_FEATURES:
        std = results[col].std()
        if std > 0:
            diff_norm[col] = abs(faux[col].mean() - ok[col].mean()) / std
    top_features = sorted(diff_norm, key=diff_norm.get, reverse=True)[:6]

    for i, feat in enumerate(top_features):
        row_idx = 1 + i // 3
        col_idx = i % 3
        ax = fig.add_subplot(gs[row_idx, col_idx])
        bins_f = np.linspace(
            min(results[feat].quantile(0.01), results[feat].quantile(0.01)),
            max(results[feat].quantile(0.99), results[feat].quantile(0.99)),
            20,
        )
        ax.hist(
            ok[feat],
            bins=bins_f,
            color="steelblue",
            alpha=0.6,
            label="OK",
            density=True,
        )
        ax.hist(
            faux[feat],
            bins=bins_f,
            color="salmon",
            alpha=0.7,
            label="Faux all-clear",
            density=True,
        )
        ax.set_title(f"{feat}\n(diff normalisée : {diff_norm[feat]:.2f})", fontsize=9)
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel("Densité", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Analyse des faux all-clear — {len(faux)}/{len(results)} alertes ({len(faux) / len(results):.0%})\n"
        f"Seuil : {SEUIL:.0%}  |  Marge d'erreur médiane : {faux['early_by'].median():.1f} min trop tôt",
        fontsize=12,
        fontweight="bold",
    )

    save_path = "outputs/false_allclear_analysis.png"
    Path(save_path).parent.mkdir(exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nGraphique sauvegardé : {save_path}")
    plt.close("all")

    return results, faux, ok


if __name__ == "__main__":
    features_path = sys.argv[1] if len(sys.argv) > 1 else "data/features.parquet"
    results, faux, ok = run_analysis(features_path)
