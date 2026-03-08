"""
evaluate.py
-----------
Évaluation complète du modèle sur tout le jeu de test.

Pour chaque alerte : calcule le temps de levée prédit, le compare à la
baseline 30 min, mesure le gain et détecte les faux all-clear.

Usage :
    python src/evaluate.py data/features.parquet
"""

import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from predict import Predictor
from features import ACTIVE_FEATURES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEUIL = 0.80
TEST_SIZE = 0.20
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------


def evaluate(features_path: str, model_path: str = "models/rsf_model.pkl"):

    df = pd.read_parquet(features_path)

    # Même split que model.py — train_test_split stratifié par aéroport
    from sklearn.model_selection import train_test_split

    _, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df["airport"]
    )
    airport_counts = df_test["airport"].value_counts().to_dict()
    print(f"Jeu de test : {len(df_test)} alertes")
    for a, n in sorted(airport_counts.items()):
        print(f"  {a:<12} : {n}")

    predictor = Predictor(model_path, seuil=SEUIL)

    # ── Calcul des métriques par alerte ─────────────────────────────────────
    records = []
    for _, row in df_test.iterrows():
        fd = {col: row[col] for col in ACTIVE_FEATURES}
        duration = row["duration"]
        airport = row["airport"]

        r = predictor.predict(fd, time_since_last_cg=0)

        records.append(
            {
                "airport": airport,
                "duration": duration,
                "t_lever_abs": r["t_lever_abs"],
                "gain": r["gain_vs_baseline"],  # 30 - t_lever
                "faux_allclear": r["t_lever_abs"] < duration,
            }
        )

    results = pd.DataFrame(records)

    # ---------------------------------------------------------------------------
    # Stats globales
    # ---------------------------------------------------------------------------
    print("\n" + "═" * 60)
    print("  RÉSULTATS GLOBAUX")
    print("═" * 60)
    print(f"  Alertes évaluées               : {len(results)}")
    print(f"  Gain moyen                     : {results['gain'].mean():+.1f} min")
    print(f"  Gain médian                    : {results['gain'].median():+.1f} min")
    n_gain = (results["gain"] > 0).sum()
    print(
        f"  Modèle bat baseline (gain > 0) : {n_gain}/{len(results)}  ({n_gain / len(results):.0%})"
    )
    n_faux = results["faux_allclear"].sum()
    print(
        f"  Faux all-clear                 : {n_faux}/{len(results)}  ({n_faux / len(results):.0%})"
    )
    print(f"  Durée réelle moyenne           : {results['duration'].mean():.1f} min")
    print(f"  t_lever moyen (modèle)         : {results['t_lever_abs'].mean():.1f} min")
    print(f"  t_lever baseline               : 30.0 min (fixe)")

    # Stats par aéroport
    print("\n" + "─" * 60)
    print("  PAR AÉROPORT")
    print("─" * 60)
    by_airport = (
        results.groupby("airport")
        .agg(
            n=("gain", "count"),
            gain_mean=("gain", "mean"),
            gain_med=("gain", "median"),
            pct_gain=("gain", lambda x: f"{(x > 0).mean():.0%}"),
            pct_faux=("faux_allclear", lambda x: f"{x.mean():.0%}"),
        )
        .round(1)
    )
    print(by_airport.to_string())

    # ---------------------------------------------------------------------------
    # Graphiques
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. Distribution des gains ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    gains = results["gain"]
    ax1.hist(
        gains[gains > 0],
        bins=20,
        color="steelblue",
        alpha=0.7,
        label=f"Gain > 0  ({(gains > 0).sum()})",
    )
    ax1.hist(
        gains[gains <= 0],
        bins=10,
        color="salmon",
        alpha=0.7,
        label=f"Gain ≤ 0  ({(gains <= 0).sum()})",
    )
    ax1.axvline(
        x=gains.mean(),
        color="black",
        linewidth=1.5,
        linestyle="--",
        label=f"Moyenne {gains.mean():+.1f} min",
    )
    ax1.axvline(
        x=gains.median(),
        color="navy",
        linewidth=1.2,
        linestyle=":",
        label=f"Médiane {gains.median():+.1f} min",
    )
    ax1.axvline(x=0, color="red", linewidth=1, alpha=0.5)
    ax1.set_title("Distribution des gains vs baseline (30 min)")
    ax1.set_xlabel("Gain (minutes)  —  positif = modèle lève avant baseline")
    ax1.set_ylabel("Nombre d'alertes")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ── 2. t_lever_abs vs durée réelle ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ok = results[~results["faux_allclear"]]
    faux = results[results["faux_allclear"]]
    ax2.scatter(
        ok["duration"],
        ok["t_lever_abs"],
        c="steelblue",
        alpha=0.5,
        s=18,
        label=f"OK ({len(ok)})",
    )
    ax2.scatter(
        faux["duration"],
        faux["t_lever_abs"],
        c="salmon",
        alpha=0.5,
        s=18,
        label=f"Faux all-clear ({len(faux)})",
    )
    lim = max(results["duration"].max(), results["t_lever_abs"].max()) + 5
    ax2.plot(
        [0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.4, label="Levée = fin réelle"
    )
    ax2.axhline(
        y=30,
        color="red",
        linewidth=1,
        linestyle="--",
        alpha=0.6,
        label="Baseline 30 min",
    )
    ax2.set_title("Levée prédite vs durée réelle de l'alerte")
    ax2.set_xlabel("Durée réelle (min)")
    ax2.set_ylabel("t_lever prédit par le modèle (min)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── 3. Boxplot gain par aéroport ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    airports = sorted(results["airport"].unique())
    data_box = [results.loc[results["airport"] == a, "gain"].values for a in airports]
    bp = ax3.boxplot(data_box, tick_labels=airports, patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax3.axhline(
        y=0, color="red", linewidth=1, linestyle="--", alpha=0.7, label="Baseline = 0"
    )
    ax3.set_title("Distribution du gain par aéroport")
    ax3.set_xlabel("Aéroport")
    ax3.set_ylabel("Gain (minutes)")
    ax3.tick_params(axis="x", rotation=15)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3, axis="y")

    # ── 4. Trade-off gain vs faux all-clear selon le seuil ───────────────────
    # Optimisation : on calcule les courbes de survie UNE SEULE FOIS
    # puis on fait varier le seuil sans retoucher au modèle
    ax4 = fig.add_subplot(gs[1, 1])
    print("Calcul du trade-off (une passe sur les données)...")

    TIMES_ABS = np.arange(0, 91, 1, dtype=float)  # horizon 90 min

    # Charger le modèle brut pour prédire en batch
    checkpoint = joblib.load(model_path)
    raw_model = checkpoint["model"]
    raw_scaler = checkpoint["scaler"]
    model_type = type(raw_model).__name__

    # Construire la matrice X_test
    X_test = df_test[ACTIVE_FEATURES].values
    if raw_scaler is not None:
        X_test = raw_scaler.transform(X_test)

    # Prédire toutes les courbes de survie en batch
    print(f"  Prédiction batch sur {len(df_test)} alertes...")
    if model_type == "RandomSurvivalForest":
        sf_funcs = raw_model.predict_survival_function(X_test)
        # Interpoler chaque courbe sur TIMES_ABS
        S_all = np.array(
            [np.interp(TIMES_ABS, sf.x, sf(sf.x), right=0.0) for sf in sf_funcs]
        )  # shape (n_alertes, len(TIMES_ABS))
    else:
        # Cox — predict_survival_function retourne un DataFrame
        X_df = df_test[ACTIVE_FEATURES]
        sf_df = raw_model.predict_survival_function(X_df, times=TIMES_ABS)
        S_all = sf_df.values.T  # shape (n_alertes, len(TIMES_ABS))

    durations = df_test["duration"].values

    # Pour chaque seuil, calculer gain et faux all-clear sans retoucher au modèle
    seuils_range = np.arange(0.50, 0.98, 0.02)
    gains_list, faux_list = [], []

    for s in seuils_range:
        # t_lever_abs = premier t où 1 - S(t) >= s
        prob_end_all = 1.0 - S_all  # shape (n_alertes, len(TIMES_ABS))
        # Pour chaque alerte : premier index où prob_end >= s
        idx_levers = np.argmax(prob_end_all >= s, axis=1)
        # argmax retourne 0 si jamais atteint → on corrige
        never_reached = prob_end_all[:, -1] < s
        idx_levers[never_reached] = len(TIMES_ABS) - 1
        t_levers = TIMES_ABS[idx_levers]

        gains_s = 30.0 - t_levers
        faux_s = t_levers < durations
        gains_list.append(gains_s.mean())
        faux_list.append(faux_s.mean())

    print("  Trade-off calculé.")

    ax4.plot(faux_list, gains_list, "o-", color="steelblue", markersize=5)

    # Annoter seuils clés
    for i, s in enumerate(seuils_range):
        if any(abs(s - ref) < 0.02 for ref in [0.60, 0.70, 0.80, 0.90]):
            ax4.annotate(
                f"{s:.0%}",
                xy=(faux_list[i], gains_list[i]),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=8,
            )

    # Point seuil actuel
    idx_cur = np.argmin(np.abs(seuils_range - SEUIL))
    ax4.scatter(
        [faux_list[idx_cur]],
        [gains_list[idx_cur]],
        color="red",
        s=80,
        zorder=5,
        label=f"Seuil actuel ({SEUIL:.0%})",
    )

    ax4.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax4.set_title(
        "Trade-off : Gain moyen vs Taux faux all-clear\n(chaque point = un seuil différent)"
    )
    ax4.set_xlabel("Taux de faux all-clear")
    ax4.set_ylabel("Gain moyen vs baseline (min)")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    fig.suptitle(
        f"Évaluation RSF — {len(df_test)} alertes de test  |  "
        f"Gain moyen {results['gain'].mean():+.1f} min  |  "
        f"Faux all-clear {results['faux_allclear'].mean():.0%}",
        fontsize=12,
        fontweight="bold",
    )

    save_path = "outputs/evaluation.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nGraphique sauvegardé : {save_path}")
    plt.close("all")

    return results


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    features_path = sys.argv[1] if len(sys.argv) > 1 else "data/features.parquet"
    evaluate(features_path)
