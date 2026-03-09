"""
optimize_threshold.py
---------------------
Optimisation du seuil de levée d'alerte par aéroport.
Le seuil c'est la probabilité minimale que l'orage soit terminé pour que le modèle recommande de lever l'alerte.


Au lieu d'un seuil global à 0.80, on cherche pour chaque aéroport
le seuil qui maximise le gain moyen tout en maintenant le taux de
faux all-clear sous un plafond acceptable (MAX_FAUX_ALLCLEAR).

Usage :
    python optimize_threshold.py data/features.parquet
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
from features import ACTIVE_FEATURES

TEST_SIZE = 0.20
RANDOM_SEED = 42
MODEL_PATH = "models/rsf_model.pkl"
SEUIL_GLOBAL = 0.80
MAX_FAUX_ALLCLEAR = 0.15  # affiché dans les graphiques pour référence
PENALITE = 100.0  # 1% de faux all-clear coûte PENALITE minutes de gain
TIMES_ABS = np.arange(0, 91, 1, dtype=float)


def optimize_thresholds(features_path: str):

    # ── Chargement & split ───────────────────────────────────────────────────
    df = pd.read_parquet(features_path)
    _, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df["airport"]
    )
    print(f"Jeu de test : {len(df_test)} alertes")

    # ── Chargement modèle ────────────────────────────────────────────────────
    checkpoint = joblib.load(MODEL_PATH)
    raw_model = checkpoint["model"]
    raw_scaler = checkpoint["scaler"]

    X_test = df_test[ACTIVE_FEATURES].values
    if raw_scaler is not None:
        X_test = raw_scaler.transform(X_test)

    # ── Prédiction batch des courbes de survie ───────────────────────────────
    print("Prédiction batch...")
    sf_funcs = raw_model.predict_survival_function(X_test)
    S_all = np.array(
        [np.interp(TIMES_ABS, sf.x, sf(sf.x), right=0.0) for sf in sf_funcs]
    )
    durations = df_test["duration"].values
    airports = df_test["airport"].values

    # ── Optimisation par aéroport ────────────────────────────────────────────
    seuils_range = np.arange(0.50, 0.99, 0.01)
    best_seuils = {}
    results_by_airport = {}

    print(f"\n{'─' * 65}")
    print(f"  Optimisation (pénalité faux all-clear : {PENALITE:.0f} min / 1%)")
    print(f"{'─' * 65}")

    for airport in sorted(df_test["airport"].unique()):
        mask = airports == airport
        S_ap = S_all[mask]
        dur_ap = durations[mask]

        gains_list, faux_list = [], []
        for s in seuils_range:
            prob_end = 1.0 - S_ap
            idx_levers = np.argmax(prob_end >= s, axis=1)
            never = prob_end[:, -1] < s
            idx_levers[never] = len(TIMES_ABS) - 1
            t_levers = TIMES_ABS[idx_levers]
            gains_list.append((30.0 - t_levers).mean())
            faux_list.append((t_levers < dur_ap).mean())

        gains_arr = np.array(gains_list)
        faux_arr = np.array(faux_list)

        # Meilleur seuil = max(gain - PENALITE * faux_all_clear)
        # Ex : PENALITE=3 → 1% de faux all-clear coûte 3 min de gain
        scores = gains_arr - PENALITE * faux_arr * 100
        best_idx = np.argmax(scores)
        best_seuil = seuils_range[best_idx]
        best_gain = gains_arr[best_idx]
        best_faux = faux_arr[best_idx]

        best_seuils[airport] = best_seuil
        results_by_airport[airport] = {
            "seuils": seuils_range,
            "gains": gains_arr,
            "faux": faux_arr,
            "best_seuil": best_seuil,
            "best_gain": best_gain,
            "best_faux": best_faux,
        }

        # Résultats avec seuil global pour comparaison
        idx_global = np.argmin(np.abs(seuils_range - SEUIL_GLOBAL))
        gain_global = gains_arr[idx_global]
        faux_global = faux_arr[idx_global]

        print(f"\n  {airport}")
        print(
            f"    Seuil global  {SEUIL_GLOBAL:.0%} → gain {gain_global:+.1f} min | faux {faux_global:.0%}"
        )
        print(
            f"    Seuil optimal {best_seuil:.0%} → gain {best_gain:+.1f} min | faux {best_faux:.0%}"
        )
        delta_gain = best_gain - gain_global
        delta_faux = best_faux - faux_global
        print(f"    Δ gain : {delta_gain:+.1f} min | Δ faux : {delta_faux:+.1%}")

    # ── Résultats globaux avec seuils optimisés ──────────────────────────────
    print(f"\n{'═' * 65}")
    print("  RÉSULTATS GLOBAUX — seuils optimisés vs seuil global")
    print(f"{'═' * 65}")

    gains_opt, faux_opt = [], []
    gains_glob, faux_glob = [], []

    for i, (airport, dur) in enumerate(zip(airports, durations)):
        s_opt = best_seuils[airport]
        s_glob = SEUIL_GLOBAL

        for s, gains_l, faux_l in [
            (s_opt, gains_opt, faux_opt),
            (s_glob, gains_glob, faux_glob),
        ]:
            prob_end = 1.0 - S_all[i]
            idx = np.argmax(prob_end >= s)
            if prob_end[-1] < s:
                idx = len(TIMES_ABS) - 1
            t_lever = TIMES_ABS[idx]
            gains_l.append(30.0 - t_lever)
            faux_l.append(t_lever < dur)

    gains_opt = np.array(gains_opt)
    faux_opt = np.array(faux_opt)
    gains_glob = np.array(gains_glob)
    faux_glob = np.array(faux_glob)

    print(f"  {'':20} {'Seuil global':>15} {'Seuils optimisés':>18}")
    print(
        f"  {'Gain moyen':20} {gains_glob.mean():>+14.1f} {gains_opt.mean():>+17.1f} min"
    )
    print(
        f"  {'Gain médian':20} {np.median(gains_glob):>+14.1f} {np.median(gains_opt):>+17.1f} min"
    )
    print(
        f"  {'% gain > 0':20} {(gains_glob > 0).mean():>14.0%} {(gains_opt > 0).mean():>17.0%}"
    )
    print(f"  {'Faux all-clear':20} {faux_glob.mean():>14.0%} {faux_opt.mean():>17.0%}")

    # ── Seuils optimaux résumé ───────────────────────────────────────────────
    print(f"\n  Seuils optimaux par aéroport (pénalité {PENALITE:.0f} min / 1% faux) :")
    for airport, s in sorted(best_seuils.items()):
        print(f"    {airport:<12} : {s:.0%}")

    # ── Graphiques ───────────────────────────────────────────────────────────
    airports_list = sorted(df_test["airport"].unique())
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    for i, airport in enumerate(airports_list):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        r = results_by_airport[airport]

        ax.plot(
            r["faux"] * 100,
            r["gains"],
            "o-",
            color="steelblue",
            markersize=3,
            linewidth=1.5,
            label="Courbe trade-off",
        )

        # Point seuil global
        idx_glob = np.argmin(np.abs(r["seuils"] - SEUIL_GLOBAL))
        ax.scatter(
            [r["faux"][idx_glob] * 100],
            [r["gains"][idx_glob]],
            color="red",
            s=80,
            zorder=5,
            label=f"Global {SEUIL_GLOBAL:.0%} ({r['gains'][idx_glob]:+.1f} min, {r['faux'][idx_glob]:.0%})",
        )

        # Point seuil optimal
        ax.scatter(
            [r["best_faux"] * 100],
            [r["best_gain"]],
            color="green",
            s=80,
            zorder=5,
            marker="*",
            label=f"Optimal {r['best_seuil']:.0%} ({r['best_gain']:+.1f} min, {r['best_faux']:.0%})",
        )

        # Plafond faux all-clear
        ax.axvline(
            x=MAX_FAUX_ALLCLEAR * 100,
            color="orange",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"Plafond {MAX_FAUX_ALLCLEAR:.0%}",
        )

        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(airport, fontsize=11, fontweight="bold")
        ax.set_xlabel("Taux faux all-clear (%)")
        ax.set_ylabel("Gain moyen (min)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Dernier subplot : comparaison globale par aéroport
    ax_last = fig.add_subplot(gs[1, 2])
    x = np.arange(len(airports_list))
    width = 0.35
    gains_glob_by_ap = [
        results_by_airport[a]["gains"][
            np.argmin(np.abs(results_by_airport[a]["seuils"] - SEUIL_GLOBAL))
        ]
        for a in airports_list
    ]
    gains_opt_by_ap = [results_by_airport[a]["best_gain"] for a in airports_list]

    ax_last.bar(
        x - width / 2,
        gains_glob_by_ap,
        width,
        label=f"Seuil global {SEUIL_GLOBAL:.0%}",
        color="salmon",
        alpha=0.8,
    )
    ax_last.bar(
        x + width / 2,
        gains_opt_by_ap,
        width,
        label="Seuil optimal",
        color="steelblue",
        alpha=0.8,
    )
    ax_last.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
    ax_last.set_xticks(x)
    ax_last.set_xticklabels(airports_list, rotation=15)
    ax_last.set_title("Gain moyen : global vs optimal", fontsize=11, fontweight="bold")
    ax_last.set_ylabel("Gain moyen (min)")
    ax_last.legend(fontsize=8)
    ax_last.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Optimisation du seuil par aéroport (pénalité : {PENALITE:.0f} min / 1% faux all-clear)\n"
        f"Gain moyen : seuil global {gains_glob.mean():+.1f} min → seuils optimisés {gains_opt.mean():+.1f} min",
        fontsize=12,
        fontweight="bold",
    )

    save_path = "outputs/threshold_optimization.png"
    Path(save_path).parent.mkdir(exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nGraphique sauvegardé : {save_path}")
    plt.close("all")

    return best_seuils


if __name__ == "__main__":
    features_path = sys.argv[1] if len(sys.argv) > 1 else "data/features.parquet"
    best_seuils = optimize_thresholds(features_path)
