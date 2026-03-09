"""
compute_pause_thresholds.py
---------------------------
Calcule empiriquement la pause minimale de sécurité par aéroport et par saison,
au-delà de laquelle l'orage ne reprend quasi-jamais (objectif : P(reprise) < 1%).

Logique :
  Pour chaque alerte, on reconstitue la séquence d'éclairs CG et on identifie
  toutes les pauses INTERNES (entre deux éclairs consécutifs, avant le dernier).
  Une pause est "trompeuse" si l'orage a repris après elle.
  On cherche le seuil minimal X tel que P(reprise | pause >= X) < TARGET_RISK.

Facteurs pris en compte :
  - Aéroport (position géographique, type d'orage dominant)
  - Saison (été vs automne/hiver vs printemps)

Usage :
    python compute_pause_thresholds.py data/segment_alerts_all_airports_train.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Paramètres ───────────────────────────────────────────────────────────────
TARGET_RISK = 0.01  # P(reprise | pause >= X) cible : 1%
MIN_SAMPLES = 10  # nombre minimum de pauses pour estimer le seuil
PAUSE_RANGE = np.arange(1, 61, 1, dtype=float)  # seuils testés : 1 à 60 min

SAISONS = {
    1: "Hiver",
    2: "Hiver",
    3: "Printemps",
    4: "Printemps",
    5: "Printemps",
    6: "Été",
    7: "Été",
    8: "Été",
    9: "Automne",
    10: "Automne",
    11: "Automne",
    12: "Hiver",
}


def get_season(month: int) -> str:
    return SAISONS[month]


# Fenêtre de fin d'alerte : on ne considère que les pauses dans les
# WINDOW_END_MIN dernières minutes de l'alerte
WINDOW_END_MIN = 30


def extract_internal_pauses(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque alerte, extrait les pauses en FIN D'ALERTE uniquement
    (dans les WINDOW_END_MIN dernières minutes), là où on cherche à prédire
    la fin de l'orage.

    Une pause est "trompeuse" (reprise=True) si un éclair CG survient après elle.
    La dernière pause (après le dernier éclair) a reprise=False.

    Retourne un DataFrame avec :
      airport, saison, pause_min, reprise (bool)
    """
    alerts = df_raw[df_raw["airport_alert_id"].notna()].copy()
    cg = alerts[alerts["icloud"] == False].copy()
    cg["date"] = pd.to_datetime(cg["date"], utc=True)

    records = []

    for (airport, alert_id), grp in cg.groupby(["airport", "airport_alert_id"]):
        grp = grp.sort_values("date")
        dates = grp["date"].values
        if len(dates) < 3:
            continue

        saison = get_season(pd.Timestamp(dates[0]).month)
        times_min = (dates - dates[0]).astype("timedelta64[s]").astype(float) / 60
        duration = times_min[-1]

        if duration <= 0:
            continue

        # Fenêtre de fin : dernières WINDOW_END_MIN minutes de l'alerte
        t_window_start = duration - WINDOW_END_MIN

        # Pauses internes dans la fenêtre de fin (hors dernière pause)
        for i in range(len(times_min) - 2):
            t_pause_start = times_min[i]
            if t_pause_start < t_window_start:
                continue  # pause trop tôt dans l'alerte
            pause = times_min[i + 1] - times_min[i]
            records.append(
                {
                    "airport": airport,
                    "saison": saison,
                    "pause_min": pause,
                    "reprise": True,  # orage a repris après cette pause
                    "duration": duration,
                }
            )

        # Dernière pause (toujours en fin d'alerte par définition)
        last_pause = times_min[-1] - times_min[-2]
        records.append(
            {
                "airport": airport,
                "saison": saison,
                "pause_min": last_pause,
                "reprise": False,  # fin d'orage
                "duration": duration,
            }
        )

    return pd.DataFrame(records)


def compute_thresholds(pauses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque (aéroport, saison), calcule le seuil de pause minimal
    tel que P(reprise | pause >= seuil) < TARGET_RISK.
    """
    results = []

    for (airport, saison), grp in pauses_df.groupby(["airport", "saison"]):
        thresholds_found = []

        for X in PAUSE_RANGE:
            subset = grp[grp["pause_min"] >= X]
            if len(subset) < MIN_SAMPLES:
                break
            p_reprise = subset["reprise"].mean()
            thresholds_found.append(
                {
                    "X": X,
                    "n": len(subset),
                    "p_reprise": p_reprise,
                }
            )

        if not thresholds_found:
            results.append(
                {
                    "airport": airport,
                    "saison": saison,
                    "pause_threshold": np.nan,
                    "p_reprise_at_threshold": np.nan,
                    "n_pauses": len(grp),
                    "note": "données insuffisantes",
                }
            )
            continue

        df_t = pd.DataFrame(thresholds_found)

        # Seuil = plus petite valeur X telle que p_reprise < TARGET_RISK
        below = df_t[df_t["p_reprise"] < TARGET_RISK]
        if below.empty:
            # On n'atteint jamais le target → on prend le minimum disponible
            best = df_t.iloc[-1]
            note = f"cible {TARGET_RISK:.0%} non atteinte, min={best['p_reprise']:.1%}"
        else:
            best = below.iloc[0]
            note = f"p_reprise={best['p_reprise']:.1%} à X={best['X']:.0f} min"

        results.append(
            {
                "airport": airport,
                "saison": saison,
                "pause_threshold": best["X"],
                "p_reprise_at_threshold": best["p_reprise"],
                "n_pauses": len(grp),
                "note": note,
            }
        )

    return pd.DataFrame(results).sort_values(["airport", "saison"])


def plot_results(pauses_df: pd.DataFrame, thresholds_df: pd.DataFrame):
    """
    Pour chaque aéroport, trace P(reprise | pause >= X) en fonction de X,
    par saison, avec le seuil optimal marqué.
    """
    airports = sorted(pauses_df["airport"].unique())
    saisons_colors = {
        "Hiver": "steelblue",
        "Printemps": "seagreen",
        "Été": "orange",
        "Automne": "firebrick",
    }

    n_cols = 3
    n_rows = (len(airports) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten()

    for i, airport in enumerate(airports):
        ax = axes[i]
        grp_airport = pauses_df[pauses_df["airport"] == airport]

        for saison, color in saisons_colors.items():
            grp_s = grp_airport[grp_airport["saison"] == saison]
            if len(grp_s) < MIN_SAMPLES:
                continue

            p_reprises = []
            ns = []
            for X in PAUSE_RANGE:
                subset = grp_s[grp_s["pause_min"] >= X]
                if len(subset) < MIN_SAMPLES:
                    break
                p_reprises.append(subset["reprise"].mean() * 100)
                ns.append(len(subset))

            if not p_reprises:
                continue

            xs = PAUSE_RANGE[: len(p_reprises)]
            ax.plot(xs, p_reprises, color=color, linewidth=2, label=saison)

            # Marquer le seuil optimal
            row = thresholds_df[
                (thresholds_df["airport"] == airport)
                & (thresholds_df["saison"] == saison)
            ]
            if not row.empty and not np.isnan(row.iloc[0]["pause_threshold"]):
                X_opt = row.iloc[0]["pause_threshold"]
                p_opt = row.iloc[0]["p_reprise_at_threshold"] * 100
                ax.scatter([X_opt], [p_opt], color=color, s=80, zorder=5, marker="*")
                ax.axvline(x=X_opt, color=color, linestyle=":", linewidth=1, alpha=0.5)

        ax.axhline(
            y=TARGET_RISK * 100,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Cible {TARGET_RISK:.0%}",
        )
        ax.set_title(airport, fontsize=12, fontweight="bold")
        ax.set_xlabel("Pause minimale (min)")
        ax.set_ylabel("P(reprise | pause ≥ X) %")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 45)
        ax.set_ylim(0, 105)

    # Masquer les axes vides
    for j in range(len(airports), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Seuil de pause minimal par aéroport et saison\n"
        f"(objectif : P(reprise | pause ≥ X) < {TARGET_RISK:.0%})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    out = "outputs/pause_thresholds.png"
    Path(out).parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nGraphique sauvegardé : {out}")
    plt.close("all")


def print_summary(thresholds_df: pd.DataFrame):
    print(f"\n{'═' * 65}")
    print(f"  SEUILS DE PAUSE MINIMALE PAR AÉROPORT ET SAISON")
    print(f"  (objectif : P(reprise) < {TARGET_RISK:.0%})")
    print(f"{'═' * 65}")
    print(
        f"  {'Aéroport':<12} {'Saison':<12} {'Seuil (min)':>12} {'P(reprise)':>12} {'N pauses':>10}"
    )
    print(f"  {'─' * 60}")

    for _, row in thresholds_df.iterrows():
        seuil = (
            f"{row['pause_threshold']:.0f}"
            if not np.isnan(row["pause_threshold"])
            else "N/A"
        )
        p = (
            f"{row['p_reprise_at_threshold']:.1%}"
            if not np.isnan(row["p_reprise_at_threshold"])
            else "N/A"
        )
        print(
            f"  {row['airport']:<12} {row['saison']:<12} {seuil:>12} {p:>12} {int(row['n_pauses']):>10}"
        )

    print(f"\n{'─' * 65}")
    print("  SEUIL RECOMMANDÉ PAR AÉROPORT (max sur toutes saisons)")
    print(f"{'─' * 65}")

    for airport, grp in thresholds_df.groupby("airport"):
        seuil_max = grp["pause_threshold"].max()
        seuil_str = f"{seuil_max:.0f} min" if not np.isnan(seuil_max) else "N/A"
        print(
            f"  {airport:<12} : {seuil_str}  ← utiliser le max pour garantir la sécurité toutes saisons"
        )


if __name__ == "__main__":
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/segment_alerts_all_airports_train.csv"
    )

    print(f"Chargement de {path}...")
    df_raw = pd.read_csv(path)
    df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)

    print("Extraction des pauses internes...")
    pauses_df = extract_internal_pauses(df_raw)
    print(
        f"  → {len(pauses_df)} pauses extraites sur {pauses_df['airport'].nunique()} aéroports"
    )

    print("Calcul des seuils...")
    thresholds_df = compute_thresholds(pauses_df)

    print_summary(thresholds_df)
    plot_results(pauses_df, thresholds_df)

    # Sauvegarde
    out_csv = "outputs/pause_thresholds.csv"
    thresholds_df.to_csv(out_csv, index=False)
    print(f"Tableau sauvegardé : {out_csv}")
