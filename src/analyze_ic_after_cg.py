"""
analyze_ic_after_cg.py
----------------------
Pour les orages longs, analyse l'activité intra-nuage (IC) après le dernier
éclair CG, pour savoir si on peut utiliser la cessation IC comme signal de
fin d'orage.

Correction clé : les IC n'ont pas forcément de airport_alert_id renseigné.
On les matche aux alertes CG par aéroport et fenêtre temporelle.

Usage :
    python src/analyze_ic_after_cg.py data/segment_alerts_all_airports_train.csv
"""

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

from optimize_threshold_xgb import LONG_ALERT_MIN, BASELINE_MIN

THRESHOLDS_ALL = list(range(1, 31))


def load_data(path: str) -> pd.DataFrame:
    print(f"Chargement : {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["icloud"] = df["icloud"].astype(str).str.strip().str.lower() == "true"
    return df


def compute_alert_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque alerte CG, calcule les stats IC en cherchant les IC
    dans TOUS les éclairs (pas seulement ceux avec airport_alert_id).
    """
    # Séparer CG avec alert_id (pour définir les alertes) et tous les IC
    cg_alerts = df[(df["icloud"] == False) & (df["airport_alert_id"].notna())].copy()
    all_ic = df[df["icloud"] == True].copy()

    print(f"  Éclairs CG avec alert_id : {len(cg_alerts)}")
    print(f"  Éclairs IC totaux        : {len(all_ic)}")

    rows = []
    for (airport, alert_id), grp in cg_alerts.groupby(["airport", "airport_alert_id"]):
        grp = grp.sort_values("date")

        if len(grp) < 2:
            continue

        t_start = grp["date"].min()
        t_last_cg = grp["date"].max()
        elapsed = (t_last_cg - t_start).total_seconds() / 60

        # Chercher les IC du même aéroport dans la fenêtre de l'alerte + 30 min
        t_window_end = t_last_cg + pd.Timedelta(minutes=BASELINE_MIN)
        ic_airport = all_ic[
            (all_ic["airport"] == airport)
            & (all_ic["date"] >= t_start)
            & (all_ic["date"] <= t_window_end)
        ]

        # IC après le dernier CG
        ic_after = ic_airport[ic_airport["date"] > t_last_cg]
        n_ic_after = len(ic_after)
        t_last_ic = ic_after["date"].max() if n_ic_after > 0 else pd.NaT

        delta_ic = (
            (t_last_ic - t_last_cg).total_seconds() / 60 if n_ic_after > 0 else 0.0
        )
        t_last_all = max(t_last_cg, t_last_ic) if n_ic_after > 0 else t_last_cg
        delta_all = (t_last_all - t_last_cg).total_seconds() / 60

        rows.append(
            {
                "airport": airport,
                "alert_id": alert_id,
                "elapsed": elapsed,
                "is_long": elapsed >= LONG_ALERT_MIN,
                "n_cg": len(grp),
                "n_ic_after": n_ic_after,
                "delta_ic": delta_ic,
                "delta_all": delta_all,
            }
        )

    return pd.DataFrame(rows)


def compute_faux_all_threshold(stats: pd.DataFrame, t_lever_all: int) -> dict:
    """
    Calcule le taux de faux all-clear si on lève l'alerte à t_lever_all min
    après le dernier ALL (CG + IC), pour les orages longs.

    Faux all-clear = t_last_all + t_lever_all < t_last_cg
    (on lèverait AVANT le dernier CG connu → forcément un faux)
    """
    long = stats[stats["is_long"]].copy()
    n = len(long)
    if n == 0:
        return {
            "t_lever_all": t_lever_all,
            "n": 0,
            "n_faux": 0,
            "pct_faux": 0.0,
            "gain_mean": 0.0,
            "gain_median": 0.0,
            "pct_gain": 0.0,
        }

    # Gain = baseline - t_lever_all - delta_all
    # (on lève à delta_all + t_lever_all min après le dernier CG)
    long["t_lever_from_cg"] = long["delta_all"] + t_lever_all
    long["gain"] = BASELINE_MIN - long["t_lever_from_cg"]
    # Faux all-clear = on lève avant la fin de la fenêtre 30 min
    # ET delta_all < 0 (impossible) ou t_lever_from_cg <= 0 (on lève avant le dernier CG)
    # Simplifié : faux si t_lever_from_cg <= 0 (on lèverait avant ou au moment du dernier CG)
    long["is_faux"] = (long["t_lever_from_cg"] <= 0).astype(int)

    n_faux = long["is_faux"].sum()
    n_gain = (long["gain"] > 0).sum()

    return {
        "t_lever_all": t_lever_all,
        "n": n,
        "n_faux": int(n_faux),
        "pct_faux": 100 * n_faux / n,
        "gain_mean": long["gain"].mean(),
        "gain_median": long["gain"].median(),
        "pct_gain": 100 * n_gain / n,
    }


if __name__ == "__main__":
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/segment_alerts_all_airports_train.csv"
    )
    df = load_data(path)

    print(f"\nÉclairs totaux  : {len(df)}")
    print(f"  dont CG (False) : {(df['icloud'] == False).sum()}")
    print(f"  dont IC (True)  : {(df['icloud'] == True).sum()}")
    print(f"Alertes uniques : {df['airport_alert_id'].nunique()}\n")

    stats = compute_alert_stats(df)
    long = stats[stats["is_long"]]
    short = stats[~stats["is_long"]]

    print(f"\nOrages longs (>= {LONG_ALERT_MIN} min) : {len(long)}")
    print(f"Orages courts                          : {len(short)}")

    # ── Stats IC après dernier CG ─────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  ACTIVITÉ IC APRÈS LE DERNIER CG — orages longs")
    print("═" * 65)

    has_ic = long[long["n_ic_after"] > 0]
    print(
        f"\n  Orages longs avec IC après dernier CG : {len(has_ic)}/{len(long)} ({100 * len(has_ic) / len(long):.0f}%)"
    )

    print(f"\n  delta_ic (min entre dernier CG et dernier IC après CG) :")
    print(f"    Mean   : {long['delta_ic'].mean():.1f} min")
    print(f"    Median : {long['delta_ic'].median():.1f} min")
    print(f"    P75    : {np.percentile(long['delta_ic'], 75):.1f} min")
    print(f"    P90    : {np.percentile(long['delta_ic'], 90):.1f} min")
    print(f"    Max    : {long['delta_ic'].max():.1f} min")

    print(f"\n  delta_all (min entre dernier CG et dernier ALL) :")
    print(f"    Mean   : {long['delta_all'].mean():.1f} min")
    print(f"    Median : {long['delta_all'].median():.1f} min")
    print(f"    P75    : {np.percentile(long['delta_all'], 75):.1f} min")
    print(f"    P90    : {np.percentile(long['delta_all'], 90):.1f} min")
    print(f"    Max    : {long['delta_all'].max():.1f} min")

    in_window = (long["delta_all"] < BASELINE_MIN).mean()
    print(
        f"\n  Dernier ALL dans la fenêtre {BASELINE_MIN} min après dernier CG : {100 * in_window:.0f}%"
    )
    print(
        f"  → Gain moyen potentiel si delta_all < {BASELINE_MIN} min : "
        f"+{(BASELINE_MIN - long['delta_all']).mean():.1f} min"
    )

    # ── Simulation nouvelle règle ─────────────────────────────────────────────
    print("\n" + "═" * 80)
    print(f"  NOUVELLE RÈGLE : lever X min après dernier ALL (CG + IC)")
    print(f"  Baseline actuelle : {BASELINE_MIN} min après dernier CG")
    print("═" * 80)
    print(
        f"  {'t_lever_all':>12s} | {'gain moy':>9s} | {'gain med':>9s} | {'bat base':>9s} | {'faux':>10s} | {'% faux':>8s}"
    )
    print("─" * 80)

    sim_results = []
    for t in THRESHOLDS_ALL:
        r = compute_faux_all_threshold(stats, t)
        sim_results.append(r)
        print(
            f"  {t:>10d} min | {r['gain_mean']:>+8.1f} min | {r['gain_median']:>+8.1f} min"
            f" | {r['pct_gain']:>8.0f}% | {r['n_faux']:>4d}/{r['n']:<4d} | {r['pct_faux']:>6.1f}%"
        )

    print("─" * 80)

    zero = [r for r in sim_results if r["pct_faux"] == 0.0]
    if zero:
        best = max(zero, key=lambda r: r["gain_mean"])
        print(
            f"\n  Meilleur t_lever_all avec 0% faux : {best['t_lever_all']} min après dernier ALL"
        )
        print(f"  → Gain moyen   : +{best['gain_mean']:.1f} min")
        print(f"  → Gain médian  : +{best['gain_median']:.1f} min")
        print(f"  → Bat baseline : {best['pct_gain']:.0f}% des alertes longues")
    else:
        print("\n  Aucun seuil ne donne 0% de faux all-clear.")

    # ── Graphiques ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(
        long["delta_all"].clip(0, 35),
        bins=35,
        color="steelblue",
        alpha=0.8,
        edgecolor="white",
    )
    ax.axvline(
        BASELINE_MIN,
        color="tomato",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline CG = {BASELINE_MIN} min",
    )
    ax.set_xlabel("Durée entre dernier CG et dernier ALL (min)")
    ax.set_ylabel("Nombre d'alertes")
    ax.set_title(f"Distribution delta_all — orages longs (n={len(long)})")
    ax.legend()

    ax2 = axes[1]
    ax2b = ax2.twinx()
    t_vals = [r["t_lever_all"] for r in sim_results]
    faux_vals = [r["pct_faux"] for r in sim_results]
    gain_vals = [r["gain_mean"] for r in sim_results]
    ax2.plot(t_vals, faux_vals, "o-", color="tomato", label="Faux all-clear (%)")
    ax2b.plot(t_vals, gain_vals, "s--", color="steelblue", label="Gain moyen (min)")
    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    if zero:
        ax2.axvline(
            best["t_lever_all"],
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"0% faux a t={best['t_lever_all']} min",
        )
    ax2.set_xlabel("t_lever_all (min apres dernier ALL)")
    ax2.set_ylabel("Faux all-clear (%)", color="tomato")
    ax2b.set_ylabel("Gain moyen (min)", color="steelblue")
    ax2.set_title("Nouvelle regle : lever apres dernier ALL")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig("outputs/analyze_ic_after_cg.png", dpi=150)
    print("\nGraphique sauvegarde : outputs/analyze_ic_after_cg.png")
