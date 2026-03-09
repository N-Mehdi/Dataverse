"""
features.py
-----------
Feature engineering pour la prédiction de fin d'alerte orageuse.

Pour chaque alerte (identifiée par airport + airport_alert_id),
on calcule des features à l'instant du dernier éclair nuage-sol connu,
puis on construit la variable cible pour l'analyse de survie :
  - duration  : durée totale de l'alerte en minutes
  - event     : 1 si l'alerte est terminée (non censurée), 0 sinon

Orientations v3 :
  - Ajout features azimuth : direction et rotation de l'orage
  - Ajout features maxis : prise en compte de l'erreur de localisation
  - Mieux détecter les orages à longues pauses qui reprennent
  - Maximiser la sécurité (réduire les faux all-clear)
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Features actives (utilisées par les modèles)
# ---------------------------------------------------------------------------

ACTIVE_FEATURES = [
    # ── Comptage ────────────────────────────────────────────────────────────
    "n_cg_total",  # nombre total d'éclairs CG dans l'alerte
    "n_cg_recent",  # éclairs CG dans les 10 dernières minutes
    "n_cg_last5",  # éclairs CG dans les 5 dernières minutes
    "activity_trend",  # 2ème moitié - 1ère moitié (< 0 = décroissance)
    "cg_density",  # n_cg_total / duration
    # ── Amplitude ───────────────────────────────────────────────────────────
    "amp_max",
    "amp_mean",
    "amp_last",
    "amp_trend_global",
    "amp_trend_recent",
    "amp_decay_rate",
    # ── Distance brute ───────────────────────────────────────────────────────
    "dist_min",
    "dist_mean",
    "dist_last",
    "dist_last_vs_mean",
    "dist_recent_min",
    "dist_trend_global",
    "dist_trend_recent",
    "dist_trend_last5",
    # ── Distance ajustée par maxis (pessimiste) ──────────────────────────────
    "dist_min_adjusted",  # dist_min - maxis à cet éclair (distance pessimiste)
    "dist_last_adjusted",  # dist_last - maxis_last
    "maxis_mean",  # erreur de localisation moyenne de l'alerte
    "maxis_last",  # erreur du dernier éclair CG
    # ── Direction / azimuth ──────────────────────────────────────────────────
    "azimuth_last",  # direction du dernier éclair (0-360°)
    "azimuth_spread",  # dispersion angulaire (std circulaire) — orage localisé vs entourant
    "azimuth_trend",  # rotation angulaire moyenne entre éclairs consécutifs (deg/min)
    "azimuth_last_vs_mean",  # écart entre dernier azimuth et azimuth moyen
    # ── Temporel ────────────────────────────────────────────────────────────
    "elapsed_time",
    "inter_time_last3",
    # ── Persistance / structure temporelle ──────────────────────────────────
    "n_bursts",
    "activity_variance",
    "pause_max",
    "pause_ratio",
    "intensity_persistence",
    "pause_since_peak",
    "resume_risk",
    "long_pause_count",
    "pause_cv",
    # ── Intra-nuage ──────────────────────────────────────────────────────────
    "n_ic_recent",
    "ratio_ic_cg",
    # ── Saison / mois ────────────────────────────────────────────────────────
    "month",
    "season",
]

# ---------------------------------------------------------------------------
# Features inactives
# ---------------------------------------------------------------------------

INACTIVE_FEATURES = [
    "n_ic_total",
    "ratio_ic_recent",
    "amp_recent_mean",
    "time_since_last_cg",
]


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def get_alerts(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["airport_alert_id"].notna()].copy()


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------


def _linear_slope(times_min: np.ndarray, values: np.ndarray) -> float:
    if len(times_min) < 2:
        return 0.0
    t = times_min - times_min[0]
    t_mean = t.mean()
    v_mean = values.mean()
    num = ((t - t_mean) * (values - v_mean)).sum()
    den = ((t - t_mean) ** 2).sum()
    return float(num / den) if den > 0 else 0.0


def _circular_std(angles_deg: np.ndarray) -> float:
    """Écart-type circulaire des angles en degrés (0-360)."""
    if len(angles_deg) < 2:
        return 0.0
    rad = np.deg2rad(angles_deg)
    S = np.mean(np.sin(rad))
    C = np.mean(np.cos(rad))
    R = np.sqrt(S**2 + C**2)
    # std circulaire en degrés
    return float(np.rad2deg(np.sqrt(-2 * np.log(np.clip(R, 1e-10, 1.0)))))


def _circular_mean(angles_deg: np.ndarray) -> float:
    """Moyenne circulaire des angles en degrés."""
    if len(angles_deg) == 0:
        return 0.0
    rad = np.deg2rad(angles_deg)
    S = np.mean(np.sin(rad))
    C = np.mean(np.cos(rad))
    mean_rad = np.arctan2(S, C)
    return float(np.rad2deg(mean_rad) % 360)


def _angular_diff(a1: float, a2: float) -> float:
    """Différence angulaire signée entre deux angles (résultat dans [-180, 180])."""
    diff = (a2 - a1 + 180) % 360 - 180
    return float(diff)


# ---------------------------------------------------------------------------
# Features structure temporelle
# ---------------------------------------------------------------------------


def _compute_burst_features(
    dates: pd.Series,
    duration_min: float,
    bin_size: int = 5,
) -> dict:
    if len(dates) < 2 or duration_min <= 0:
        return {
            "n_bursts": 0,
            "activity_variance": 0.0,
            "pause_max": 0.0,
            "intensity_persistence": 0.0,
            "pause_since_peak": 0.0,
            "resume_risk": 0.0,
            "long_pause_count": 0,
            "pause_cv": 0.0,
        }

    t_start = dates.min()
    times_min = (dates - t_start).dt.total_seconds().values / 60

    n_bins = max(1, int(np.ceil(duration_min / bin_size)))
    counts, bin_edges = np.histogram(times_min, bins=n_bins, range=(0, duration_min))

    is_active = counts > 0
    n_bursts = int(np.sum((~is_active[:-1]) & is_active[1:]))
    activity_variance = float(counts.var())

    inter_times = np.diff(np.sort(times_min))
    pause_max = float(inter_times.max()) if len(inter_times) > 0 else 0.0

    mean_count = counts.mean()
    intensity_persistence = (
        float((counts > mean_count).mean()) if mean_count > 0 else 0.0
    )

    peak_bin = int(np.argmax(counts))
    peak_time = bin_edges[peak_bin]
    pause_since_peak = max(0.0, duration_min - peak_time)

    # resume_risk : pause_max * n_bursts — orages qui ont déjà repris après
    # une longue pause sont les plus dangereux
    resume_risk = pause_max * n_bursts

    # long_pause_count : nombre de pauses > 5 min entre éclairs consécutifs
    long_pause_count = int((inter_times > 5).sum()) if len(inter_times) > 0 else 0

    # pause_cv : coefficient de variation des pauses (irrégularité temporelle)
    # élevé = orage très irrégulier = imprévisible
    if len(inter_times) > 1 and inter_times.mean() > 0:
        pause_cv = float(inter_times.std() / inter_times.mean())
    else:
        pause_cv = 0.0

    return {
        "n_bursts": n_bursts,
        "activity_variance": activity_variance,
        "pause_max": pause_max,
        "intensity_persistence": intensity_persistence,
        "pause_since_peak": pause_since_peak,
        "resume_risk": resume_risk,
        "long_pause_count": long_pause_count,
        "pause_cv": pause_cv,
    }


# ---------------------------------------------------------------------------
# Features par alerte
# ---------------------------------------------------------------------------


def compute_alert_features(df: pd.DataFrame, window_minutes: int = 10) -> pd.DataFrame:
    alerts = get_alerts(df)
    cg = alerts[alerts["icloud"] == False].copy()
    all_lgt = alerts.copy()

    records = []

    for (airport, alert_id), grp_cg in cg.groupby(["airport", "airport_alert_id"]):
        grp_cg = grp_cg.sort_values("date")
        grp_all = all_lgt[
            (all_lgt["airport"] == airport) & (all_lgt["airport_alert_id"] == alert_id)
        ].sort_values("date")

        # ── Début et fin ─────────────────────────────────────────────────────
        t_start = grp_cg["date"].min()
        t_end = grp_cg["date"].max()
        duration = (t_end - t_start).total_seconds() / 60
        elapsed_time = duration
        event = 1

        times_all = (grp_cg["date"] - t_start).dt.total_seconds().values / 60

        # ── Comptage ─────────────────────────────────────────────────────────
        n_cg_total = len(grp_cg)
        t_mid = t_start + (t_end - t_start) / 2
        activity_trend = len(grp_cg[grp_cg["date"] > t_mid]) - len(
            grp_cg[grp_cg["date"] <= t_mid]
        )
        cg_density = n_cg_total / duration if duration > 0 else 0.0

        # ── Amplitude ────────────────────────────────────────────────────────
        amps = grp_cg["amplitude"].abs().values
        amp_max = float(amps.max())
        amp_mean = float(amps.mean())
        amp_trend_global = _linear_slope(times_all, amps)
        last_row = grp_cg.iloc[-1]
        amp_last = float(abs(last_row["amplitude"]))
        amp_decay_rate = amp_trend_global / amp_mean if amp_mean > 0 else 0.0

        # ── Distance brute ───────────────────────────────────────────────────
        dists = grp_cg["dist"].values
        dist_min = float(dists.min())
        dist_mean = float(dists.mean())
        dist_last = float(last_row["dist"])
        dist_last_vs_mean = dist_last - dist_mean
        dist_trend_global = _linear_slope(times_all, dists)

        # ── Maxis (erreur de localisation) ───────────────────────────────────
        maxis_vals = grp_cg["maxis"].values
        maxis_mean = float(maxis_vals.mean())
        maxis_last = float(last_row["maxis"])

        # Distance minimale ajustée : on soustrait l'erreur de localisation
        # pour avoir une borne pessimiste (l'éclair pourrait être plus proche)
        idx_min = int(np.argmin(dists))
        maxis_at_min = float(maxis_vals[idx_min])
        dist_min_adjusted = max(0.0, dist_min - maxis_at_min)

        # Distance du dernier éclair ajustée
        dist_last_adjusted = max(0.0, dist_last - maxis_last)

        # ── Azimuth (direction de l'orage) ───────────────────────────────────
        azimuths = grp_cg["azimuth"].values

        # Direction du dernier éclair
        azimuth_last = float(last_row["azimuth"])

        # Dispersion angulaire : élevée = orage qui entoure l'aéroport (dangereux)
        azimuth_spread = _circular_std(azimuths)

        # Moyenne circulaire
        azimuth_mean = _circular_mean(azimuths)

        # Écart entre dernier azimuth et moyenne
        # > 0 = dernier éclair dans une direction inhabituelle (reprise possible)
        azimuth_last_vs_mean = abs(_angular_diff(azimuth_mean, azimuth_last))

        # Tendance de rotation : vitesse angulaire moyenne entre éclairs consécutifs
        # Orage qui tourne = plus imprévisible
        if len(azimuths) >= 2 and duration > 0:
            angular_diffs = np.array(
                [
                    _angular_diff(azimuths[i], azimuths[i + 1])
                    for i in range(len(azimuths) - 1)
                ]
            )
            # Vitesse angulaire en deg/min
            azimuth_trend = float(
                np.abs(angular_diffs).mean() / (duration / len(azimuths))
            )
        else:
            azimuth_trend = 0.0

        # ── Fenêtre récente 10 min ───────────────────────────────────────────
        t_window10 = t_end - pd.Timedelta(minutes=window_minutes)
        recent_cg = grp_cg[grp_cg["date"] >= t_window10]
        n_cg_recent = len(recent_cg)

        dist_recent_min = (
            float(recent_cg["dist"].min()) if n_cg_recent > 0 else dist_min
        )

        if n_cg_recent >= 2:
            times_rec = (recent_cg["date"] - t_window10).dt.total_seconds().values / 60
            dist_trend_recent = _linear_slope(times_rec, recent_cg["dist"].values)
            amp_trend_recent = _linear_slope(
                times_rec, recent_cg["amplitude"].abs().values
            )
        else:
            dist_trend_recent = dist_trend_global
            amp_trend_recent = amp_trend_global

        # ── Fenêtre récente 5 min ────────────────────────────────────────────
        t_window5 = t_end - pd.Timedelta(minutes=5)
        recent_cg5 = grp_cg[grp_cg["date"] >= t_window5]
        n_cg_last5 = len(recent_cg5)

        if n_cg_last5 >= 2:
            times_rec5 = (recent_cg5["date"] - t_window5).dt.total_seconds().values / 60
            dist_trend_last5 = _linear_slope(times_rec5, recent_cg5["dist"].values)
        else:
            dist_trend_last5 = dist_trend_recent

        # ── Temps entre les 3 derniers éclairs ───────────────────────────────
        if n_cg_total >= 3:
            last3_times = (
                grp_cg.iloc[-3:]["date"] - t_start
            ).dt.total_seconds().values / 60
            inter_time_last3 = float(np.diff(last3_times).mean())
        elif n_cg_total >= 2:
            last2_times = (
                grp_cg.iloc[-2:]["date"] - t_start
            ).dt.total_seconds().values / 60
            inter_time_last3 = float(np.diff(last2_times).mean())
        else:
            inter_time_last3 = 0.0

        # ── Structure temporelle ─────────────────────────────────────────────
        burst_feats = _compute_burst_features(grp_cg["date"], duration, bin_size=5)
        pause_ratio = burst_feats["pause_max"] / duration if duration > 0 else 0.0

        # ── Saison ───────────────────────────────────────────────────────────
        month = t_start.month
        if month in [12, 1, 2]:
            season = 0
        elif month in [3, 4, 5]:
            season = 1
        elif month in [6, 7, 8]:
            season = 2
        else:
            season = 3

        # ── Intra-nuage ──────────────────────────────────────────────────────
        recent_all = grp_all[grp_all["date"] >= t_window10]
        n_ic_total = len(grp_all) - n_cg_total
        ratio_ic_cg = n_ic_total / max(n_cg_total, 1)
        n_ic_recent = len(recent_all) - n_cg_recent
        ratio_ic_recent = n_ic_recent / max(n_cg_recent, 1)
        amp_recent_mean = (
            float(recent_cg["amplitude"].abs().mean()) if n_cg_recent > 0 else 0.0
        )
        time_since_last_cg = 0.0

        records.append(
            {
                # Identifiants
                "airport": airport,
                "airport_alert_id": alert_id,
                # Cibles survie
                "duration": duration,
                "event": event,
                # ── Actives ──────────────────────────────────────────────────────
                "n_cg_total": n_cg_total,
                "n_cg_recent": n_cg_recent,
                "n_cg_last5": n_cg_last5,
                "activity_trend": activity_trend,
                "cg_density": cg_density,
                "amp_max": amp_max,
                "amp_mean": amp_mean,
                "amp_last": amp_last,
                "amp_trend_global": amp_trend_global,
                "amp_trend_recent": amp_trend_recent,
                "amp_decay_rate": amp_decay_rate,
                "dist_min": dist_min,
                "dist_mean": dist_mean,
                "dist_last": dist_last,
                "dist_last_vs_mean": dist_last_vs_mean,
                "dist_recent_min": dist_recent_min,
                "dist_trend_global": dist_trend_global,
                "dist_trend_recent": dist_trend_recent,
                "dist_trend_last5": dist_trend_last5,
                "dist_min_adjusted": dist_min_adjusted,
                "dist_last_adjusted": dist_last_adjusted,
                "maxis_mean": maxis_mean,
                "maxis_last": maxis_last,
                "azimuth_last": azimuth_last,
                "azimuth_spread": azimuth_spread,
                "azimuth_trend": azimuth_trend,
                "azimuth_last_vs_mean": azimuth_last_vs_mean,
                "elapsed_time": elapsed_time,
                "inter_time_last3": inter_time_last3,
                "n_bursts": burst_feats["n_bursts"],
                "activity_variance": burst_feats["activity_variance"],
                "pause_max": burst_feats["pause_max"],
                "pause_ratio": pause_ratio,
                "intensity_persistence": burst_feats["intensity_persistence"],
                "pause_since_peak": burst_feats["pause_since_peak"],
                "resume_risk": burst_feats["resume_risk"],
                "long_pause_count": burst_feats["long_pause_count"],
                "pause_cv": burst_feats["pause_cv"],
                "n_ic_recent": n_ic_recent,
                "ratio_ic_cg": ratio_ic_cg,
                "month": month,
                "season": season,
                # ── Inactives ─────────────────────────────────────────────────────
                "n_ic_total": n_ic_total,
                "ratio_ic_recent": ratio_ic_recent,
                "amp_recent_mean": amp_recent_mean,
                "time_since_last_cg": time_since_last_cg,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/segment_alerts_all_airports_train.csv"
    )
    print(f"Chargement de {path}...")
    df = load_data(path)

    print("Calcul des features...")
    features = compute_alert_features(df, window_minutes=10)

    print(f"\nAlertes : {len(features)}")
    print(f"Features actives   ({len(ACTIVE_FEATURES)}) : {ACTIVE_FEATURES}")
    print(f"Features inactives ({len(INACTIVE_FEATURES)}) : {INACTIVE_FEATURES}")
    print("\nStatistiques durée (minutes) :")
    print(features["duration"].describe().round(1))
    print("\nAperçu des nouvelles features (azimuth + maxis) :")
    new_cols = [
        "duration",
        "azimuth_last",
        "azimuth_spread",
        "azimuth_trend",
        "azimuth_last_vs_mean",
        "dist_min_adjusted",
        "dist_last_adjusted",
        "maxis_mean",
        "maxis_last",
    ]
    print(features[new_cols].describe().round(3))

    out = "data/features.parquet"
    features.to_parquet(out, index=False)
    print(f"\nSauvegardé dans {out}")
