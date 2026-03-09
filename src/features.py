"""
features.py
-----------
Feature engineering pour la prédiction de fin d'alerte orageuse.

Pour chaque alerte (identifiée par airport + airport_alert_id),
on calcule des features à l'instant du dernier éclair nuage-sol connu,
puis on construit la variable cible pour l'analyse de survie :
  - duration  : durée totale de l'alerte en minutes
  - event     : 1 si l'alerte est terminée (non censurée), 0 sinon
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
    "n_cg_last5",  # éclairs CG dans les 5 dernières minutes (fenêtre courte)
    "activity_trend",  # 2ème moitié - 1ère moitié (< 0 = décroissance)
    "cg_density",  # n_cg_total / duration — densité d'éclairs/min
    # ── Amplitude (intensité en kA) ─────────────────────────────────────────
    "amp_max",  # intensité maximale
    "amp_mean",  # intensité moyenne
    "amp_last",  # amplitude du dernier éclair CG (fort = orage encore intense)
    "amp_trend_global",  # pente régression linéaire amplitude/temps (kA/min)
    "amp_trend_recent",  # idem sur les 10 dernières minutes
    "amp_decay_rate",  # amp_trend_global / amp_mean — décroissance normalisée
    # ── Distance à l'aéroport ───────────────────────────────────────────────
    "dist_min",  # distance minimale atteinte
    "dist_mean",  # distance moyenne
    "dist_last",  # distance du dernier éclair CG
    "dist_last_vs_mean",  # dist_last - dist_mean (> 0 = s'éloigne, < 0 = se rapproche)
    "dist_recent_min",  # distance minimale sur les 10 dernières minutes
    "dist_trend_global",  # pente régression linéaire distance/temps (km/min)
    "dist_trend_recent",  # idem sur les 10 dernières minutes
    "dist_trend_last5",  # tendance distance sur les 5 dernières minutes
    # ── Temporel ────────────────────────────────────────────────────────────
    "elapsed_time",  # durée déjà écoulée depuis le début de l'alerte (min)
    "inter_time_last3",  # temps moyen entre les 3 derniers éclairs CG
    # ── Persistance / structure temporelle ──────────────────────────────────
    "n_bursts",  # nombre de reprises d'activité après un creux
    "activity_variance",  # variance du comptage par fenêtre de 5 min
    "pause_max",  # plus longue pause entre deux éclairs CG (min)
    "pause_ratio",  # pause_max / elapsed_time
    "intensity_persistence",  # proportion du temps avec activité > moyenne
    "pause_since_peak",  # minutes écoulées depuis le pic d'activité
    # ── Risque de reprise (orages longs avec pauses) ─────────────────────────
    "resume_risk",  # n_bursts * pause_max — score de risque de reprise
    "long_pause_count",  # nombre de pauses > 10 min
    "pause_cv",  # coefficient de variation des pauses (irrégularité)
    # ── Intra-nuage (réactivé) ────────────────────────────────────────────────
    "n_ic_recent",  # activité intra-nuage récente = orage encore électriquement actif
    "ratio_ic_cg",  # ratio intra-nuage / nuage-sol global
    # ── Saisonnalité ─────────────────────────────────────────────────────────
    "month",  # mois de l'alerte (orages d'été plus imprévisibles)
    "season",  # saison encodée (0=hiver, 1=printemps, 2=été, 3=automne)
]

# ---------------------------------------------------------------------------
# Features inactives
# ---------------------------------------------------------------------------

INACTIVE_FEATURES = [
    "n_ic_total",  # éclairs intra-nuage total (corrélé à ratio_ic_cg)
    "ratio_ic_recent",  # ratio intra-nuage récents
    "amp_recent_mean",  # amplitude moyenne récente (redondant avec amp_trend_recent)
    "time_since_last_cg",  # toujours 0 dans le train set, utile en inférence
]


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Sous-ensembles utiles
# ---------------------------------------------------------------------------


def get_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne uniquement les éclairs appartenant à une alerte (< 20 km)."""
    return df[df["airport_alert_id"].notna()].copy()


# ---------------------------------------------------------------------------
# Utilitaire : pente de régression linéaire
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


# ---------------------------------------------------------------------------
# Features structure temporelle + risque de reprise
# ---------------------------------------------------------------------------


def _compute_burst_features(
    dates: pd.Series,
    duration_min: float,
    bin_size: int = 5,
) -> dict:
    """
    Calcule des features capturant la structure temporelle de l'activité CG,
    avec focus sur la détection des orages longs avec reprises.
    """
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
    times_min_sorted = np.sort(times_min)

    # ── Comptage par fenêtre ─────────────────────────────────────────────────
    n_bins = max(1, int(np.ceil(duration_min / bin_size)))
    counts, bin_edges = np.histogram(times_min, bins=n_bins, range=(0, duration_min))

    # n_bursts : transitions creux→actif
    is_active = counts > 0
    n_bursts = int(np.sum((~is_active[:-1]) & is_active[1:]))

    activity_variance = float(counts.var())

    # ── Pauses inter-éclairs ─────────────────────────────────────────────────
    inter_times = np.diff(times_min_sorted)
    pause_max = float(inter_times.max()) if len(inter_times) > 0 else 0.0

    # long_pause_count : nombre de pauses > 10 min
    # Un orage qui s'interrompt plusieurs fois longtemps est très suspect
    long_pause_count = int((inter_times > 10.0).sum()) if len(inter_times) > 0 else 0

    # pause_cv : coefficient de variation des pauses
    # Élevé = pauses très irrégulières = orage chaotique, imprévisible
    if len(inter_times) > 1 and inter_times.mean() > 0:
        pause_cv = float(inter_times.std() / inter_times.mean())
    else:
        pause_cv = 0.0

    # intensity_persistence
    mean_count = counts.mean()
    intensity_persistence = (
        float((counts > mean_count).mean()) if mean_count > 0 else 0.0
    )

    # pause_since_peak : minutes depuis le pic d'activité jusqu'à la fin
    # Si le pic est récent → orage encore intense
    # Si le pic est lointain avec des reprises → très dangereux
    peak_bin = int(np.argmax(counts))
    peak_time = bin_edges[peak_bin]
    pause_since_peak = max(0.0, float(duration_min - peak_time))

    # resume_risk : score composite orages longs avec reprises
    # n_bursts * pause_max : beaucoup de reprises + longues pauses = danger
    resume_risk = float(n_bursts * pause_max)

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
# Encodage de la saison
# ---------------------------------------------------------------------------


def _get_season(month: int) -> int:
    """0=hiver, 1=printemps, 2=été, 3=automne"""
    if month in (12, 1, 2):
        return 0
    elif month in (3, 4, 5):
        return 1
    elif month in (6, 7, 8):
        return 2
    else:
        return 3


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

        # ── Début et fin de l'alerte ─────────────────────────────────────────
        t_start = grp_cg["date"].min()
        t_end = grp_cg["date"].max()
        duration = (t_end - t_start).total_seconds() / 60
        elapsed_time = duration
        event = 1

        times_all = (grp_cg["date"] - t_start).dt.total_seconds().values / 60

        # ── Saisonnalité ─────────────────────────────────────────────────────
        month = int(t_start.month)
        season = _get_season(month)

        # ── Comptage ─────────────────────────────────────────────────────────
        n_cg_total = len(grp_cg)
        t_mid = t_start + (t_end - t_start) / 2
        n_cg_first_half = len(grp_cg[grp_cg["date"] <= t_mid])
        n_cg_second_half = len(grp_cg[grp_cg["date"] > t_mid])
        activity_trend = n_cg_second_half - n_cg_first_half

        cg_density = float(n_cg_total / duration) if duration > 0 else 0.0

        # ── Amplitude globale ────────────────────────────────────────────────
        amps = grp_cg["amplitude"].abs().values
        amp_max = float(amps.max())
        amp_mean = float(amps.mean())
        amp_last = float(amps[-1])
        amp_trend_global = _linear_slope(times_all, amps)
        amp_decay_rate = float(amp_trend_global / amp_mean) if amp_mean > 0 else 0.0

        # ── Distance globale ─────────────────────────────────────────────────
        dists = grp_cg["dist"].values
        dist_min = float(dists.min())
        dist_mean = float(dists.mean())
        dist_last = float(dists[-1])
        dist_last_vs_mean = dist_last - dist_mean
        dist_trend_global = _linear_slope(times_all, dists)

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
        recent5_cg = grp_cg[grp_cg["date"] >= t_window5]
        n_cg_last5 = len(recent5_cg)

        if n_cg_last5 >= 2:
            times_rec5 = (recent5_cg["date"] - t_window5).dt.total_seconds().values / 60
            dist_trend_last5 = _linear_slope(times_rec5, recent5_cg["dist"].values)
        else:
            dist_trend_last5 = dist_trend_global

        # ── Temps entre les 3 derniers éclairs ───────────────────────────────
        if n_cg_total >= 3:
            last3_times = times_all[-3:]
            inter_time_last3 = float(np.diff(last3_times).mean())
        elif n_cg_total >= 2:
            inter_time_last3 = float(times_all[-1] - times_all[-2])
        else:
            inter_time_last3 = 0.0

        # ── Structure temporelle + risque de reprise ─────────────────────────
        burst_feats = _compute_burst_features(grp_cg["date"], duration, bin_size=5)
        pause_ratio = burst_feats["pause_max"] / duration if duration > 0 else 0.0

        # ── Features intra-nuage ─────────────────────────────────────────────
        t_window_ic = t_end - pd.Timedelta(minutes=window_minutes)
        recent_all = grp_all[grp_all["date"] >= t_window_ic]
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
                # ── Actives ──────────────────────────────────────────────────
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
                # ── Inactives ─────────────────────────────────────────────────
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
    print("\nAperçu des nouvelles features (risque de reprise) :")
    new_cols = [
        "duration",
        "n_bursts",
        "pause_max",
        "resume_risk",
        "long_pause_count",
        "pause_cv",
        "pause_since_peak",
        "inter_time_last3",
        "dist_last_vs_mean",
    ]
    print(features[new_cols].describe().round(3))

    out = "data/features.parquet"
    features.to_parquet(out, index=False)
    print(f"\nSauvegardé dans {out}")
