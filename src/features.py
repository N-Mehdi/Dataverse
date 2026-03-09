"""
features.py
-----------
Feature engineering pour la prédiction de fin d'alerte orageuse.

Pour chaque alerte (identifiée par airport + airport_alert_id),
on calcule des features à l'instant du dernier éclair nuage-sol connu,
puis on construit la variable cible pour l'analyse de survie :
  - duration  : durée totale de l'alerte en minutes
  - event     : 1 si l'alerte est terminée (non censurée), 0 sinon

Orientations v7 :
  - Features "queue d'alerte" : caractérisent explicitement la mort de l'orage
    * last_cg_amp_ratio        : dernier éclair faible vs max = signe de fin
    * last3_dist_trend         : les 3 derniers éclairs s'éloignent ?
    * inter_time_acceleration  : intervalles entre éclairs s'allongent progressivement ?
    * final_activity_ratio     : activité fin / activité moyenne (proche 0 = bon signe)
    * dying_score              : score composite de "mort d'orage"
  - Suppression des features quasi-nulles v6 (hour_of_day, n_cg_last2, etc.)
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Features actives
# ---------------------------------------------------------------------------

ACTIVE_FEATURES = [
    # ── Comptage ────────────────────────────────────────────────────────────
    "n_cg_total",
    "activity_trend",
    "cg_density",
    # ── Amplitude ───────────────────────────────────────────────────────────
    "amp_max",
    "amp_trend_global",
    "amp_decay_rate",
    # ── Distance ────────────────────────────────────────────────────────────
    "dist_min",
    "dist_min_adjusted",
    "dist_trend_global",
    # ── Maxis ────────────────────────────────────────────────────────────────
    "maxis_mean",
    # ── Azimuth ──────────────────────────────────────────────────────────────
    "azimuth_spread",
    "azimuth_trend",
    # ── Centroïde ────────────────────────────────────────────────────────────
    "centroid_speed",
    "spatial_spread",
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
    # ── Queue d'alerte (v7) ──────────────────────────────────────────────────
    "last_cg_amp_ratio",  # amplitude dernier éclair / amp max (proche 0 = orage mourant)
    "last3_dist_trend",  # tendance distance sur les 3 derniers éclairs (> 0 = s'éloigne)
    "inter_time_acceleration",  # accélération des intervalles (> 0 = ralentit = bon signe)
    "final_activity_ratio",  # activité 5 dernières min / activité moyenne (proche 0 = fin)
    "dying_score",  # score composite : combine les 4 signaux ci-dessus
]

# ---------------------------------------------------------------------------
# Features inactives
# ---------------------------------------------------------------------------

INACTIVE_FEATURES = [
    # Importance négative ou nulle v6
    "hour_of_day",
    "time_since_prev_alert",
    "n_prev_alerts_6h",
    "n_cg_last2",
    "dist_trend_last2",
    # Importance négative ou nulle v5
    "dist_last_adjusted",
    "n_cg_recent",
    "season",
    "amp_mean",
    "dist_last",
    "azimuth_last_vs_mean",
    "centroid_approach",
    "centroid_dist_last",
    "month",
    "amp_last",
    "centroid_dist_change",
    "dist_mean",
    "dist_recent_min",
    "maxis_last",
    "amp_trend_recent",
    "dist_trend_last5",
    "dist_trend_recent",
    "ratio_ic_cg",
    "n_ic_recent",
    "dist_last_vs_mean",
    "n_cg_last5",
    "n_ic_total",
    "ratio_ic_recent",
    "amp_recent_mean",
    "time_since_last_cg",
    "azimuth_last",
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
    if len(angles_deg) < 2:
        return 0.0
    rad = np.deg2rad(angles_deg)
    S = np.mean(np.sin(rad))
    C = np.mean(np.cos(rad))
    R = np.sqrt(S**2 + C**2)
    return float(np.rad2deg(np.sqrt(-2 * np.log(np.clip(R, 1e-10, 1.0)))))


def _circular_mean(angles_deg: np.ndarray) -> float:
    if len(angles_deg) == 0:
        return 0.0
    rad = np.deg2rad(angles_deg)
    S = np.mean(np.sin(rad))
    C = np.mean(np.cos(rad))
    return float(np.rad2deg(np.arctan2(S, C)) % 360)


def _angular_diff(a1: float, a2: float) -> float:
    return float((a2 - a1 + 180) % 360 - 180)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlam = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


# ---------------------------------------------------------------------------
# Positions des aéroports (WGS84)
# ---------------------------------------------------------------------------

AIRPORT_POSITIONS = {
    "Ajaccio": {"lat": 41.9236, "lon": 8.8029},
    "Bastia": {"lat": 42.5527, "lon": 9.4837},
    "Biarritz": {"lat": 43.4683, "lon": -1.524},
    "Bron": {"lat": 45.7294, "lon": 4.9389},
    "Nantes": {"lat": 47.1532, "lon": -1.6107},
    "Pise": {"lat": 43.695, "lon": 10.399},
}


# ---------------------------------------------------------------------------
# Features centroïde
# ---------------------------------------------------------------------------


def _compute_centroid_features(
    grp_cg: pd.DataFrame,
    airport_lat: float,
    airport_lon: float,
    duration_min: float,
    n_first: int = 5,
    n_last: int = 5,
) -> dict:
    lats = grp_cg["lat"].values
    lons = grp_cg["lon"].values
    n = len(grp_cg)
    n_f = min(n_first, n)
    n_l = min(n_last, n)

    lat_first = float(lats[:n_f].mean())
    lon_first = float(lons[:n_f].mean())
    lat_last = float(lats[-n_l:].mean())
    lon_last = float(lons[-n_l:].mean())

    centroid_travel = _haversine_km(lat_first, lon_first, lat_last, lon_last)
    centroid_speed = centroid_travel / duration_min if duration_min > 0 else 0.0

    centroid_dist_last = _haversine_km(lat_last, lon_last, airport_lat, airport_lon)
    centroid_dist_first = _haversine_km(lat_first, lon_first, airport_lat, airport_lon)
    centroid_dist_change = centroid_dist_last - centroid_dist_first
    centroid_approach = (
        -centroid_dist_change / duration_min if duration_min > 0 else 0.0
    )

    mean_lat = float(lats.mean())
    std_lat_km = float(lats.std()) * 111.0
    std_lon_km = float(lons.std()) * 111.0 * np.cos(np.deg2rad(mean_lat))
    spatial_spread = float(np.sqrt(std_lat_km**2 + std_lon_km**2))

    return {
        "centroid_speed": centroid_speed,
        "centroid_approach": centroid_approach,
        "centroid_dist_last": centroid_dist_last,
        "centroid_dist_change": centroid_dist_change,
        "spatial_spread": spatial_spread,
    }


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

    resume_risk = pause_max * n_bursts
    long_pause_count = int((inter_times > 5).sum()) if len(inter_times) > 0 else 0

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
# Features queue d'alerte (v7)
# ---------------------------------------------------------------------------


def _compute_dying_features(
    grp_cg: pd.DataFrame,
    duration_min: float,
    amp_max: float,
    cg_density: float,
) -> dict:
    """
    Caractérise la 'mort' de l'orage en analysant le comportement
    des derniers éclairs : amplitude, distance, intervalles, activité finale.
    """
    grp_sorted = grp_cg.sort_values("date")
    n = len(grp_sorted)
    t_end = grp_sorted["date"].max()
    t_start = grp_sorted["date"].min()

    # ── last_cg_amp_ratio ─────────────────────────────────────────────────
    # Ratio amplitude dernier éclair / amplitude max
    # Proche de 0 → dernier éclair faible = orage en train de mourir
    amp_last = float(abs(grp_sorted.iloc[-1]["amplitude"]))
    last_cg_amp_ratio = amp_last / amp_max if amp_max > 0 else 1.0

    # ── last3_dist_trend ──────────────────────────────────────────────────
    # Tendance de la distance sur les 3 derniers éclairs
    # Positif → les derniers éclairs s'éloignent = bon signe
    if n >= 3:
        last3 = grp_sorted.iloc[-3:]
        times3 = (last3["date"] - t_start).dt.total_seconds().values / 60
        last3_dist_trend = _linear_slope(times3, last3["dist"].values)
    elif n >= 2:
        last2 = grp_sorted.iloc[-2:]
        times2 = (last2["date"] - t_start).dt.total_seconds().values / 60
        last3_dist_trend = _linear_slope(times2, last2["dist"].values)
    else:
        last3_dist_trend = 0.0

    # ── inter_time_acceleration ───────────────────────────────────────────
    # Les intervalles entre éclairs s'allongent-ils ?
    # Positif → ralentissement progressif = orage en train de s'arrêter
    if n >= 4:
        times_all = (grp_sorted["date"] - t_start).dt.total_seconds().values / 60
        inter_times = np.diff(times_all)
        # Tendance des intervalles : slope positive = intervalles qui s'allongent
        idx = np.arange(len(inter_times), dtype=float)
        inter_time_acceleration = _linear_slope(idx, inter_times)
    else:
        inter_time_acceleration = 0.0

    # ── final_activity_ratio ──────────────────────────────────────────────
    # Activité dans les 5 dernières minutes vs densité globale
    # Proche de 0 → fin d'activité = bon signe
    t_final = t_end - pd.Timedelta(minutes=5)
    n_final = len(grp_sorted[grp_sorted["date"] >= t_final])
    final_density = n_final / 5.0  # éclairs par minute dans les 5 dernières min
    final_activity_ratio = final_density / cg_density if cg_density > 0 else 1.0

    # ── dying_score ───────────────────────────────────────────────────────
    # Score composite normalisé [0, 1] — proche de 1 = orage clairement mourant
    # Combine les 4 signaux avec des poids égaux :
    #   - amplitude faible (last_cg_amp_ratio bas)
    #   - éclairs qui s'éloignent (last3_dist_trend positif, capé à 1)
    #   - intervalles qui s'allongent (inter_time_acceleration positif, capé à 1)
    #   - activité finale faible (final_activity_ratio bas)
    s_amp = 1.0 - min(last_cg_amp_ratio, 1.0)
    s_dist = min(max(last3_dist_trend / 5.0, 0.0), 1.0)  # normalisé sur 5 km/min
    s_accel = min(max(inter_time_acceleration / 5.0, 0.0), 1.0)  # normalisé
    s_act = 1.0 - min(final_activity_ratio, 1.0)
    dying_score = float((s_amp + s_dist + s_accel + s_act) / 4.0)

    return {
        "last_cg_amp_ratio": last_cg_amp_ratio,
        "last3_dist_trend": last3_dist_trend,
        "inter_time_acceleration": inter_time_acceleration,
        "final_activity_ratio": final_activity_ratio,
        "dying_score": dying_score,
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

        # ── Distance ─────────────────────────────────────────────────────────
        dists = grp_cg["dist"].values
        dist_min = float(dists.min())
        dist_mean = float(dists.mean())
        dist_last = float(last_row["dist"])
        dist_last_vs_mean = dist_last - dist_mean
        dist_trend_global = _linear_slope(times_all, dists)

        # ── Maxis ────────────────────────────────────────────────────────────
        maxis_vals = grp_cg["maxis"].values
        maxis_mean = float(maxis_vals.mean())
        maxis_last = float(last_row["maxis"])
        idx_min = int(np.argmin(dists))
        dist_min_adjusted = max(0.0, dist_min - float(maxis_vals[idx_min]))
        dist_last_adjusted = max(0.0, dist_last - maxis_last)

        # ── Azimuth ──────────────────────────────────────────────────────────
        azimuths = grp_cg["azimuth"].values
        azimuth_last = float(last_row["azimuth"])
        azimuth_spread = _circular_std(azimuths)
        azimuth_mean = _circular_mean(azimuths)
        azimuth_last_vs_mean = abs(_angular_diff(azimuth_mean, azimuth_last))

        if len(azimuths) >= 2 and duration > 0:
            angular_diffs = np.array(
                [
                    _angular_diff(azimuths[i], azimuths[i + 1])
                    for i in range(len(azimuths) - 1)
                ]
            )
            azimuth_trend = float(
                np.abs(angular_diffs).mean() / (duration / len(azimuths))
            )
        else:
            azimuth_trend = 0.0

        # ── Centroïde ────────────────────────────────────────────────────────
        ap = AIRPORT_POSITIONS.get(airport, None)
        if ap is not None and "lat" in grp_cg.columns and "lon" in grp_cg.columns:
            centroid_feats = _compute_centroid_features(
                grp_cg, ap["lat"], ap["lon"], duration
            )
        else:
            centroid_feats = {
                "centroid_speed": 0.0,
                "centroid_approach": 0.0,
                "centroid_dist_last": dist_last,
                "centroid_dist_change": 0.0,
                "spatial_spread": 0.0,
            }

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

        # ── Queue d'alerte (v7) ──────────────────────────────────────────────
        dying_feats = _compute_dying_features(grp_cg, duration, amp_max, cg_density)

        # ── Saison ───────────────────────────────────────────────────────────
        month = t_start.month
        season = (
            0
            if month in [12, 1, 2]
            else 1
            if month in [3, 4, 5]
            else 2
            if month in [6, 7, 8]
            else 3
        )

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
                "airport": airport,
                "airport_alert_id": alert_id,
                "duration": duration,
                "event": event,
                # ── Actives ──────────────────────────────────────────────────────
                "n_cg_total": n_cg_total,
                "activity_trend": activity_trend,
                "cg_density": cg_density,
                "amp_max": amp_max,
                "amp_trend_global": amp_trend_global,
                "amp_decay_rate": amp_decay_rate,
                "dist_min": dist_min,
                "dist_min_adjusted": dist_min_adjusted,
                "dist_trend_global": dist_trend_global,
                "maxis_mean": maxis_mean,
                "azimuth_spread": azimuth_spread,
                "azimuth_trend": azimuth_trend,
                "centroid_speed": centroid_feats["centroid_speed"],
                "spatial_spread": centroid_feats["spatial_spread"],
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
                # ── Queue d'alerte (v7) ──────────────────────────────────────────
                "last_cg_amp_ratio": dying_feats["last_cg_amp_ratio"],
                "last3_dist_trend": dying_feats["last3_dist_trend"],
                "inter_time_acceleration": dying_feats["inter_time_acceleration"],
                "final_activity_ratio": dying_feats["final_activity_ratio"],
                "dying_score": dying_feats["dying_score"],
                # ── Inactives ─────────────────────────────────────────────────────
                "hour_of_day": float(t_start.hour + t_start.minute / 60.0),
                "time_since_prev_alert": 999.0,
                "n_prev_alerts_6h": 0,
                "n_cg_last2": len(
                    grp_cg[grp_cg["date"] >= t_end - pd.Timedelta(minutes=2)]
                ),
                "dist_trend_last2": dist_trend_recent,
                "n_cg_recent": n_cg_recent,
                "n_cg_last5": n_cg_last5,
                "dist_last": dist_last,
                "dist_last_vs_mean": dist_last_vs_mean,
                "dist_mean": dist_mean,
                "dist_recent_min": dist_recent_min,
                "dist_trend_recent": dist_trend_recent,
                "dist_trend_last5": dist_trend_last5,
                "dist_last_adjusted": dist_last_adjusted,
                "maxis_last": maxis_last,
                "amp_mean": amp_mean,
                "amp_last": amp_last,
                "amp_trend_recent": amp_trend_recent,
                "azimuth_last": azimuth_last,
                "azimuth_last_vs_mean": azimuth_last_vs_mean,
                "centroid_approach": centroid_feats["centroid_approach"],
                "centroid_dist_last": centroid_feats["centroid_dist_last"],
                "centroid_dist_change": centroid_feats["centroid_dist_change"],
                "n_ic_recent": n_ic_recent,
                "ratio_ic_cg": ratio_ic_cg,
                "month": month,
                "season": season,
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
    print("\nAperçu des features queue d'alerte (v7) :")
    dying_cols = [
        "last_cg_amp_ratio",
        "last3_dist_trend",
        "inter_time_acceleration",
        "final_activity_ratio",
        "dying_score",
    ]
    print(features[dying_cols].describe().round(3))

    out = "data/features.parquet"
    features.to_parquet(out, index=False)
    print(f"\nSauvegardé dans {out}")
