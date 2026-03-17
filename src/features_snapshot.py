"""
features_snapshot.py
--------------------
Génère des snapshots temporels pour l'approche XGBoost.

Pour chaque alerte, on génère des snapshots :
  1. PENDANT l'alerte (à intervalles réguliers) → y=1 car l'orage continue
  2. APRÈS le dernier éclair CG (période de silence) → y=0 ou y=1 selon reprise

La feature clé : time_since_last_cg
  - Pendant l'alerte : faible (dernier éclair récent)
  - Après le dernier éclair : croît avec le temps → signal fort de fin d'orage

Cible y :
    y = 1 si un éclair CG survient dans les `horizon` prochaines minutes
    y = 0 sinon (on peut lever l'alerte)
"""

import pandas as pd
import numpy as np
import sys

# Intervalle entre snapshots (minutes)
SNAPSHOT_INTERVAL = 2

# Horizon de prédiction (minutes)
HORIZON = 30

# Durée max de la période de silence après le dernier éclair (minutes)
SILENCE_WINDOW = 35

# Nombre minimum d'éclairs CG pour générer un snapshot
MIN_CG = 2

AIRPORT_POSITIONS = {
    "Ajaccio": {"lat": 41.9236, "lon": 8.8029},
    "Bastia": {"lat": 42.5527, "lon": 9.4837},
    "Biarritz": {"lat": 43.4683, "lon": -1.524},
    "Bron": {"lat": 45.7294, "lon": 4.9389},
    "Nantes": {"lat": 47.1532, "lon": -1.6107},
    "Pise": {"lat": 43.695, "lon": 10.399},
}


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------


def _slope(t: np.ndarray, v: np.ndarray) -> float:
    if len(t) < 2:
        return 0.0
    t = t - t[0]
    tm, vm = t.mean(), v.mean()
    den = ((t - tm) ** 2).sum()
    return float(((t - tm) * (v - vm)).sum() / den) if den > 0 else 0.0


def _circ_std(a: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    r = np.deg2rad(a)
    R = np.sqrt(np.mean(np.sin(r)) ** 2 + np.mean(np.cos(r)) ** 2)
    return float(np.rad2deg(np.sqrt(-2 * np.log(np.clip(R, 1e-10, 1)))))


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


# ---------------------------------------------------------------------------
# Features à partir de l'historique jusqu'à t
# ---------------------------------------------------------------------------


def compute_snapshot_features(
    hist: pd.DataFrame,
    t_snapshot: pd.Timestamp,
    t_alert_start: pd.Timestamp,
    airport: str,
) -> dict:
    n = len(hist)
    elapsed = (t_snapshot - t_alert_start).total_seconds() / 60
    last_cg_time = hist["date"].max()
    time_since_last_cg = (t_snapshot - last_cg_time).total_seconds() / 60

    times = (hist["date"] - t_alert_start).dt.total_seconds().values / 60
    amps = hist["amplitude"].abs().values
    dists = hist["dist"].values

    amp_max = float(amps.max())
    amp_mean = float(amps.mean())
    amp_last = float(amps[-1])
    amp_trend = _slope(times, amps)
    amp_decay = amp_trend / amp_mean if amp_mean > 0 else 0.0
    last_cg_amp_ratio = amp_last / amp_max if amp_max > 0 else 1.0

    dist_min = float(dists.min())
    dist_mean = float(dists.mean())
    dist_last = float(dists[-1])
    dist_trend = _slope(times, dists)

    maxis = hist["maxis"].values
    maxis_mean = float(maxis.mean())
    dist_min_adj = max(0.0, dist_min - float(maxis[np.argmin(dists)]))

    duration_cg = (last_cg_time - t_alert_start).total_seconds() / 60
    cg_density = n / duration_cg if duration_cg > 0 else float(n)

    t_mid = duration_cg / 2
    activity_trend = int((times > t_mid).sum()) - int((times <= t_mid).sum())

    azimuth_spread = _circ_std(hist["azimuth"].values)

    ap = AIRPORT_POSITIONS.get(airport)
    if ap and "lat" in hist.columns:
        lats, lons = hist["lat"].values, hist["lon"].values
        n_edge = min(5, n)
        lat_f, lon_f = lats[:n_edge].mean(), lons[:n_edge].mean()
        lat_l, lon_l = lats[-n_edge:].mean(), lons[-n_edge:].mean()
        centroid_speed = (
            _haversine(lat_f, lon_f, lat_l, lon_l) / duration_cg
            if duration_cg > 0
            else 0.0
        )
        std_lat = float(lats.std()) * 111.0
        std_lon = float(lons.std()) * 111.0 * np.cos(np.deg2rad(lats.mean()))
        spatial_spread = float(np.sqrt(std_lat**2 + std_lon**2))
    else:
        centroid_speed = 0.0
        spatial_spread = 0.0

    if n >= 3:
        inter_time_last3 = float(np.diff(times[-3:]).mean())
    elif n >= 2:
        inter_time_last3 = float(np.diff(times[-2:]).mean())
    else:
        inter_time_last3 = 0.0

    if n >= 4:
        inter = np.diff(times)
        inter_time_acceleration = _slope(np.arange(len(inter), dtype=float), inter)
    else:
        inter_time_acceleration = 0.0

    dur_hist = max(duration_cg, 1.0)
    n_bins = max(1, int(np.ceil(dur_hist / 5)))
    counts, bin_edges = np.histogram(times, bins=n_bins, range=(0, dur_hist))
    inter_all = np.diff(np.sort(times))
    pause_max = float(inter_all.max()) if len(inter_all) > 0 else 0.0
    pause_ratio = pause_max / dur_hist
    activity_variance = float(counts.var())
    mean_count = counts.mean()
    intensity_persistence = (
        float((counts > mean_count).mean()) if mean_count > 0 else 0.0
    )
    peak_bin = int(np.argmax(counts))
    pause_since_peak = max(0.0, dur_hist - bin_edges[peak_bin])
    is_active = counts > 0
    n_bursts = int(np.sum((~is_active[:-1]) & is_active[1:]))
    resume_risk = pause_max * n_bursts
    long_pause_count = int((inter_all > 5).sum()) if len(inter_all) > 0 else 0
    pause_cv = (
        float(inter_all.std() / inter_all.mean())
        if (len(inter_all) > 1 and inter_all.mean() > 0)
        else 0.0
    )

    # Activité dans les 5 dernières minutes avant le snapshot
    t_final_start = elapsed - 5
    n_final = int((times >= t_final_start).sum())
    final_density = n_final / 5.0
    final_activity_ratio = final_density / cg_density if cg_density > 0 else 0.0

    s_amp = 1.0 - min(last_cg_amp_ratio, 1.0)
    s_dist = min(max(dist_trend / 5.0, 0.0), 1.0)
    s_accel = min(max(inter_time_acceleration / 5.0, 0.0), 1.0)
    s_act = 1.0 - min(final_activity_ratio, 1.0)
    dying_score = float((s_amp + s_dist + s_accel + s_act) / 4.0)

    return {
        "time_since_last_cg": time_since_last_cg,
        "elapsed_time": elapsed,
        "n_cg_total": n,
        "amp_max": amp_max,
        "amp_mean": amp_mean,
        "amp_decay_rate": amp_decay,
        "amp_trend_global": amp_trend,
        "last_cg_amp_ratio": last_cg_amp_ratio,
        "dist_min": dist_min,
        "dist_min_adjusted": dist_min_adj,
        "dist_mean": dist_mean,
        "dist_last": dist_last,
        "dist_trend_global": dist_trend,
        "maxis_mean": maxis_mean,
        "azimuth_spread": azimuth_spread,
        "centroid_speed": centroid_speed,
        "spatial_spread": spatial_spread,
        "cg_density": cg_density,
        "activity_trend": activity_trend,
        "inter_time_last3": inter_time_last3,
        "inter_time_acceleration": inter_time_acceleration,
        "pause_max": pause_max,
        "pause_ratio": pause_ratio,
        "activity_variance": activity_variance,
        "intensity_persistence": intensity_persistence,
        "pause_since_peak": pause_since_peak,
        "n_bursts": n_bursts,
        "resume_risk": resume_risk,
        "long_pause_count": long_pause_count,
        "pause_cv": pause_cv,
        "final_activity_ratio": final_activity_ratio,
        "dying_score": dying_score,
    }


# ---------------------------------------------------------------------------
# Génération des snapshots
# ---------------------------------------------------------------------------


def generate_snapshots(
    df: pd.DataFrame,
    horizon: int = HORIZON,
    silence_window: int = SILENCE_WINDOW,
    interval: int = SNAPSHOT_INTERVAL,
) -> pd.DataFrame:
    alerts = df[df["airport_alert_id"].notna()].copy()
    cg_all = alerts[alerts["icloud"] == False].sort_values("date")

    # Index global par aéroport pour calculer la cible (éclairs futurs)
    cg_by_airport = {
        airport: grp.sort_values("date") for airport, grp in cg_all.groupby("airport")
    }

    records = []

    for (airport, alert_id), grp in cg_all.groupby(["airport", "airport_alert_id"]):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) < MIN_CG:
            continue

        t_start = grp["date"].min()
        t_last_cg = grp["date"].max()
        all_cg_ap = cg_by_airport[airport]

        def make_snapshot(t_snap, hist):
            t_horizon = t_snap + pd.Timedelta(minutes=horizon)
            future = all_cg_ap[
                (all_cg_ap["date"] > t_snap) & (all_cg_ap["date"] <= t_horizon)
            ]
            y = 1 if len(future) > 0 else 0
            feats = compute_snapshot_features(hist, t_snap, t_start, airport)
            feats["airport"] = airport
            feats["airport_alert_id"] = alert_id
            feats["t_snapshot"] = t_snap
            feats["y"] = y
            return feats

        # ── PENDANT l'alerte ── NOUVEAU (un snapshot par éclair)
        for i in range(MIN_CG, len(grp) + 1):
            hist = grp.iloc[:i]
            t_snap = hist["date"].max()
            records.append(make_snapshot(t_snap, hist))    

        # ── APRÈS le dernier éclair (silence) ────────────────────────────────
        # time_since_last_cg croît ici → signal fort de fin d'orage
        t_snap = t_last_cg + pd.Timedelta(minutes=interval)
        t_end = t_last_cg + pd.Timedelta(minutes=silence_window)
        while t_snap <= t_end:
            records.append(make_snapshot(t_snap, grp))  # hist = alerte entière
            t_snap += pd.Timedelta(minutes=interval)

    df_snap = pd.DataFrame(records)

    n_total = len(df_snap)
    n_pos = int(df_snap["y"].sum())
    n_neg = n_total - n_pos
    print(f"Snapshots générés : {n_total}")
    print(f"  y=1 (alerte continue) : {n_pos} ({100 * n_pos / n_total:.1f}%)")
    print(f"  y=0 (fin possible)    : {n_neg} ({100 * n_neg / n_total:.1f}%)")
    print(f"  time_since_last_cg mean : {df_snap['time_since_last_cg'].mean():.1f} min")
    print(f"  time_since_last_cg max  : {df_snap['time_since_last_cg'].max():.1f} min")
    return df_snap


FEATURE_COLS = [
    "time_since_last_cg",
    "elapsed_time",
    "n_cg_total",
    "amp_max",
    "amp_mean",
    "amp_decay_rate",
    "amp_trend_global",
    "last_cg_amp_ratio",
    "dist_min",
    "dist_min_adjusted",
    "dist_mean",
    "dist_last",
    "dist_trend_global",
    "maxis_mean",
    "azimuth_spread",
    "centroid_speed",
    "spatial_spread",
    "cg_density",
    "activity_trend",
    "inter_time_last3",
    "inter_time_acceleration",
    "pause_max",
    "pause_ratio",
    "activity_variance",
    "intensity_persistence",
    "pause_since_peak",
    "n_bursts",
    "resume_risk",
    "long_pause_count",
    "pause_cv",
    "final_activity_ratio",
    "dying_score",
]


if __name__ == "__main__":
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/segment_alerts_all_airports_train.csv"
    )
    print(f"Chargement de {path}...")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    print("Génération des snapshots...")
    snapshots = generate_snapshots(df, horizon=HORIZON, silence_window=SILENCE_WINDOW)

    print(f"\nFeatures : {len(FEATURE_COLS)}")
    out = "data/snapshots.parquet"
    snapshots.to_parquet(out, index=False)
    print(f"Sauvegardé dans {out}")
