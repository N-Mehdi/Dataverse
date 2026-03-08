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
    "activity_trend",  # 2ème moitié - 1ère moitié (< 0 = décroissance)
    # ── Amplitude (intensité en kA) ─────────────────────────────────────────
    "amp_max",  # intensité maximale
    "amp_mean",  # intensité moyenne
    "amp_trend_global",  # pente régression linéaire amplitude/temps (kA/min)
    "amp_trend_recent",  # idem sur les 10 dernières minutes
    # ── Distance à l'aéroport ───────────────────────────────────────────────
    "dist_min",  # distance minimale atteinte
    "dist_mean",  # distance moyenne
    "dist_recent_min",  # distance minimale sur les 10 dernières minutes
    "dist_trend_global",  # pente régression linéaire distance/temps (km/min)
    "dist_trend_recent",  # idem sur les 10 dernières minutes
    # ── Temporel ────────────────────────────────────────────────────────────
    "elapsed_time",  # durée déjà écoulée depuis le début de l'alerte (min)
    # ── Persistance / structure temporelle ──────────────────────────────────
    # (nouvelles features pour mieux détecter les orages longs et irréguliers)
    "n_bursts",  # nombre de reprises d'activité après un creux
    "activity_variance",  # variance du comptage par fenêtre de 5 min
    "pause_max",  # plus longue pause entre deux éclairs CG (min)
    "intensity_persistence",  # proportion du temps avec activité > moyenne
    "pause_ratio",  # pause_max / elapsed_time — relativise la pause par rapport à la durée totale
]

# ---------------------------------------------------------------------------
# Features inactives (calculées mais importance faible dans RSF v1)
# Conservées dans le parquet pour usage futur
# ---------------------------------------------------------------------------

INACTIVE_FEATURES = [
    "n_ic_total",  # éclairs intra-nuage total
    "ratio_ic_cg",  # ratio intra-nuage / nuage-sol
    "n_ic_recent",  # intra-nuage récents
    "ratio_ic_recent",  # ratio intra-nuage récents
    "amp_recent_mean",  # amplitude moyenne récente
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
    """
    Calcule la pente d'une régression linéaire values ~ temps.
    Retourne 0.0 si moins de 2 points.
    Positif = augmentation, négatif = diminution.
    """
    if len(times_min) < 2:
        return 0.0
    t = times_min - times_min[0]
    t_mean = t.mean()
    v_mean = values.mean()
    num = ((t - t_mean) * (values - v_mean)).sum()
    den = ((t - t_mean) ** 2).sum()
    return float(num / den) if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Nouvelles features : structure temporelle de l'activité
# ---------------------------------------------------------------------------


def _compute_burst_features(
    dates: pd.Series,
    duration_min: float,
    bin_size: int = 5,
) -> dict:
    """
    Calcule des features capturant la structure temporelle de l'activité CG.

    Paramètres
    ----------
    dates      : Series de timestamps des éclairs CG (triés)
    duration_min : durée totale de l'alerte en minutes
    bin_size   : taille des fenêtres temporelles en minutes

    Retourne
    --------
    dict avec :
      n_bursts             : nombre de reprises d'activité après un creux
      activity_variance    : variance du comptage par fenêtre de bin_size min
      pause_max            : plus longue pause entre deux éclairs consécutifs (min)
      intensity_persistence: proportion de fenêtres avec activité > moyenne
    """
    if len(dates) < 2 or duration_min <= 0:
        return {
            "n_bursts": 0,
            "activity_variance": 0.0,
            "pause_max": 0.0,
            "intensity_persistence": 0.0,
        }

    t_start = dates.min()

    # ── Comptage par fenêtre de bin_size minutes ─────────────────────────────
    n_bins = max(1, int(np.ceil(duration_min / bin_size)))
    times_min = (dates - t_start).dt.total_seconds().values / 60
    counts, _ = np.histogram(times_min, bins=n_bins, range=(0, duration_min))

    # n_bursts : nombre de transitions creux→actif dans la séquence temporelle
    # Un "creux" = fenêtre avec 0 éclair, une "reprise" = fenêtre > 0 qui suit un creux
    is_active = counts > 0
    n_bursts = int(np.sum((~is_active[:-1]) & is_active[1:]))

    # activity_variance : variance du nombre d'éclairs par fenêtre
    activity_variance = float(counts.var())

    # pause_max : plus longue pause entre deux éclairs CG consécutifs
    inter_times = np.diff(np.sort(times_min))
    pause_max = float(inter_times.max()) if len(inter_times) > 0 else 0.0

    # intensity_persistence : proportion de fenêtres avec activité > moyenne
    mean_count = counts.mean()
    if mean_count > 0:
        intensity_persistence = float((counts > mean_count).mean())
    else:
        intensity_persistence = 0.0

    return {
        "n_bursts": n_bursts,
        "activity_variance": activity_variance,
        "pause_max": pause_max,
        "intensity_persistence": intensity_persistence,
    }


# ---------------------------------------------------------------------------
# Features par alerte
# ---------------------------------------------------------------------------


def compute_alert_features(df: pd.DataFrame, window_minutes: int = 10) -> pd.DataFrame:
    """
    Pour chaque alerte, calcule des features à partir de la séquence d'éclairs.

    Parameters
    ----------
    df : DataFrame complet (toutes colonnes)
    window_minutes : fenêtre glissante en minutes pour les features récentes

    Returns
    -------
    DataFrame avec une ligne par alerte et les colonnes :
        airport, airport_alert_id,
        duration, event,       <- variables cibles (survie)
        ACTIVE_FEATURES        <- utilisées par les modèles
        INACTIVE_FEATURES      <- conservées pour usage futur
    """
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
        duration = (t_end - t_start).total_seconds() / 60  # minutes
        elapsed_time = duration  # dans le train set = durée totale observée
        event = 1  # toutes les alertes du train set sont terminées

        # Temps en minutes depuis t_start pour chaque éclair (pour régressions)
        times_all = (grp_cg["date"] - t_start).dt.total_seconds().values / 60

        # ── Comptage ─────────────────────────────────────────────────────────
        n_cg_total = len(grp_cg)
        t_mid = t_start + (t_end - t_start) / 2
        n_cg_first_half = len(grp_cg[grp_cg["date"] <= t_mid])
        n_cg_second_half = len(grp_cg[grp_cg["date"] > t_mid])
        activity_trend = n_cg_second_half - n_cg_first_half

        # ── Amplitude globale ────────────────────────────────────────────────
        amps = grp_cg["amplitude"].abs().values
        amp_max = float(amps.max())
        amp_mean = float(amps.mean())
        amp_trend_global = _linear_slope(times_all, amps)

        # ── Distance globale ─────────────────────────────────────────────────
        dists = grp_cg["dist"].values
        dist_min = float(dists.min())
        dist_mean = float(dists.mean())
        dist_trend_global = _linear_slope(times_all, dists)

        # ── Fenêtre récente ──────────────────────────────────────────────────
        t_window = t_end - pd.Timedelta(minutes=window_minutes)
        recent_cg = grp_cg[grp_cg["date"] >= t_window]
        n_cg_recent = len(recent_cg)

        dist_recent_min = (
            float(recent_cg["dist"].min()) if n_cg_recent > 0 else dist_min
        )

        if n_cg_recent >= 2:
            times_rec = (recent_cg["date"] - t_window).dt.total_seconds().values / 60
            dist_trend_recent = _linear_slope(times_rec, recent_cg["dist"].values)
            amp_trend_recent = _linear_slope(
                times_rec, recent_cg["amplitude"].abs().values
            )
        else:
            dist_trend_recent = dist_trend_global
            amp_trend_recent = amp_trend_global

        # ── Structure temporelle (nouvelles features) ────────────────────────
        burst_feats = _compute_burst_features(grp_cg["date"], duration, bin_size=5)

        # pause_ratio : pause_max rapportée à la durée totale
        # Ex : pause de 15 min sur une alerte de 20 min = 0.75 (inquiétant)
        #      pause de 15 min sur une alerte de 120 min = 0.125 (moins inquiétant)
        pause_ratio = burst_feats["pause_max"] / duration if duration > 0 else 0.0

        # ── Features inactives ───────────────────────────────────────────────
        recent_all = grp_all[grp_all["date"] >= t_window]
        n_ic_total = len(grp_all) - n_cg_total
        ratio_ic_cg = n_ic_total / max(n_cg_total, 1)
        n_ic_recent = len(recent_all) - n_cg_recent
        ratio_ic_recent = n_ic_recent / max(n_cg_recent, 1)
        amp_recent_mean = (
            float(recent_cg["amplitude"].abs().mean()) if n_cg_recent > 0 else 0.0
        )
        time_since_last_cg = 0.0  # toujours 0 dans le train, utile en inférence

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
                "activity_trend": activity_trend,
                "amp_max": amp_max,
                "amp_mean": amp_mean,
                "amp_trend_global": amp_trend_global,
                "amp_trend_recent": amp_trend_recent,
                "dist_min": dist_min,
                "dist_mean": dist_mean,
                "dist_recent_min": dist_recent_min,
                "dist_trend_global": dist_trend_global,
                "dist_trend_recent": dist_trend_recent,
                "elapsed_time": elapsed_time,
                # nouvelles
                "n_bursts": burst_feats["n_bursts"],
                "activity_variance": burst_feats["activity_variance"],
                "pause_max": burst_feats["pause_max"],
                "intensity_persistence": burst_feats["intensity_persistence"],
                "pause_ratio": pause_ratio,
                # ── Inactives ────────────────────────────────────────────────────
                "n_ic_total": n_ic_total,
                "ratio_ic_cg": ratio_ic_cg,
                "n_ic_recent": n_ic_recent,
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
    print("\nAperçu des nouvelles features :")
    new_cols = [
        "duration",
        "n_bursts",
        "activity_variance",
        "pause_max",
        "intensity_persistence",
    ]
    print(features[new_cols].describe().round(3))

    out = "data/features.parquet"
    features.to_parquet(out, index=False)
    print(f"\nSauvegardé dans {out}")
