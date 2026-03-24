# build_silence_dataset.py

import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

INNER_RADIUS_KM = 20.0
PRE_ALERT_WINDOW_MIN = 10
SILENCE_GRID_MIN = 1
MAX_SILENCE_MIN = 30
ROLLING_WINDOWS_MIN = [5, 10, 20]


def minutes_between(t1: pd.Timestamp, t0: pd.Timestamp) -> float:
    return (t1 - t0).total_seconds() / 60.0


def safe_ratio(a: float, b: float) -> float:
    if b is None or b == 0 or pd.isna(b):
        return 0.0
    return float(a / b)


def compute_label(alert_cg_inner: pd.DataFrame, t: pd.Timestamp) -> int:
    """y = 1 s'il n'y a plus aucun CG inner après t, sinon 0"""
    return int((alert_cg_inner["date"] > t).sum() == 0)


def build_silence_decision_times(alert_cg_inner: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dates = alert_cg_inner["date"].sort_values().reset_index(drop=True)

    for i, current_cg_time in enumerate(dates):
        start_t = current_cg_time + pd.Timedelta(minutes=SILENCE_GRID_MIN)

        if i < len(dates) - 1:
            next_cg_time = dates.iloc[i + 1]
            end_t = min(
                current_cg_time + pd.Timedelta(minutes=MAX_SILENCE_MIN),
                next_cg_time - pd.Timedelta(seconds=1),
            )
        else:
            end_t = current_cg_time + pd.Timedelta(minutes=MAX_SILENCE_MIN)

        if end_t < start_t:
            continue

        grid = pd.date_range(
            start=start_t,
            end=end_t,
            freq=f"{SILENCE_GRID_MIN}min",
            tz=current_cg_time.tz,
        )
        for k, t in enumerate(grid, start=1):
            rows.append(
                {
                    "decision_time": t,
                    "minutes_since_reference_cg": minutes_between(t, current_cg_time),
                    "cg_reference_index": i,
                }
            )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    return (
        out.drop_duplicates(subset=["decision_time"])
        .sort_values("decision_time")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Contexte pré-calculé par alerte — cœur de l'optimisation
#
# Au lieu de reconstruire un sous-DataFrame par filtrage booléen O(n)
# à chaque instant t, on précalcule une seule fois :
#   - les timestamps en int64 nanosecondes → searchsorted O(log n)
#   - les masques booléens par type/zone → slices O(1)
#   - les arrays amplitude/dist
# ---------------------------------------------------------------------------


@dataclass
class AlertContext:
    df: pd.DataFrame
    dates_ns: np.ndarray  # timestamps en nanosecondes int64
    start_idx: int  # premier indice >= obs_start
    is_cg: np.ndarray
    is_ic: np.ndarray
    is_inner: np.ndarray
    is_outer: np.ndarray
    is_cg_inner: np.ndarray
    is_cg_outer: np.ndarray
    is_ic_inner: np.ndarray
    is_ic_outer: np.ndarray
    amplitudes: np.ndarray
    dists: np.ndarray
    obs_start: pd.Timestamp
    alert_start: pd.Timestamp
    obs_elapsed_full: float


def build_alert_context(
    airport_hist: pd.DataFrame,
    obs_start: pd.Timestamp,
    alert_start: pd.Timestamp,
) -> AlertContext:
    # Cast en datetime64[ns] avant int64 pour garantir l'unité nanoseconde
    # (pyarrow peut stocker en microsecondes, ce qui décalerait searchsorted d'un facteur 1000)
    dates_ns = airport_hist["date"].values.astype("datetime64[ns]").astype("int64")
    # .value retourne directement les nanosecondes int64, sans warning timezone
    start_idx = int(np.searchsorted(dates_ns, obs_start.value, side="left"))

    types = airport_hist["type"].values
    zones = airport_hist["zone"].values
    is_cg = types == "CG"
    is_ic = types == "IC"
    is_inner = zones == "inner"
    is_outer = zones == "outer"

    return AlertContext(
        df=airport_hist,
        dates_ns=dates_ns,
        start_idx=start_idx,
        is_cg=is_cg,
        is_ic=is_ic,
        is_inner=is_inner,
        is_outer=is_outer,
        is_cg_inner=is_cg & is_inner,
        is_cg_outer=is_cg & is_outer,
        is_ic_inner=is_ic & is_inner,
        is_ic_outer=is_ic & is_outer,
        amplitudes=airport_hist["amplitude"].values.astype("float64"),
        dists=airport_hist["dist"].values.astype("float64"),
        obs_start=obs_start,
        alert_start=alert_start,
        obs_elapsed_full=minutes_between(airport_hist["date"].iloc[-1], obs_start)
        if len(airport_hist) > 0
        else 0.0,
    )


def _time_since_last_ns(dates_ns, mask, s, e, fallback_min):
    active = dates_ns[s:e][mask[s:e]]
    if len(active) == 0:
        return fallback_min
    return (dates_ns[e - 1] - active[-1]) / 6e10  # ns → minutes


def compute_features_at_t(ctx: AlertContext, t: pd.Timestamp) -> dict:
    # .value retourne les nanosecondes int64, sans warning timezone
    t_ns = t.value
    s = ctx.start_idx
    e = int(np.searchsorted(ctx.dates_ns, t_ns, side="right"))

    if e <= s:
        raise ValueError(f"Historique vide à l'instant t={t}")

    elapsed_min = minutes_between(t, ctx.alert_start)
    obs_elapsed_min = minutes_between(t, ctx.obs_start)

    # Comptages par type/zone — simples sommes sur des slices NumPy
    n_total = e - s
    n_cg = int(ctx.is_cg[s:e].sum())
    n_ic = int(ctx.is_ic[s:e].sum())
    n_inner = int(ctx.is_inner[s:e].sum())
    n_outer = int(ctx.is_outer[s:e].sum())
    n_cg_inner = int(ctx.is_cg_inner[s:e].sum())
    n_cg_outer = int(ctx.is_cg_outer[s:e].sum())
    n_ic_inner = int(ctx.is_ic_inner[s:e].sum())
    n_ic_outer = int(ctx.is_ic_outer[s:e].sum())

    # Amplitude et distance
    amp_abs = np.abs(ctx.amplitudes[s:e])
    valid_amp = ~np.isnan(amp_abs)
    amp_abs_mean = float(np.nanmean(amp_abs)) if valid_amp.any() else 0.0
    amp_abs_max = float(np.nanmax(amp_abs)) if valid_amp.any() else 0.0

    valid_dist = ~np.isnan(ctx.dists[s:e])
    dist_mean = float(np.nanmean(ctx.dists[s:e])) if valid_dist.any() else 0.0
    dist_min = float(np.nanmin(ctx.dists[s:e])) if valid_dist.any() else 0.0

    # Dernier événement
    last_idx = e - 1
    last_row = ctx.df.iloc[last_idx]
    last_amp = (
        float(ctx.amplitudes[last_idx])
        if not np.isnan(ctx.amplitudes[last_idx])
        else 0.0
    )
    last_dist = float(ctx.dists[last_idx]) if not np.isnan(ctx.dists[last_idx]) else 0.0

    # Temps depuis le dernier événement de chaque catégorie
    time_since_last_event_min = (t_ns - ctx.dates_ns[e - 1]) / 6e10
    time_since_last_cg_min = _time_since_last_ns(
        ctx.dates_ns, ctx.is_cg, s, e, obs_elapsed_min
    )
    time_since_last_ic_min = _time_since_last_ns(
        ctx.dates_ns, ctx.is_ic, s, e, obs_elapsed_min
    )
    time_since_last_inner_min = _time_since_last_ns(
        ctx.dates_ns, ctx.is_inner, s, e, obs_elapsed_min
    )
    time_since_last_cg_inner_min = _time_since_last_ns(
        ctx.dates_ns, ctx.is_cg_inner, s, e, obs_elapsed_min
    )

    # Inter-arrivées
    if n_total >= 2:
        diffs = np.diff(ctx.dates_ns[s:e]) / 6e10
        mean_interarrival_min = float(np.mean(diffs))
        median_interarrival_min = float(np.median(diffs))
        max_interarrival_min = float(np.max(diffs))
    else:
        mean_interarrival_min = median_interarrival_min = max_interarrival_min = 0.0

    feats = {
        "elapsed_min": round(elapsed_min, 3),
        "obs_elapsed_min": round(obs_elapsed_min, 3),
        "last_event_type": last_row["type"],
        "last_event_zone": last_row["zone"],
        "last_event_amplitude": last_amp,
        "last_event_dist": last_dist,
        "n_total": n_total,
        "n_cg": n_cg,
        "n_ic": n_ic,
        "n_inner": n_inner,
        "n_outer": n_outer,
        "n_cg_inner": n_cg_inner,
        "n_cg_outer": n_cg_outer,
        "n_ic_inner": n_ic_inner,
        "n_ic_outer": n_ic_outer,
        "amp_abs_mean": amp_abs_mean,
        "amp_abs_max": amp_abs_max,
        "dist_mean": dist_mean,
        "dist_min": dist_min,
        "time_since_last_event_min": round(time_since_last_event_min, 3),
        "time_since_last_cg_min": round(time_since_last_cg_min, 3),
        "time_since_last_ic_min": round(time_since_last_ic_min, 3),
        "time_since_last_inner_min": round(time_since_last_inner_min, 3),
        "time_since_last_cg_inner_min": round(time_since_last_cg_inner_min, 3),
        "mean_interarrival_min": round(mean_interarrival_min, 3),
        "median_interarrival_min": round(median_interarrival_min, 3),
        "max_interarrival_min": round(max_interarrival_min, 3),
        "current_silence_over_mean_interarrival": round(
            safe_ratio(time_since_last_event_min, mean_interarrival_min), 3
        ),
        "current_silence_over_median_interarrival": round(
            safe_ratio(time_since_last_event_min, median_interarrival_min), 3
        ),
        "current_silence_over_max_interarrival": round(
            safe_ratio(time_since_last_event_min, max_interarrival_min), 3
        ),
    }

    # Fenêtres glissantes — searchsorted sur la borne inférieure, puis slice
    for w in ROLLING_WINDOWS_MIN:
        w_ns = w * 60 * int(1e9)
        sw = int(np.searchsorted(ctx.dates_ns, t_ns - w_ns, side="right"))
        sw = max(sw, s)

        feats[f"n_total_last_{w}m"] = e - sw
        feats[f"n_cg_inner_last_{w}m"] = int(ctx.is_cg_inner[sw:e].sum())
        feats[f"n_cg_outer_last_{w}m"] = int(ctx.is_cg_outer[sw:e].sum())
        feats[f"n_ic_inner_last_{w}m"] = int(ctx.is_ic_inner[sw:e].sum())
        feats[f"n_ic_outer_last_{w}m"] = int(ctx.is_ic_outer[sw:e].sum())

        d_w = ctx.dists[sw:e]
        valid_d_w = ~np.isnan(d_w)
        feats[f"dist_mean_last_{w}m"] = (
            float(np.nanmean(d_w)) if valid_d_w.any() else 0.0
        )
        feats[f"dist_min_last_{w}m"] = float(np.nanmin(d_w)) if valid_d_w.any() else 0.0

        a_w = np.abs(ctx.amplitudes[sw:e])
        feats[f"amp_abs_mean_last_{w}m"] = float(np.nanmean(a_w)) if (e > sw) else 0.0

    return feats


def build_silence_dataset(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    alert_rows = df[df["airport_alert_id"].notna()].copy()
    grouped_alerts = alert_rows.groupby(["airport", "airport_alert_id"], sort=False)
    n_alerts = grouped_alerts.ngroups
    print(f"Nombre d'alertes à traiter : {n_alerts}")

    airport_groups = {
        airport: g.sort_values("date").reset_index(drop=True)
        for airport, g in df.groupby("airport", sort=False)
    }

    for idx, ((airport, alert_id), alert_df) in enumerate(grouped_alerts, start=1):
        if idx % 100 == 0:
            print(f"  {idx}/{n_alerts}")

        alert_df = alert_df.sort_values("date").reset_index(drop=True)
        alert_cg_inner = (
            alert_df[(alert_df["type"] == "CG") & (alert_df["zone"] == "inner")]
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(alert_cg_inner) == 0:
            continue

        alert_start = alert_cg_inner["date"].iloc[0]
        obs_start = alert_start - pd.Timedelta(minutes=PRE_ALERT_WINDOW_MIN)

        airport_hist = airport_groups[airport]
        airport_hist_for_alert = airport_hist[airport_hist["date"] >= obs_start].copy()

        if len(airport_hist_for_alert) == 0:
            continue

        # Précalcul une seule fois par alerte
        ctx = build_alert_context(airport_hist_for_alert, obs_start, alert_start)

        decision_times = build_silence_decision_times(alert_cg_inner)
        if len(decision_times) == 0:
            continue

        # itertuples ~3× plus rapide qu'iterrows
        for row in decision_times.itertuples(index=False):
            t = row.decision_time
            cg_reference_index = int(row.cg_reference_index)
            minutes_since_reference_cg = float(row.minutes_since_reference_cg)

            try:
                feats = compute_features_at_t(ctx, t)
            except ValueError:
                continue

            y = compute_label(alert_cg_inner, t)

            feats["airport"] = airport
            feats["airport_alert_id"] = str(alert_id)
            feats["alert_group"] = f"{airport}__{alert_id}"
            feats["obs_start"] = obs_start
            feats["alert_start"] = alert_start
            feats["decision_time"] = t
            feats["cg_reference_index"] = cg_reference_index
            feats["minutes_since_reference_cg"] = round(minutes_since_reference_cg, 3)
            feats["y"] = y

            records.append(feats)

    out = pd.DataFrame(records)
    if len(out) == 0:
        return out

    first_cols = [
        "airport",
        "airport_alert_id",
        "alert_group",
        "obs_start",
        "alert_start",
        "decision_time",
        "cg_reference_index",
        "minutes_since_reference_cg",
    ]
    other_cols = [c for c in out.columns if c not in first_cols + ["y"]]
    out = out[first_cols + other_cols + ["y"]]
    return out.sort_values(
        ["airport", "airport_alert_id", "decision_time"]
    ).reset_index(drop=True)


def main():
    input_path = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/segment_alerts_all_airports_train.csv"
    )
    output_path = Path(
        sys.argv[2] if len(sys.argv) > 2 else "output/silence_dataset.parquet"
    )

    # Conversion automatique CSV → parquet au premier lancement
    if input_path.suffix == ".csv":
        parquet_input = input_path.with_suffix(".parquet")
        if not parquet_input.exists():
            print(f"Conversion CSV → parquet : {input_path} → {parquet_input}")
            tmp = pd.read_csv(input_path)
            tmp["date"] = pd.to_datetime(tmp["date"], utc=True)
            parquet_input.parent.mkdir(parents=True, exist_ok=True)
            tmp.to_parquet(
                parquet_input, index=False, engine="pyarrow", compression="snappy"
            )
            print(
                f"  ({input_path.stat().st_size // 1024} Ko → {parquet_input.stat().st_size // 1024} Ko)"
            )
            del tmp
        else:
            print(f"Parquet déjà existant, skip conversion : {parquet_input}")
        input_path = parquet_input

    print(f"Chargement : {input_path}")
    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["type"] = np.where(df["icloud"].fillna(False), "IC", "CG")
    df["zone"] = np.where(df["dist"] < INNER_RADIUS_KM, "inner", "outer")
    df = df.sort_values(["airport", "date"]).reset_index(drop=True)
    df["airport_alert_id"] = df["airport_alert_id"].astype("string")

    print("Construction du dataset de silences décisionnels...")
    silence_df = build_silence_dataset(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    silence_df.to_parquet(
        output_path, index=False, engine="pyarrow", compression="snappy"
    )

    print("\nRésumé")
    print("-" * 60)
    print(f"Nb lignes : {len(silence_df)}")
    print(f"Nb alertes : {silence_df['alert_group'].nunique()}")
    print(f"Nb aéroports : {silence_df['airport'].nunique()}")
    print("\nRépartition de y :")
    print(silence_df["y"].value_counts(dropna=False))
    print(f"\nFichier sauvegardé : {output_path}")


if __name__ == "__main__":
    main()
