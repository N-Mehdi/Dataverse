# build_silence_dataset.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd

INNER_RADIUS_KM = 20.0
PRE_ALERT_WINDOW_MIN = 10
SILENCE_GRID_MIN = 1
MAX_SILENCE_MIN = 30
ROLLING_WINDOWS_MIN = [5, 10, 20]


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    df["type"] = np.where(df["icloud"].fillna(False), "IC", "CG")
    df["zone"] = np.where(df["dist"] < INNER_RADIUS_KM, "inner", "outer")

    df = df.sort_values(["airport", "date"]).reset_index(drop=True)
    df["airport_alert_id"] = df["airport_alert_id"].astype("string")
    return df


def minutes_between(t1: pd.Timestamp, t0: pd.Timestamp) -> float:
    return (t1 - t0).total_seconds() / 60.0


def safe_ratio(a: float, b: float) -> float:
    if b is None or b == 0 or pd.isna(b):
        return 0.0
    return float(a / b)


def _count_window(events: pd.DataFrame, t: pd.Timestamp, minutes: int) -> pd.DataFrame:
    start = t - pd.Timedelta(minutes=minutes)
    return events[(events["date"] > start) & (events["date"] <= t)]


def compute_label(alert_cg_inner: pd.DataFrame, t: pd.Timestamp) -> int:
    """
    y = 1 s'il n'y a plus aucun CG inner après t, sinon 0
    """
    future_cg_inner = alert_cg_inner[alert_cg_inner["date"] > t]
    return int(len(future_cg_inner) == 0)


def build_silence_decision_times(alert_cg_inner: pd.DataFrame) -> pd.DataFrame:
    """
    Construit uniquement des instants de décision dans les silences après chaque CG inner.
    
    Pour chaque CG inner à la date t_i :
    - on commence à t_i + 1 min
    - on avance minute par minute
    - on s'arrête au prochain CG inner exclus
    - ou à MAX_SILENCE_MIN si c'est le dernier segment observé

    Cela évite les points pendant l'activité pure.
    """
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

        grid = pd.date_range(start=start_t, end=end_t, freq=f"{SILENCE_GRID_MIN}min", tz=current_cg_time.tz)

        for k, t in enumerate(grid, start=1):
            rows.append({
                "decision_time": t,
                "minutes_since_reference_cg": minutes_between(t, current_cg_time),
                "cg_reference_index": i,
            })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    out = out.drop_duplicates(subset=["decision_time"]).sort_values("decision_time").reset_index(drop=True)
    return out


def compute_features_at_t(
    airport_df: pd.DataFrame,
    t: pd.Timestamp,
    obs_start: pd.Timestamp,
    alert_start: pd.Timestamp,
) -> dict:
    """
    Features calculées uniquement à partir des données disponibles entre obs_start et t.
    """
    hist = airport_df[(airport_df["date"] >= obs_start) & (airport_df["date"] <= t)].copy()
    hist = hist.sort_values("date").reset_index(drop=True)

    if len(hist) == 0:
        raise ValueError("Historique vide à l'instant t.")

    elapsed_min = minutes_between(t, alert_start)
    obs_elapsed_min = minutes_between(t, obs_start)

    cg = hist[hist["type"] == "CG"]
    ic = hist[hist["type"] == "IC"]
    inner = hist[hist["zone"] == "inner"]
    outer = hist[hist["zone"] == "outer"]

    cg_inner = hist[(hist["type"] == "CG") & (hist["zone"] == "inner")]
    cg_outer = hist[(hist["type"] == "CG") & (hist["zone"] == "outer")]
    ic_inner = hist[(hist["type"] == "IC") & (hist["zone"] == "inner")]
    ic_outer = hist[(hist["type"] == "IC") & (hist["zone"] == "outer")]

    last_event = hist.iloc[-1]

    def time_since_last(subdf: pd.DataFrame) -> float:
        if len(subdf) == 0:
            return obs_elapsed_min
        return minutes_between(t, subdf["date"].iloc[-1])

    time_since_last_event_min = minutes_between(t, hist["date"].iloc[-1])
    time_since_last_cg_min = time_since_last(cg)
    time_since_last_ic_min = time_since_last(ic)
    time_since_last_inner_min = time_since_last(inner)
    time_since_last_cg_inner_min = time_since_last(cg_inner)

    if len(hist) >= 2:
        interarrivals = hist["date"].diff().dropna().dt.total_seconds() / 60.0
        mean_interarrival_min = float(interarrivals.mean())
        median_interarrival_min = float(interarrivals.median())
        max_interarrival_min = float(interarrivals.max())
    else:
        mean_interarrival_min = 0.0
        median_interarrival_min = 0.0
        max_interarrival_min = 0.0

    feats = {
        "elapsed_min": round(elapsed_min, 3),
        "obs_elapsed_min": round(obs_elapsed_min, 3),
        "last_event_type": last_event["type"],
        "last_event_zone": last_event["zone"],
        "last_event_amplitude": float(last_event["amplitude"]) if pd.notna(last_event["amplitude"]) else 0.0,
        "last_event_dist": float(last_event["dist"]) if pd.notna(last_event["dist"]) else 0.0,
        "n_total": int(len(hist)),
        "n_cg": int(len(cg)),
        "n_ic": int(len(ic)),
        "n_inner": int(len(inner)),
        "n_outer": int(len(outer)),
        "n_cg_inner": int(len(cg_inner)),
        "n_cg_outer": int(len(cg_outer)),
        "n_ic_inner": int(len(ic_inner)),
        "n_ic_outer": int(len(ic_outer)),
        "amp_abs_mean": float(hist["amplitude"].abs().mean()) if hist["amplitude"].notna().any() else 0.0,
        "amp_abs_max": float(hist["amplitude"].abs().max()) if hist["amplitude"].notna().any() else 0.0,
        "dist_mean": float(hist["dist"].mean()) if hist["dist"].notna().any() else 0.0,
        "dist_min": float(hist["dist"].min()) if hist["dist"].notna().any() else 0.0,
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

    for w in ROLLING_WINDOWS_MIN:
        sub = _count_window(hist, t, w)
        feats[f"n_total_last_{w}m"] = int(len(sub))
        feats[f"n_cg_inner_last_{w}m"] = int(len(sub[(sub["type"] == "CG") & (sub["zone"] == "inner")]))
        feats[f"n_cg_outer_last_{w}m"] = int(len(sub[(sub["type"] == "CG") & (sub["zone"] == "outer")]))
        feats[f"n_ic_inner_last_{w}m"] = int(len(sub[(sub["type"] == "IC") & (sub["zone"] == "inner")]))
        feats[f"n_ic_outer_last_{w}m"] = int(len(sub[(sub["type"] == "IC") & (sub["zone"] == "outer")]))
        feats[f"dist_mean_last_{w}m"] = float(sub["dist"].mean()) if len(sub) > 0 else 0.0
        feats[f"dist_min_last_{w}m"] = float(sub["dist"].min()) if len(sub) > 0 else 0.0
        feats[f"amp_abs_mean_last_{w}m"] = float(sub["amplitude"].abs().mean()) if len(sub) > 0 else 0.0

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

        alert_cg_inner = alert_df[
            (alert_df["type"] == "CG") & (alert_df["zone"] == "inner")
        ].sort_values("date").reset_index(drop=True)

        if len(alert_cg_inner) == 0:
            continue

        alert_start = alert_cg_inner["date"].iloc[0]
        obs_start = alert_start - pd.Timedelta(minutes=PRE_ALERT_WINDOW_MIN)

        airport_df = airport_groups[airport]
        airport_hist_for_alert = airport_df[airport_df["date"] >= obs_start].copy()

        decision_times = build_silence_decision_times(alert_cg_inner)

        if len(decision_times) == 0:
            continue

        for _, row in decision_times.iterrows():
            t = row["decision_time"]
            cg_reference_index = int(row["cg_reference_index"])
            minutes_since_reference_cg = float(row["minutes_since_reference_cg"])

            feats = compute_features_at_t(
                airport_df=airport_hist_for_alert,
                t=t,
                obs_start=obs_start,
                alert_start=alert_start,
            )
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

    out = out.sort_values(["airport", "airport_alert_id", "decision_time"]).reset_index(drop=True)
    return out


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/segment_alerts_all_airports_train.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/silence_dataset.csv"

    print(f"Chargement : {input_path}")
    df = load_raw_data(input_path)

    print("Construction du dataset de silences décisionnels...")
    silence_df = build_silence_dataset(df)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    silence_df.to_csv(output, index=False)

    print("\nRésumé")
    print("-" * 60)
    print(f"Nb lignes : {len(silence_df)}")
    print(f"Nb alertes : {silence_df['alert_group'].nunique()}")
    print(f"Nb aéroports : {silence_df['airport'].nunique()}")
    print("\nRépartition de y :")
    print(silence_df["y"].value_counts(dropna=False))
    print(f"\nFichier sauvegardé : {output}")


if __name__ == "__main__":
    main()