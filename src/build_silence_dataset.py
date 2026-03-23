# build_silence_dataset.py

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INNER_RADIUS_KM = 20.0
PRE_ALERT_WINDOW_MIN = 10
SILENCE_GRID_MIN = 1
MAX_SILENCE_MIN = 30
ROLLING_WINDOWS_MIN = [5, 10, 20]

# True  -> features calculées à partir du contexte aéroport depuis obs_start
# False -> features calculées uniquement à partir de l'alerte courante depuis obs_start
USE_AIRPORT_CONTEXT = True


# ============================================================
# Chargement / utilitaires de base
# ============================================================

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
    if b is None or pd.isna(b) or b <= 0:
        return np.nan
    return float(a / b)


def _count_window(events: pd.DataFrame, t: pd.Timestamp, minutes: int) -> pd.DataFrame:
    start = t - pd.Timedelta(minutes=minutes)
    return events[(events["date"] > start) & (events["date"] <= t)]


# ============================================================
# Labels
# ============================================================

def compute_label_terminal(alert_cg_inner: pd.DataFrame, t: pd.Timestamp) -> int:
    """
    1 s'il n'y a plus aucun CG inner après t, sinon 0.
    """
    future_cg_inner = alert_cg_inner[alert_cg_inner["date"] > t]
    return int(len(future_cg_inner) == 0)


def compute_label_horizon(
    alert_cg_inner: pd.DataFrame,
    t: pd.Timestamp,
    horizon_min: int,
) -> int:
    """
    1 s'il n'y a aucun CG inner dans les horizon_min prochaines minutes, sinon 0.
    """
    horizon_end = t + pd.Timedelta(minutes=horizon_min)
    future_cg_inner = alert_cg_inner[
        (alert_cg_inner["date"] > t) & (alert_cg_inner["date"] <= horizon_end)
    ]
    return int(len(future_cg_inner) == 0)


# ============================================================
# Instants de décision
# ============================================================

def build_silence_decision_times(alert_cg_inner: pd.DataFrame) -> pd.DataFrame:
    """
    Construit des instants de décision uniquement pendant les silences
    qui suivent chaque CG inner.

    Pour chaque CG inner à la date t_i :
    - on commence à t_i + SILENCE_GRID_MIN
    - on avance minute par minute
    - on s'arrête au prochain CG inner (exclu)
    - ou à MAX_SILENCE_MIN si c'est le dernier segment observé
    """
    rows: List[Dict] = []

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

    out = (
        out.drop_duplicates(subset=["decision_time"])
        .sort_values("decision_time")
        .reset_index(drop=True)
    )
    return out


# ============================================================
# Features
# ============================================================

def compute_features_at_t(
    hist_source: pd.DataFrame,
    t: pd.Timestamp,
    obs_start: pd.Timestamp,
    alert_start: pd.Timestamp,
) -> Dict[str, float]:
    """
    Features calculées uniquement à partir des données disponibles
    entre obs_start et t.
    """
    hist = hist_source[(hist_source["date"] >= obs_start) & (hist_source["date"] <= t)].copy()
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
        mean_interarrival_min = np.nan
        median_interarrival_min = np.nan
        max_interarrival_min = np.nan

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
        "mean_interarrival_min": round(mean_interarrival_min, 3) if pd.notna(mean_interarrival_min) else np.nan,
        "median_interarrival_min": round(median_interarrival_min, 3) if pd.notna(median_interarrival_min) else np.nan,
        "max_interarrival_min": round(max_interarrival_min, 3) if pd.notna(max_interarrival_min) else np.nan,
        "current_silence_over_mean_interarrival": round(
            safe_ratio(time_since_last_event_min, mean_interarrival_min), 3
        ) if pd.notna(safe_ratio(time_since_last_event_min, mean_interarrival_min)) else np.nan,
        "current_silence_over_median_interarrival": round(
            safe_ratio(time_since_last_event_min, median_interarrival_min), 3
        ) if pd.notna(safe_ratio(time_since_last_event_min, median_interarrival_min)) else np.nan,
        "current_silence_over_max_interarrival": round(
            safe_ratio(time_since_last_event_min, max_interarrival_min), 3
        ) if pd.notna(safe_ratio(time_since_last_event_min, max_interarrival_min)) else np.nan,
    }

    for w in ROLLING_WINDOWS_MIN:
        sub = _count_window(hist, t, w)
        feats[f"n_total_last_{w}m"] = int(len(sub))
        feats[f"n_cg_inner_last_{w}m"] = int(len(sub[(sub["type"] == "CG") & (sub["zone"] == "inner")]))
        feats[f"n_cg_outer_last_{w}m"] = int(len(sub[(sub["type"] == "CG") & (sub["zone"] == "outer")]))
        feats[f"n_ic_inner_last_{w}m"] = int(len(sub[(sub["type"] == "IC") & (sub["zone"] == "inner")]))
        feats[f"n_ic_outer_last_{w}m"] = int(len(sub[(sub["type"] == "IC") & (sub["zone"] == "outer")]))
        feats[f"dist_mean_last_{w}m"] = float(sub["dist"].mean()) if len(sub) > 0 else np.nan
        feats[f"dist_min_last_{w}m"] = float(sub["dist"].min()) if len(sub) > 0 else np.nan
        feats[f"amp_abs_mean_last_{w}m"] = float(sub["amplitude"].abs().mean()) if len(sub) > 0 else np.nan

    # Tendance activité / distance / amplitude
    if "n_cg_inner_last_5m" in feats and "n_cg_inner_last_10m" in feats:
        feats["cg_inner_trend_5_vs_10"] = (
            feats["n_cg_inner_last_5m"]
            - (feats["n_cg_inner_last_10m"] - feats["n_cg_inner_last_5m"])
        )

    if "dist_mean_last_5m" in feats and "dist_mean_last_20m" in feats:
        if pd.notna(feats["dist_mean_last_5m"]) and pd.notna(feats["dist_mean_last_20m"]):
            feats["dist_trend_5_vs_20"] = feats["dist_mean_last_5m"] - feats["dist_mean_last_20m"]
        else:
            feats["dist_trend_5_vs_20"] = np.nan

    if "amp_abs_mean_last_5m" in feats and "amp_abs_mean_last_20m" in feats:
        if pd.notna(feats["amp_abs_mean_last_5m"]) and pd.notna(feats["amp_abs_mean_last_20m"]):
            feats["amp_trend_5_vs_20"] = feats["amp_abs_mean_last_5m"] - feats["amp_abs_mean_last_20m"]
        else:
            feats["amp_trend_5_vs_20"] = np.nan

    return feats


# ============================================================
# Construction du dataset
# ============================================================

def build_silence_dataset(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict] = []

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

        if USE_AIRPORT_CONTEXT:
            hist_source = airport_df[airport_df["date"] >= obs_start].copy()
        else:
            hist_source = alert_df[alert_df["date"] >= obs_start].copy()

        decision_times = build_silence_decision_times(alert_cg_inner)

        if len(decision_times) == 0:
            continue

        for _, row in decision_times.iterrows():
            t = row["decision_time"]
            cg_reference_index = int(row["cg_reference_index"])
            minutes_since_reference_cg = float(row["minutes_since_reference_cg"])

            feats = compute_features_at_t(
                hist_source=hist_source,
                t=t,
                obs_start=obs_start,
                alert_start=alert_start,
            )

            y_terminal = compute_label_terminal(alert_cg_inner, t)
            y_h20 = compute_label_horizon(alert_cg_inner, t, horizon_min=20)
            y_h30 = compute_label_horizon(alert_cg_inner, t, horizon_min=30)

            record = {
                "airport": airport,
                "airport_alert_id": str(alert_id),
                "alert_group": f"{airport}__{alert_id}",
                "obs_start": obs_start,
                "alert_start": alert_start,
                "decision_time": t,
                "cg_reference_index": cg_reference_index,
                "minutes_since_reference_cg": round(minutes_since_reference_cg, 3),
                "use_airport_context": int(USE_AIRPORT_CONTEXT),
                "y_terminal": int(y_terminal),
                "y_h20": int(y_h20),
                "y_h30": int(y_h30),
            }
            record.update(feats)
            records.append(record)

    out = pd.DataFrame(records)

    if len(out) == 0:
        return out

    # Optionnel : suppression d'une redondance quasi directe
    redundant_cols = [
        # "time_since_last_cg_inner_min",
    ]
    out = out.drop(columns=[c for c in redundant_cols if c in out.columns])

    first_cols = [
        "airport",
        "airport_alert_id",
        "alert_group",
        "obs_start",
        "alert_start",
        "decision_time",
        "cg_reference_index",
        "minutes_since_reference_cg",
        "use_airport_context",
    ]
    target_cols = ["y_terminal", "y_h20", "y_h30"]
    other_cols = [c for c in out.columns if c not in first_cols + target_cols]

    out = out[first_cols + other_cols + target_cols]
    out = out.sort_values(["airport", "airport_alert_id", "decision_time"]).reset_index(drop=True)
    return out


# ============================================================
# Main
# ============================================================

def main():
    input_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/segment_alerts_all_airports_train.csv"
    )
    output_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "output/silence_dataset.csv"
    )

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
    print(f"Nb alertes : {silence_df['alert_group'].nunique() if len(silence_df) > 0 else 0}")
    print(f"Nb aéroports : {silence_df['airport'].nunique() if len(silence_df) > 0 else 0}")

    if len(silence_df) > 0:
        for target in ["y_terminal", "y_h20", "y_h30"]:
            print(f"\nRépartition de {target} :")
            print(silence_df[target].value_counts(dropna=False))

    print(f"\nFichier sauvegardé : {output}")


if __name__ == "__main__":
    main()