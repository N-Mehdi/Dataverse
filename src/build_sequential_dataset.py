# build_sequential_dataset.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd

INNER_RADIUS_KM = 20.0
PRE_ALERT_WINDOW_MIN = 10
SILENCE_GRID_MIN = 1
MAX_GRID_GAP_MIN = 30
ROLLING_WINDOWS_MIN = [5, 10, 20]


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    df["type"] = np.where(df["icloud"].fillna(False), "IC", "CG")
    df["zone"] = np.where(df["dist"] < INNER_RADIUS_KM, "inner", "outer")

    df = df.sort_values(["airport", "date"]).reset_index(drop=True)

    # On garde une version propre des ids d'alerte
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


def build_decision_times(
    alert_events: pd.DataFrame,
    alert_start: pd.Timestamp,
    silence_grid_min: int = SILENCE_GRID_MIN,
    max_grid_gap_min: int = MAX_GRID_GAP_MIN,
) -> pd.DataFrame:
    """
    Instants de décision supervisés :
    - au début officiel de l'alerte (premier CG inner)
    - à chaque événement de l'alerte
    - points intermédiaires réguliers pendant les silences après le début de l'alerte

    On ne crée pas de lignes supervisées avant alert_start,
    mais les features pourront utiliser un historique qui commence avant.
    """
    alert_events = alert_events.sort_values("date").reset_index(drop=True)
    rows = []

    # point initial au début officiel de l'alerte
    rows.append({
        "decision_time": alert_start,
        "is_event_time": 1,
        "event_index": 0,
    })

    # temps des événements de l'alerte
    for i, t in enumerate(alert_events["date"]):
        rows.append({
            "decision_time": t,
            "is_event_time": 1,
            "event_index": i,
        })

    # grille régulière dans les silences après le début de l'alerte
    for i in range(1, len(alert_events)):
        prev_t = alert_events.loc[i - 1, "date"]
        t = alert_events.loc[i, "date"]

        # on ne densifie qu'après le début officiel
        if prev_t < alert_start:
            prev_t = alert_start
        if t <= prev_t:
            continue

        gap_min = minutes_between(t, prev_t)

        if silence_grid_min is not None and gap_min > silence_grid_min:
            max_gap = min(gap_min, max_grid_gap_min)
            n_steps = int(max_gap // silence_grid_min)

            for k in range(1, n_steps):
                grid_t = prev_t + pd.Timedelta(minutes=k * silence_grid_min)
                if grid_t < t:
                    rows.append({
                        "decision_time": grid_t,
                        "is_event_time": 0,
                        "event_index": i - 1,
                    })

    out = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["decision_time"])
        .sort_values("decision_time")
        .reset_index(drop=True)
    )

    return out


def compute_label(alert_cg_inner: pd.DataFrame, t: pd.Timestamp) -> int:
    """
    y = 1 s'il n'y a plus aucun CG inner de l'alerte après t, sinon 0
    """
    future_cg_inner = alert_cg_inner[alert_cg_inner["date"] > t]
    return int(len(future_cg_inner) == 0)


def compute_features_at_t(
    airport_df: pd.DataFrame,
    t: pd.Timestamp,
    obs_start: pd.Timestamp,
    alert_start: pd.Timestamp,
    is_event_time: int,
) -> dict:
    """
    Features calculées uniquement à partir des données disponibles entre
    obs_start et t, sur l'ensemble de l'aéroport.

    Pas de fuite :
    on utilise seulement date <= t.
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
        max_interarrival_min = float(interarrivals.max())
    else:
        mean_interarrival_min = 0.0
        max_interarrival_min = 0.0

    feats = {
        "elapsed_min": round(elapsed_min, 3),
        "obs_elapsed_min": round(obs_elapsed_min, 3),
        "is_event_time": int(is_event_time),
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
        "max_interarrival_min": round(max_interarrival_min, 3),
        "current_silence_over_mean_interarrival": round(
            safe_ratio(time_since_last_event_min, mean_interarrival_min), 3
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


def build_sequential_dataset(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    # alertes définies par la présence d'un airport_alert_id
    alert_rows = df[df["airport_alert_id"].notna()].copy()

    grouped_alerts = alert_rows.groupby(["airport", "airport_alert_id"], sort=False)
    n_alerts = grouped_alerts.ngroups
    print(f"Nombre d'alertes à traiter : {n_alerts}")

    # groupement par aéroport pour récupérer tout l'historique local
    airport_groups = {airport: g.sort_values("date").reset_index(drop=True)
                      for airport, g in df.groupby("airport", sort=False)}

    for idx, ((airport, alert_id), alert_df) in enumerate(grouped_alerts, start=1):
        if idx % 100 == 0:
            print(f"  {idx}/{n_alerts}")

        alert_df = alert_df.sort_values("date").reset_index(drop=True)

        # Le début officiel de l'alerte = premier CG inner de l'alerte
        alert_cg_inner = alert_df[
            (alert_df["type"] == "CG") & (alert_df["zone"] == "inner")
        ].sort_values("date").reset_index(drop=True)

        if len(alert_cg_inner) == 0:
            continue

        alert_start = alert_cg_inner["date"].iloc[0]
        obs_start = alert_start - pd.Timedelta(minutes=PRE_ALERT_WINDOW_MIN)

        airport_df = airport_groups[airport]

        # On garde l'historique aéroport à partir de obs_start
        airport_hist_for_alert = airport_df[airport_df["date"] >= obs_start].copy()

        decision_times = build_decision_times(
            alert_events=alert_df,
            alert_start=alert_start,
            silence_grid_min=SILENCE_GRID_MIN,
            max_grid_gap_min=MAX_GRID_GAP_MIN,
        )

        for _, row in decision_times.iterrows():
            t = row["decision_time"]

            # sécurité : pas de ligne supervisée avant le début officiel
            if t < alert_start:
                continue

            is_event_time = int(row["is_event_time"])
            event_index = int(row["event_index"])

            feats = compute_features_at_t(
                airport_df=airport_hist_for_alert,
                t=t,
                obs_start=obs_start,
                alert_start=alert_start,
                is_event_time=is_event_time,
            )
            y = compute_label(alert_cg_inner, t)

            feats["airport"] = airport
            feats["airport_alert_id"] = str(alert_id)
            feats["alert_group"] = f"{airport}__{alert_id}"
            feats["alert_start"] = alert_start
            feats["obs_start"] = obs_start
            feats["decision_time"] = t
            feats["event_index"] = event_index
            feats["y"] = y

            records.append(feats)

    out = pd.DataFrame(records)

    first_cols = [
        "airport",
        "airport_alert_id",
        "alert_group",
        "obs_start",
        "alert_start",
        "decision_time",
        "event_index",
    ]
    other_cols = [c for c in out.columns if c not in first_cols + ["y"]]
    out = out[first_cols + other_cols + ["y"]]

    out = out.sort_values(
        ["airport", "airport_alert_id", "decision_time"]
    ).reset_index(drop=True)

    return out


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/segment_alerts_all_airports_train.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/sequential_dataset.csv"

    print(f"Chargement : {input_path}")
    df = load_raw_data(input_path)

    print("Construction du dataset séquentiel...")
    seq_df = build_sequential_dataset(df)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_csv(output, index=False)

    print("\nRésumé")
    print("-" * 60)
    print(f"Nb lignes : {len(seq_df)}")
    print(f"Nb alertes : {seq_df['alert_group'].nunique()}")
    print(f"Nb aéroports : {seq_df['airport'].nunique()}")
    print("\nRépartition de y :")
    print(seq_df["y"].value_counts(dropna=False))
    print(f"\nFichier sauvegardé : {output}")


if __name__ == "__main__":
    main()