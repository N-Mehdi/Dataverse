# make_dataset_variants.py

import sys
from pathlib import Path
import pandas as pd


ID_COLS = [
    "airport",
    "airport_alert_id",
    "alert_group",
    "obs_start",
    "alert_start",
    "decision_time",
    "cg_reference_index",
    "minutes_since_reference_cg",
]

TARGET_COL = "y"

TIME_SINCE_COLS = [
    "time_since_last_event_min",
    "time_since_last_cg_min",
    "time_since_last_ic_min",
    "time_since_last_inner_min",
    "time_since_last_cg_inner_min",
]

RELATIVE_SILENCE_COLS = [
    "mean_interarrival_min",
    "median_interarrival_min",
    "max_interarrival_min",
    "current_silence_over_mean_interarrival",
    "current_silence_over_median_interarrival",
    "current_silence_over_max_interarrival",
]

RECENT_ACTIVITY_COLS = [
    "n_total_last_5m",
    "n_cg_inner_last_5m",
    "n_cg_outer_last_5m",
    "n_ic_inner_last_5m",
    "n_ic_outer_last_5m",
    "dist_mean_last_5m",
    "dist_min_last_5m",
    "amp_abs_mean_last_5m",
    "n_total_last_10m",
    "n_cg_inner_last_10m",
    "n_cg_outer_last_10m",
    "n_ic_inner_last_10m",
    "n_ic_outer_last_10m",
    "dist_mean_last_10m",
    "dist_min_last_10m",
    "amp_abs_mean_last_10m",
    "n_total_last_20m",
    "n_cg_inner_last_20m",
    "n_cg_outer_last_20m",
    "n_ic_inner_last_20m",
    "n_ic_outer_last_20m",
    "dist_mean_last_20m",
    "dist_min_last_20m",
    "amp_abs_mean_last_20m",
]

CUMULATIVE_CONTEXT_COLS = [
    "elapsed_min",
    "obs_elapsed_min",
    "last_event_type",
    "last_event_zone",
    "last_event_amplitude",
    "last_event_dist",
    "n_total",
    "n_cg",
    "n_ic",
    "n_inner",
    "n_outer",
    "n_cg_inner",
    "n_cg_outer",
    "n_ic_inner",
    "n_ic_outer",
    "amp_abs_mean",
    "amp_abs_max",
    "dist_mean",
    "dist_min",
]


def keep_existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def build_version_b(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = keep_existing(df, TIME_SINCE_COLS)
    out = df.drop(columns=drop_cols).copy()
    return out


def build_version_c(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = (
        keep_existing(df, ID_COLS)
        + keep_existing(df, CUMULATIVE_CONTEXT_COLS)
        + keep_existing(df, RELATIVE_SILENCE_COLS)
        + keep_existing(df, RECENT_ACTIVITY_COLS)
        + [TARGET_COL]
    )

    # Enlève doublons éventuels tout en gardant l'ordre
    keep_cols = list(dict.fromkeys(keep_cols))
    out = df[keep_cols].copy()
    return out


def summarize(df: pd.DataFrame, name: str) -> None:
    print(f"\n{name}")
    print("-" * 60)
    print(f"Nb lignes : {len(df)}")
    print(f"Nb colonnes : {df.shape[1]}")
    print("Répartition y :")
    print(df[TARGET_COL].value_counts(dropna=False))


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "output/silence_dataset.csv"
    out_b = sys.argv[2] if len(sys.argv) > 2 else "output/silence_dataset_B.csv"
    out_c = sys.argv[3] if len(sys.argv) > 3 else "output/silence_dataset_C.csv"

    df = pd.read_csv(input_path, parse_dates=["obs_start", "alert_start", "decision_time"])

    b = build_version_b(df)
    c = build_version_c(df)

    Path(out_b).parent.mkdir(parents=True, exist_ok=True)
    b.to_csv(out_b, index=False)
    c.to_csv(out_c, index=False)

    summarize(df, "Dataset source")
    summarize(b, "Version B")
    summarize(c, "Version C")

    print(f"\nSauvegardé : {out_b}")
    print(f"Sauvegardé : {out_c}")


if __name__ == "__main__":
    main()