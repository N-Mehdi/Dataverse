import json
from pathlib import Path

import pandas as pd


def latest_codecarbon_row(csv_path: Path):
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}
    row = df.iloc[-1].to_dict()
    return {f"cc_{k}": v for k, v in row.items()}


def extract_energy_kwh(codecarbon_row: dict):
    candidates = [
        "cc_energy_consumed",
        "cc_energy_consumed_kwh",
        "cc_energy_kwh",
    ]
    for key in candidates:
        if key in codecarbon_row and pd.notna(codecarbon_row[key]):
            return float(codecarbon_row[key])
    return None


def main(folder="impact_runs", output_csv="impact_summary.csv"):
    rows = []
    folder_path = Path(folder)
    for meta_path in folder_path.glob("*_meta.json"):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cc_path = folder_path / f"{meta['run_name']}_codecarbon.csv"
        cc_row = latest_codecarbon_row(cc_path)
        energy_kwh = extract_energy_kwh(cc_row)
        row = {**meta, **cc_row}
        row["energy_kwh"] = energy_kwh
        row["elapsed_hours"] = row["elapsed_seconds"] / 3600 if row.get("elapsed_seconds") is not None else None
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        preferred_cols = [
            "run_name",
            "status",
            "elapsed_seconds",
            "elapsed_hours",
            "energy_kwh",
            "emissions_kg_co2eq",
            "max_rss_gb",
            "avg_cpu_percent",
            "max_cpu_percent",
            "process_read_gb",
            "process_write_gb",
            "cpu_count_logical",
            "cpu_count_physical",
            "gpus",
            "notes",
        ]
        ordered = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
        df = df[ordered]
    df.to_csv(output_csv, index=False)
    print(df)
    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()
