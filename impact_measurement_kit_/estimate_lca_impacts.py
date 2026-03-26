import argparse
import math
from pathlib import Path

import pandas as pd
import yaml


def safe_div(a, b):
    return a / b if b else math.nan


def main():
    parser = argparse.ArgumentParser(description="Convert measured energy into water / abiotic / cost estimates using documented factors.")
    parser.add_argument("--summary-csv", default="impact_summary.csv")
    parser.add_argument("--config", default="lca_factors_example.yaml")
    parser.add_argument("--output-csv", default="impact_enriched.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    electricity_price_eur_per_kwh = cfg["electricity"]["price_eur_per_kwh"]
    water_l_per_kwh = cfg["electricity"].get("water_l_per_kwh")
    adpe_kg_sbeq_per_kwh = cfg["electricity"].get("adpe_kg_sbeq_per_kwh")
    pe_mj_per_kwh = cfg["electricity"].get("primary_energy_mj_per_kwh")

    hw = cfg["hardware"]
    hw_lifetime_hours = hw["lifetime_hours"]

    embodied = hw.get("embodied", {})
    embodied_gwp = embodied.get("gwp_kg_co2eq")
    embodied_water = embodied.get("water_l")
    embodied_adpe = embodied.get("adpe_kg_sbeq")
    embodied_pe = embodied.get("primary_energy_mj")

    df["electricity_cost_eur"] = df["energy_kwh"] * electricity_price_eur_per_kwh
    if water_l_per_kwh is not None:
        df["water_use_phase_l"] = df["energy_kwh"] * water_l_per_kwh
    if adpe_kg_sbeq_per_kwh is not None:
        df["adpe_use_phase_kg_sbeq"] = df["energy_kwh"] * adpe_kg_sbeq_per_kwh
    if pe_mj_per_kwh is not None:
        df["primary_energy_use_phase_mj"] = df["energy_kwh"] * pe_mj_per_kwh

    # Allocate embodied impacts by actual runtime share.
    df["runtime_share_of_device_life"] = df["elapsed_hours"] / hw_lifetime_hours
    if embodied_gwp is not None:
        df["embodied_gwp_alloc_kg_co2eq"] = df["runtime_share_of_device_life"] * embodied_gwp
    if embodied_water is not None:
        df["embodied_water_alloc_l"] = df["runtime_share_of_device_life"] * embodied_water
    if embodied_adpe is not None:
        df["embodied_adpe_alloc_kg_sbeq"] = df["runtime_share_of_device_life"] * embodied_adpe
    if embodied_pe is not None:
        df["embodied_primary_energy_alloc_mj"] = df["runtime_share_of_device_life"] * embodied_pe

    # Totals when the corresponding components exist.
    if {"water_use_phase_l", "embodied_water_alloc_l"}.issubset(df.columns):
        df["water_total_l"] = df["water_use_phase_l"] + df["embodied_water_alloc_l"]
    if {"adpe_use_phase_kg_sbeq", "embodied_adpe_alloc_kg_sbeq"}.issubset(df.columns):
        df["adpe_total_kg_sbeq"] = df["adpe_use_phase_kg_sbeq"] + df["embodied_adpe_alloc_kg_sbeq"]
    if {"primary_energy_use_phase_mj", "embodied_primary_energy_alloc_mj"}.issubset(df.columns):
        df["primary_energy_total_mj"] = df["primary_energy_use_phase_mj"] + df["embodied_primary_energy_alloc_mj"]

    # Optional production scenario.
    prod = cfg.get("production", {})
    annual_alerts = prod.get("annual_alerts")
    if annual_alerts and "infer_1_alert" in df["run_name"].values:
        row = df.loc[df["run_name"] == "infer_1_alert"].iloc[0]
        annual = {
            "annual_alerts": annual_alerts,
            "annual_energy_kwh": row.get("energy_kwh", math.nan) * annual_alerts,
            "annual_cost_eur": row.get("electricity_cost_eur", math.nan) * annual_alerts,
        }
        if "water_total_l" in row.index:
            annual["annual_water_l"] = row.get("water_total_l", math.nan) * annual_alerts
        if "adpe_total_kg_sbeq" in row.index:
            annual["annual_adpe_kg_sbeq"] = row.get("adpe_total_kg_sbeq", math.nan) * annual_alerts
        annual_df = pd.DataFrame([annual])
        annual_path = Path(args.output_csv).with_name("impact_annual_scenario.csv")
        annual_df.to_csv(annual_path, index=False)
        print(f"Saved: {annual_path}")

    df.to_csv(args.output_csv, index=False)
    print(df)
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
