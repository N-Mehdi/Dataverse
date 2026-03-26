import argparse
import math
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Simple capacity and cost planning from measured inference metrics.")
    parser.add_argument("--impact-csv", default="impact_enriched.csv")
    parser.add_argument("--peak-alerts-per-hour", type=float, required=True)
    parser.add_argument("--utilization-target", type=float, default=0.7)
    parser.add_argument("--cloud-instance-hourly-eur", type=float, default=None)
    parser.add_argument("--annual-maintenance-eur", type=float, default=0.0)
    parser.add_argument("--output-csv", default="deployment_capacity_scenario.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.impact_csv)
    one = df.loc[df["run_name"] == "infer_1_alert"]
    if one.empty:
        raise SystemExit("No row named 'infer_1_alert' found in the impact file.")

    row = one.iloc[0]
    sec_per_alert = float(row["elapsed_seconds"])
    energy_per_alert = float(row.get("energy_kwh", math.nan))
    cost_electricity_per_alert = float(row.get("electricity_cost_eur", math.nan))

    peak_alerts_per_second = args.peak_alerts_per_hour / 3600.0
    instances_needed = math.ceil((sec_per_alert * peak_alerts_per_second) / max(args.utilization_target, 1e-9))
    instances_needed = max(instances_needed, 1)

    annual_alerts_at_peak_24_7 = args.peak_alerts_per_hour * 24 * 365

    out = {
        "sec_per_alert": sec_per_alert,
        "energy_kwh_per_alert": energy_per_alert,
        "electricity_cost_eur_per_alert": cost_electricity_per_alert,
        "peak_alerts_per_hour": args.peak_alerts_per_hour,
        "utilization_target": args.utilization_target,
        "instances_needed": instances_needed,
        "annual_alerts_if_peak_24_7": annual_alerts_at_peak_24_7,
        "annual_electricity_cost_eur_if_peak_24_7": cost_electricity_per_alert * annual_alerts_at_peak_24_7 if not math.isnan(cost_electricity_per_alert) else math.nan,
        "annual_maintenance_eur": args.annual_maintenance_eur,
    }

    if args.cloud_instance_hourly_eur is not None:
        out["cloud_instance_hourly_eur"] = args.cloud_instance_hourly_eur
        out["annual_cloud_compute_eur_if_peak_24_7"] = instances_needed * args.cloud_instance_hourly_eur * 24 * 365

    pd.DataFrame([out]).to_csv(args.output_csv, index=False)
    print(pd.DataFrame([out]))
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
