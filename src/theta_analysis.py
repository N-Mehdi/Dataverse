import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAX_GAP_MINUTES = 30
MIN_DIST_KM = 3.0
ACCEPTABLE_RISK = 0.02
DEFAULT_THETA = 0.906136


def load_raw_alerts(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["airport_alert_id"] = df["airport_alert_id"].astype("string")
    return df


def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["prediction_date"] = pd.to_datetime(df["prediction_date"], utc=True)
    df["predicted_date_end_alert"] = pd.to_datetime(
        df["predicted_date_end_alert"], utc=True
    )
    df["airport_alert_id"] = df["airport_alert_id"].astype("string")
    return df


def prepare_test_subset(raw_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    alert_keys = pred_df[["airport", "airport_alert_id"]].drop_duplicates()
    test_raw = raw_df.merge(
        alert_keys, on=["airport", "airport_alert_id"], how="inner"
    ).copy()
    return test_raw


def compute_theta_results(
    raw_df: pd.DataFrame, pred_df: pd.DataFrame, thetas: np.ndarray
) -> pd.DataFrame:
    alerts = raw_df.groupby(["airport", "airport_alert_id"], sort=False)
    tot_lightnings = int((raw_df["dist"] < MIN_DIST_KM).sum())

    rows = []
    for theta in thetas:
        pred_over_theta = pred_df.loc[pred_df["confidence"] >= theta].copy()

        if len(pred_over_theta) == 0:
            rows.append(
                {
                    "theta": float(theta),
                    "gain_seconds": 0.0,
                    "gain_hours": 0.0,
                    "missed_lightnings": 0,
                    "risk": 0.0,
                    "selected_alerts": 0,
                }
            )
            continue

        pred_over_theta_min = (
            pred_over_theta.groupby(["airport", "airport_alert_id"], sort=False)[
                "predicted_date_end_alert"
            ].min()
        )

        gain = 0.0
        missed_lights = 0

        for (airport, alert_id), end_alert_pred in pred_over_theta_min.items():
            lightings = alerts.get_group((airport, alert_id))
            end_alert_baseline = lightings["date"].max() + pd.Timedelta(
                minutes=MAX_GAP_MINUTES
            )
            gain += (end_alert_baseline - end_alert_pred).total_seconds()

            missed_lights += int(
                (
                    lightings.loc[lightings["dist"] < MIN_DIST_KM, "date"]
                    > end_alert_pred
                ).sum()
            )

        risk = missed_lights / tot_lightnings if tot_lightnings > 0 else 0.0

        rows.append(
            {
                "theta": float(theta),
                "gain_seconds": float(gain),
                "gain_hours": float(gain / 3600.0),
                "missed_lightnings": int(missed_lights),
                "risk": float(risk),
                "selected_alerts": int(pred_over_theta_min.shape[0]),
            }
        )

    return pd.DataFrame(rows).sort_values("theta").reset_index(drop=True)


def choose_best_theta(
    results_df: pd.DataFrame, acceptable_risk: float = ACCEPTABLE_RISK
):
    admissible = results_df.loc[results_df["risk"] < acceptable_risk].copy()
    if len(admissible) == 0:
        return None
    return admissible.sort_values(
        ["gain_seconds", "theta"], ascending=[False, True]
    ).iloc[0]


def make_plots(results_df: pd.DataFrame, best_theta: float, out_dir: Path):
    plt.figure(figsize=(9, 6))
    plt.plot(results_df["theta"], results_df["gain_hours"], marker="o")
    plt.axvline(best_theta, linestyle="--")
    plt.xlabel("Theta")
    plt.ylabel("Gain (heures)")
    plt.title("Gain total en fonction de theta")
    plt.tight_layout()
    plt.savefig(out_dir / "gain_vs_theta.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(results_df["theta"], results_df["risk"], marker="o")
    plt.axhline(ACCEPTABLE_RISK, linestyle="--")
    plt.axvline(best_theta, linestyle="--")
    plt.xlabel("Theta")
    plt.ylabel("Risque")
    plt.title("Risque en fonction de theta")
    plt.tight_layout()
    plt.savefig(out_dir / "risk_vs_theta.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(results_df["risk"], results_df["gain_hours"], marker="o")
    plt.axvline(ACCEPTABLE_RISK, linestyle="--")
    best_row = results_df.loc[np.isclose(results_df["theta"], best_theta)].iloc[0]
    plt.scatter([best_row["risk"]], [best_row["gain_hours"]], s=80)
    plt.xlabel("Risque")
    plt.ylabel("Gain (heures)")
    plt.title("Compromis gain / risque selon theta")
    plt.tight_layout()
    plt.savefig(out_dir / "gain_vs_risk.png", dpi=150)
    plt.close()


def main():
    raw_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/segment_alerts_all_airports_train.csv"
    )
    pred_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "output/model_comparison_with_xgboost/test_predictions_long.csv"
    )
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("output/theta_analysis")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chargement brut : {raw_path}")
    raw_df = load_raw_alerts(raw_path)

    print(f"Chargement prédictions : {pred_path}")
    pred_df = load_predictions(pred_path)

    print("Restriction aux alertes présentes dans les prédictions...")
    test_raw = prepare_test_subset(raw_df, pred_df)

    grid = np.round(np.linspace(0.0, 1.0, 101), 3)
    unique_scores = np.sort(pred_df["confidence"].dropna().unique())
    thetas = np.unique(np.concatenate([grid, unique_scores, np.array([DEFAULT_THETA])]))

    print("Calcul du gain et du risque pour chaque theta...")
    results_df = compute_theta_results(test_raw, pred_df, thetas)
    results_df.to_csv(out_dir / "theta_results.csv", index=False)

    best_row = choose_best_theta(results_df, acceptable_risk=ACCEPTABLE_RISK)
    if best_row is None:
        raise ValueError("Aucun theta ne respecte la contrainte de risque.")

    best_theta = float(best_row["theta"])

    summary_df = pd.DataFrame(
        [
            {
                "best_theta": best_theta,
                "acceptable_risk": ACCEPTABLE_RISK,
                "risk": float(best_row["risk"]),
                "gain_hours": float(best_row["gain_hours"]),
                "gain_seconds": float(best_row["gain_seconds"]),
                "missed_lightnings": int(best_row["missed_lightnings"]),
                "selected_alerts": int(best_row["selected_alerts"]),
            }
        ]
    )
    summary_df.to_csv(out_dir / "best_theta_summary.csv", index=False)

    print("Création des graphiques...")
    make_plots(results_df, best_theta, out_dir)

    print("\nRésumé du meilleur theta")
    print("-" * 60)
    print(summary_df.to_string(index=False))
    print(f"\nFichiers sauvegardés dans : {out_dir}")
    print("- theta_results.csv")
    print("- best_theta_summary.csv")
    print("- gain_vs_theta.png")
    print("- risk_vs_theta.png")
    print("- gain_vs_risk.png")


if __name__ == "__main__":
    main()