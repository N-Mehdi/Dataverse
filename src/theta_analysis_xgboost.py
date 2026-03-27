import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAX_GAP_MINUTES = 30
MIN_DIST_KM = 3.0
ACCEPTABLE_RISK = 0.02


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
    df["airport_alert_id"] = df["airport_alert_id"].astype("string")

    if "decision_time" in df.columns and "score" in df.columns:
        df = df.rename(columns={
            "decision_time": "prediction_date",
            "score":         "confidence",
        })
        df["predicted_date_end_alert"] = df["prediction_date"]

    df["prediction_date"] = pd.to_datetime(df["prediction_date"], utc=True)
    df["predicted_date_end_alert"] = pd.to_datetime(df["predicted_date_end_alert"], utc=True)
    return df


def prepare_test_subset(raw_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    alert_keys = pred_df[["airport", "airport_alert_id"]].drop_duplicates()
    return raw_df.merge(alert_keys, on=["airport", "airport_alert_id"], how="inner").copy()


def compute_theta_results(
    raw_df: pd.DataFrame, pred_df: pd.DataFrame, thetas: np.ndarray
) -> pd.DataFrame:
    tot_lightnings = int((raw_df["dist"] < MIN_DIST_KM).sum())

    alert_stats = []
    for (airport, alert_id), group in raw_df.groupby(["airport", "airport_alert_id"], sort=False):
        end_baseline = group["date"].max() + pd.Timedelta(minutes=MAX_GAP_MINUTES)
        lre_dates = pd.to_datetime(
            group.loc[group["dist"] < MIN_DIST_KM, "date"].values, utc=True
        )
        alert_stats.append({
            "airport": airport,
            "airport_alert_id": str(alert_id),
            "end_baseline": end_baseline,
            "lre_dates": lre_dates,
        })
    stats_df = pd.DataFrame(alert_stats)

    pred_sorted = (
        pred_df[["airport", "airport_alert_id", "predicted_date_end_alert", "confidence"]]
        .sort_values(["airport", "airport_alert_id", "predicted_date_end_alert"])
    )

    rows = []
    n_thetas = len(thetas)
    for i, theta in enumerate(thetas):
        if (i + 1) % 50 == 0 or i == n_thetas - 1:
            print(f"  {i+1}/{n_thetas} thetas traites")

        pred_t = (
            pred_sorted[pred_sorted["confidence"] >= theta]
            .groupby(["airport", "airport_alert_id"])["predicted_date_end_alert"]
            .min()
            .reset_index()
        )

        if len(pred_t) == 0:
            rows.append({"theta": float(theta), "gain_seconds": 0.0,
                         "gain_hours": 0.0, "missed_lightnings": 0,
                         "risk": 0.0, "selected_alerts": 0})
            continue

        merged = stats_df.merge(pred_t, on=["airport", "airport_alert_id"], how="inner")

        gain = (
            merged["end_baseline"] - merged["predicted_date_end_alert"]
        ).dt.total_seconds().sum()

        missed = sum(
            int((lre_dates > end_pred).sum())
            for lre_dates, end_pred in zip(
                merged["lre_dates"], merged["predicted_date_end_alert"]
            )
        )

        rows.append({
            "theta": float(theta),
            "gain_seconds": float(gain),
            "gain_hours": float(gain / 3600.0),
            "missed_lightnings": int(missed),
            "risk": float(missed / tot_lightnings) if tot_lightnings > 0 else 0.0,
            "selected_alerts": int(len(merged)),
        })

    return pd.DataFrame(rows).sort_values("theta").reset_index(drop=True)


def choose_best_theta(results_df: pd.DataFrame, acceptable_risk: float = ACCEPTABLE_RISK):
    admissible = results_df.loc[results_df["risk"] < acceptable_risk].copy()
    if len(admissible) == 0:
        return None
    return admissible.sort_values(["gain_seconds", "theta"], ascending=[False, True]).iloc[0]


def make_plots(results_df: pd.DataFrame, best_theta: float, out_dir: Path):
    best_row = results_df.loc[np.isclose(results_df["theta"], best_theta)].iloc[0]
    step = max(1, len(results_df) // 15)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(results_df["theta"], results_df["gain_hours"],
                 marker="o", markersize=3, color="#3078c8")
    axes[0].axvline(best_theta, linestyle="--", color="#e74c3c",
                    label=f"theta optimal = {best_theta:.3f}")
    axes[0].set_xlabel("Theta"); axes[0].set_ylabel("Gain (heures)")
    axes[0].set_title("Gain en fonction de theta")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(results_df["theta"], results_df["risk"],
                 marker="o", markersize=3, color="#e74c3c")
    axes[1].axhline(ACCEPTABLE_RISK, linestyle="--", color="#f39c12",
                    label=f"Risque max = {ACCEPTABLE_RISK}")
    axes[1].axvline(best_theta, linestyle="--", color="#e74c3c",
                    label=f"theta optimal = {best_theta:.3f}")
    axes[1].set_xlabel("Theta"); axes[1].set_ylabel("Risque")
    axes[1].set_title("Risque en fonction de theta")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(results_df["risk"], results_df["gain_hours"],
                 marker="o", markersize=4, color="#3078c8", linewidth=1.5)
    for _, row in results_df.iloc[::step].iterrows():
        axes[2].annotate(f"{row['theta']:.2f}", (row["risk"], row["gain_hours"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=7, color="#555")
    axes[2].axvline(ACCEPTABLE_RISK, linestyle="--", color="#f39c12",
                    label=f"Risque max = {ACCEPTABLE_RISK}")
    axes[2].scatter(
        [best_row["risk"]], [best_row["gain_hours"]], s=100, color="#e74c3c", zorder=5,
        label=f"theta={best_theta:.3f} | gain={best_row['gain_hours']:.1f}h | risque={best_row['risk']:.4f}"
    )
    axes[2].set_xlabel("Risque (fraction de LRE manques)")
    axes[2].set_ylabel("Gain (heures)")
    axes[2].set_title("Compromis gain / risque selon theta")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    plt.suptitle(f"Analyse theta — theta optimal = {best_theta:.3f}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "theta_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig2, ax = plt.subplots(figsize=(9, 6))
    ax.plot(results_df["risk"], results_df["gain_hours"],
            marker="o", markersize=4, color="#3078c8")
    ax.axvline(ACCEPTABLE_RISK, linestyle="--", color="#f39c12",
               label=f"Risque max = {ACCEPTABLE_RISK}")
    ax.scatter([best_row["risk"]], [best_row["gain_hours"]], s=100, color="#e74c3c", zorder=5,
               label=f"theta={best_theta:.3f} | gain={best_row['gain_hours']:.1f}h")
    for _, row in results_df.iloc[::step].iterrows():
        ax.annotate(f"theta={row['theta']:.2f}", (row["risk"], row["gain_hours"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7, color="#555")
    ax.set_xlabel("Risque (fraction de LRE manques)")
    ax.set_ylabel("Gain (heures)")
    ax.set_title("Compromis gain / risque — selection du theta optimal")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "gain_vs_risk.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "data/segment_alerts_all_airports_train.parquet"
    pred_path = sys.argv[2] if len(sys.argv) > 2 else "output/model_comparison_with_xgboost/test_predictions_long.csv"
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("output/theta_analysis_xgboost")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chargement brut : {raw_path}")
    raw_df = load_raw_alerts(raw_path)

    print(f"Chargement predictions : {pred_path}")
    pred_df = load_predictions(pred_path)

    print("Restriction aux alertes presentes dans les predictions...")
    test_raw = prepare_test_subset(raw_df, pred_df)

    grid = np.round(np.linspace(0.0, 1.0, 101), 3)
    unique_scores = np.sort(pred_df["confidence"].dropna().unique())
    thetas = np.unique(np.concatenate([grid, unique_scores]))
    print(f"\nNombre de thetas a tester : {len(thetas)}")

    print("Calcul du gain et du risque pour chaque theta...")
    results_df = compute_theta_results(test_raw, pred_df, thetas)
    results_df.to_csv(out_dir / "theta_results.csv", index=False)

    best_row = choose_best_theta(results_df, acceptable_risk=ACCEPTABLE_RISK)
    if best_row is None:
        raise ValueError("Aucun theta ne respecte la contrainte de risque.")

    best_theta = float(best_row["theta"])

    summary_df = pd.DataFrame([{
        "best_theta":        best_theta,
        "acceptable_risk":   ACCEPTABLE_RISK,
        "risk":              float(best_row["risk"]),
        "gain_hours":        float(best_row["gain_hours"]),
        "gain_seconds":      float(best_row["gain_seconds"]),
        "missed_lightnings": int(best_row["missed_lightnings"]),
        "selected_alerts":   int(best_row["selected_alerts"]),
    }])
    summary_df.to_csv(out_dir / "best_theta_summary.csv", index=False)

    print("Creation des graphiques...")
    make_plots(results_df, best_theta, out_dir)

    print("\nResume du meilleur theta")
    print("-" * 60)
    print(summary_df.to_string(index=False))
    print(f"\nFichiers sauvegardes dans : {out_dir}")
    print("- theta_results.csv")
    print("- best_theta_summary.csv")
    print("- theta_analysis.png")
    print("- gain_vs_risk.png")


if __name__ == "__main__":
    main()