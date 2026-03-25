import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from build_silence_dataset import build_silence_dataset, INNER_RADIUS_KM


def load_raw_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["type"] = np.where(df["icloud"].fillna(False), "IC", "CG")
    df["zone"] = np.where(df["dist"] < INNER_RADIUS_KM, "inner", "outer")
    df = df.sort_values(["airport", "date"]).reset_index(drop=True)
    df["airport_alert_id"] = df["airport_alert_id"].astype("string")
    return df


def build_predictions_from_scores(silence_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    pred_df = silence_df[
        ["airport", "airport_alert_id", "decision_time"]
    ].copy()

    pred_df["prediction_date"] = pred_df["decision_time"]
    pred_df["predicted_date_end_alert"] = pred_df["decision_time"]
    pred_df["confidence"] = scores.astype(float)

    pred_df = pred_df[
        [
            "airport",
            "airport_alert_id",
            "prediction_date",
            "predicted_date_end_alert",
            "confidence",
        ]
    ].sort_values(["airport", "airport_alert_id", "prediction_date"]).reset_index(drop=True)

    return pred_df


def main():
    input_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/segment_alerts_all_airports_truncated.csv"
    )
    model_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "output/model_full_with_xgboost/model_xgboost_full.pkl"
    )
    output_path = (
        sys.argv[3]
        if len(sys.argv) > 3
        else "output/predictions.csv"
    )

    print(f"Chargement des données tronquées : {input_path}")
    raw_df = load_raw_dataset(input_path)

    print("Construction du dataset de silences décisionnels...")
    silence_df = build_silence_dataset(raw_df)

    if len(silence_df) == 0:
        raise ValueError("Aucune ligne générée dans le silence dataset.")

    print(f"Chargement du modèle : {model_path}")
    model = joblib.load(model_path)

    feature_excluded = {
        "airport_alert_id",
        "alert_group",
        "obs_start",
        "alert_start",
        "decision_time",
        "cg_reference_index",
        "minutes_since_reference_cg",
        "y",
    }
    feature_cols = [c for c in silence_df.columns if c not in feature_excluded]
    X_pred = silence_df[feature_cols]

    print("Calcul des scores...")
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_pred)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_pred)
    else:
        raise ValueError("Le modèle ne fournit ni predict_proba ni decision_function.")

    print("Construction du fichier predictions.csv...")
    pred_df = build_predictions_from_scores(silence_df, scores)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)

    print("\nRésumé")
    print("-" * 60)
    print(f"Nb lignes predictions : {len(pred_df)}")
    print(f"Nb alertes : {pred_df[['airport', 'airport_alert_id']].drop_duplicates().shape[0]}")
    print(f"Fichier sauvegardé : {output_path}")


if __name__ == "__main__":
    main()