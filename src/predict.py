import sys
import joblib
import pandas as pd
from pathlib import Path
from global_roc_comparison import load_dataset, NON_FEATURE_COLS, build_feature_lists

input_path = sys.argv[1] if len(sys.argv) > 1 else "output/silence_dataset.parquet"
model_path = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "output/model_comparison_with_xgboost/model_xgboost.pkl"
)
output_path = sys.argv[3] if len(sys.argv) > 3 else "output/predictions.csv"

pipe = joblib.load(model_path)

test_df = load_dataset(input_path)
feature_cols, _, _ = build_feature_lists(test_df)
test_df["confidence"] = pipe.predict_proba(test_df[feature_cols])[:, 1]

predictions = test_df[["airport", "airport_alert_id"]].copy()
predictions["predicted_date_end_alert"] = test_df["decision_time"]
predictions["confidence"] = test_df["confidence"]

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(output_path, index=False)
print(f"predictions.csv généré : {len(predictions)} lignes → {output_path}")
