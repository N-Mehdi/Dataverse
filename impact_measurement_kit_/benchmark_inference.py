import argparse
import pickle
import time
from pathlib import Path

import pandas as pd

from measure_utils import run_measured


def infer_from_parquet(model_path: Path, parquet_path: Path, n_rows: int):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_parquet(parquet_path)
    sample = df.head(n_rows).copy()

    # Try to keep only numeric columns if the model expects a bare matrix.
    # If your saved pipeline handles categoricals itself, you can remove this block.
    X = sample.select_dtypes(include=["number", "bool"]).copy()

    if hasattr(model, "predict_proba"):
        _ = model.predict_proba(X)
    else:
        _ = model.predict(X)


def dummy_inference(n_rows: int):
    # Fallback when a production-ready prediction function is not wired yet.
    for _ in range(n_rows):
        time.sleep(0.005)


def main():
    parser = argparse.ArgumentParser(description="Benchmark single-alert and batch inference.")
    parser.add_argument("--model-path", default="model_xgboost.pkl")
    parser.add_argument("--parquet-path", default="output/silence_dataset.parquet")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--output-dir", default="impact_runs")
    parser.add_argument("--dummy", action="store_true", help="Use a dummy inference loop if your model API is not ready")
    args = parser.parse_args()

    if args.dummy:
        run_measured(lambda: dummy_inference(1), "infer_1_alert", args.output_dir, "Dummy inference 1 alerte")
        run_measured(lambda: dummy_inference(args.batch_size), f"infer_{args.batch_size}_alerts", args.output_dir, f"Dummy inference {args.batch_size} alertes")
    else:
        model_path = Path(args.model_path)
        parquet_path = Path(args.parquet_path)
        run_measured(
            lambda: infer_from_parquet(model_path, parquet_path, 1),
            "infer_1_alert",
            args.output_dir,
            "Inférence unitaire",
        )
        run_measured(
            lambda: infer_from_parquet(model_path, parquet_path, args.batch_size),
            f"infer_{args.batch_size}_alerts",
            args.output_dir,
            f"Inférence batch de {args.batch_size} alertes",
        )


if __name__ == "__main__":
    main()
