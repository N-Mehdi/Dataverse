import argparse
import shlex
import subprocess
from pathlib import Path

from measure_utils import run_measured


def run_command(command: str, cwd: str | None = None):
    subprocess.run(shlex.split(command), check=True, cwd=cwd)


def main():
    parser = argparse.ArgumentParser(description="Run an impact measurement campaign around project scripts.")
    parser.add_argument("--project-root", default=".", help="Path to the project root")
    parser.add_argument("--build-cmd", default="python build_silence_dataset.py", help="Command for dataset building")
    parser.add_argument("--train-cmd", default="python global_roc_comparison.py", help="Command for model training/evaluation")
    parser.add_argument("--predict-cmd", default="python predict.py", help="Command for batch inference")
    parser.add_argument("--output-dir", default="impact_runs", help="Directory where measurements are saved")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    if not args.skip_build:
        run_measured(
            lambda: run_command(args.build_cmd, cwd=str(project_root)),
            run_name="build_dataset",
            output_dir=args.output_dir,
            notes="Construction du dataset décisionnel",
        )

    if not args.skip_train:
        run_measured(
            lambda: run_command(args.train_cmd, cwd=str(project_root)),
            run_name="train_model",
            output_dir=args.output_dir,
            notes="Entraînement / évaluation du modèle",
        )

    if not args.skip_predict:
        run_measured(
            lambda: run_command(args.predict_cmd, cwd=str(project_root)),
            run_name="predict_batch",
            output_dir=args.output_dir,
            notes="Inférence batch / prédiction",
        )


if __name__ == "__main__":
    main()
