"""
main.py - Pipeline complet : build silence dataset + entraînement des 3 modèles

Usage :
    python main.py
    python main.py --input data/segment_alerts_all_airports_train.csv
    python main.py --skip-build   # si silence_dataset.parquet existe déjà
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]):
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def build_silence_dataset(input_path: str):
    print("\n" + "=" * 60)
    print("ÉTAPE 1 - Construction du silence dataset")
    print("=" * 60)
    run(
        [
            sys.executable,
            "-m",
            "src.build_silence_dataset",
            input_path,
        ]
    )


def train_all_models():
    models = [
        ("XGBoost", "src/XGboost/XGboost_On_All_Data.py"),
        ("Random Forest", "src/Random_Forest/Random_Forest_On_All_Data.py"),
        (
            "Logistic Regression",
            "src/Logistic_Regression/Logistic_Regression_On_All_Data.py",
        ),
    ]

    for i, (name, script) in enumerate(models, start=2):
        print("\n" + "=" * 60)
        print(f"ÉTAPE {i} - Entraînement : {name}")
        print("=" * 60)
        run([sys.executable, script])


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline complet Data Battle - Météorage"
    )
    parser.add_argument(
        "--input",
        default="data/segment_alerts_all_airports_train.csv",
        help="Fichier CSV ou Parquet d'éclairs bruts",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Passer la construction du silence dataset (utilise le parquet existant)",
    )
    args = parser.parse_args()

    if not args.skip_build:
        build_silence_dataset(args.input)

    train_all_models()

    print("\n" + "=" * 60)
    print("Pipeline terminé.")
    print("Modèles sauvegardés dans output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
