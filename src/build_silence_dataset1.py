import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "output/model_comparison_2A_A/roc_points.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/model_comparison_2A_A/roc_curve_2A.png"

    df = pd.read_csv(input_path)

    required_cols = {"model", "fpr", "tpr"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier ROC : {missing}")

    plt.figure(figsize=(8, 6))

    for model_name, sub in df.groupby("model"):
        sub = sub.sort_values("fpr")
        plt.plot(sub["fpr"], sub["tpr"], label=model_name)

    plt.plot([0, 1], [0, 1], linestyle="--", label="random classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison - 2A Models")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Courbe ROC sauvegardée : {output}")


if __name__ == "__main__":
    main()