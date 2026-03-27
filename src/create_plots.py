"""
main_plots.py - Affiche les courbes ROC des 3 modèles puis l'analyse theta
Usage : python main_plots.py
Fermer chaque fenêtre pour passer à la suivante.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import auc

import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    {
        "name": "Régression Logistique",
        "roc_csv": "output/model_comparison_with_logistic/roc_points.csv",
        "model_key": "logistic",
        "color": "#f0c040",
    },
    {
        "name": "Random Forest",
        "roc_csv": "output/model_comparison_with_random_forest/roc_points.csv",
        "model_key": "random_forest",
        "color": "#4a8fff",
    },
    {
        "name": "XGBoost",
        "roc_csv": "output/model_comparison_with_xgboost/roc_points.csv",
        "model_key": "xgboost",
        "color": "#3ddc84",
    },
]

THETA_IMGS = [
    (
        "Analyse θ - Régression Logistique",
        "output/theta_analysis_logistic/theta_analysis.png",
    ),
]

BG = "#0c0f1a"

# ── 1. Courbes ROC des 3 modèles ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
fig.suptitle(
    "\nCourbes ROC - évaluées sur le jeu de test issu de\nsegment_alerts_all_airports_train.csv",
    color="#e8eaf2",
    fontsize=13,
    y=1.02,
)

for m in MODELS:
    try:
        df = pd.read_csv(m["roc_csv"])
        df = df[df["model"] == m["model_key"]].sort_values("fpr").reset_index(drop=True)
        fpr, tpr = df["fpr"].values, df["tpr"].values
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=m["color"],
            linewidth=2,
            label=f"{m['name']} - AUC = {roc_auc:.4f}",
        )
        print(f"{m['name']} - AUC = {roc_auc:.4f}")
    except FileNotFoundError:
        print(f"Fichier introuvable : {m['roc_csv']}")

ax.plot([0, 1], [0, 1], color="#3a3f55", linewidth=1, linestyle="--", label="Aléatoire")
ax.set_xlabel("FPR", color="#7a7f99", fontsize=11)
ax.set_ylabel("TPR", color="#7a7f99", fontsize=11)
ax.tick_params(colors="#3a3f55")
for spine in ax.spines.values():
    spine.set_edgecolor("#1e2d5a")
ax.legend(
    loc="lower right",
    facecolor="#111520",
    edgecolor="#1e2d5a",
    labelcolor="#e8eaf2",
    fontsize=11,
)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_models.png"), dpi=300, bbox_inches="tight")
plt.show()

# ── 2. Analyses theta ─────────────────────────────────────────────────────────
for title, path in THETA_IMGS:
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    try:
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(title, color="#e8eaf2", fontsize=13, pad=10)
    except FileNotFoundError:
        ax.text(
            0.5,
            0.5,
            f"Image introuvable :\n{path}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#e05050",
            fontsize=11,
        )
        ax.set_title(title, color="#e8eaf2", fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "gain_risque.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()
