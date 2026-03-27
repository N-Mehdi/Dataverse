# 🏆 Data Battle IA PAU 2026 – Dataverse

## 👥 Équipe

- **Nom de l'équipe :** Dataverse
- **Membres :**
  - Mehdi NEJI (représentant)
  - Ali BEN HADJ YAHIA
  - Lina GAROUACHI

---

## 🎯 Problématique

Météorage, leader français de la détection foudre, gère des alertes foudre sur les aéroports français. Ces alertes imposent l'arrêt de toute activité en zone extérieure (piste, tarmac) pour des raisons de sécurité. Actuellement, la levée de ces alertes est effectuée manuellement par des opérateurs, avec des délais variables pouvant entraîner des interruptions prolongées et coûteuses.

**Objectif :** permettre la décision de levée d'alerte en prédisant, à chaque instant de silence observé, si l'activité orageuse est réellement terminée dans la zone de 20 km autour de l'aéroport.

---

## 💡 Solution proposée

Nous avons développé un pipeline de machine learning basé sur une **régression logistique** qui, à partir des données brutes d'éclairs captées par le réseau Météorage, génère des instants de décision pendant les silences et prédit la probabilité que l'alerte foudre soit terminée.

Le modèle calcule ~74 features à chaque instant de silence (comptages d'éclairs, distances, amplitudes, fenêtres glissantes 5/10/20 min, inter-arrivées, LRE) et produit un score de confiance. Un seuil θ = 0.906 a été optimisé pour lever les alertes.

La solution est exposée via une interface web (React) connectée à une API (FastAPI) permettant l'analyse batch de fichiers CSV (ou Parquet).

---

## ⚙️ Stack technique

- **Langages :** Python 3.12, JavaScript (ES2022)
- **Frameworks :** FastAPI, React 18, Vite 5
- **Machine Learning :** XGBoost, scikit-learn (pipeline, preprocessing)
- **Données :** pandas, numpy, pyarrow
- **API :** uvicorn, python-multipart
- **Frontend :** Space Grotesk, JetBrains Mono
- IA (outils utilisés) : Claude (Anthropic) - aide au développement


---

## 🚀 Installation & exécution

### Prérequis

- Python 3.12+
- Node.js 18+
- Le fichier des données initial : `segment_alerts_all_airports_train.csv` à placer dans `data/`

### Installation

**Installer uv (si pas déjà installé) :**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
uv : un gestionnaire de packages Python ultra-rapide (remplaçant de pip + venv), développé par Astral.
 
**Backend Python :**
```bash
cd Dataverse
uv sync
source .venv/bin/activate       # Windows : .venv\Scripts\activate
````

**Frontend React :**
```bash
cd frontend
npm install
```

### Exécution

**Courbes ROC des modèles  
Courbe du compromis gain / risque pour le modèle logistique (possible pour xgboost aussi)**
```
python src/main_plots.py
```

Lancer les deux services en parallèle dans deux terminaux séparés.

**Terminal 1 - Backend API :**
```bash
cd Dataverse
source .venv/bin/activate
uvicorn api:app --reload
# Documentation interactive (Swagger) disponible sur http://localhost:8000/docs
```

**Terminal 2 - Frontend :**
```bash
cd Dataverse/frontend
npm run dev
# Interface disponible sur http://localhost:3000
```

**Diagnostic RSE des modèles**
```bash
bash impact_measurement_kit_/compare_rse_models.sh
```


### Utilisation

1. Ouvrir `http://localhost:3000`
2. Aller dans l'onglet **Prédiction**
3. Importer un fichier CSV ou Parquet de données foudre brutes
4. Cliquer sur **Lancer l'analyse**
5. Consulter le tableau récapitulatif par alerte et télécharger `predictions.csv`

---

## 📁 Structure du projet

```
Dataverse/
├── api.py                              # Backend FastAPI
├── src/
│   ├── Logistic_Regression/      
│   │   ├── Logistic_Regression.py                          # Entraîné sur les données tests (optimisé)
│   │   ├── Logistic_Regression_On_All_Data.py              # Entraîné sur toutes les données (omptimisé)
│   │   └── Logistic_Regression_Hyperparameter_Search.py    # Optimise les hyperpamètres du modèle
│   ├── Random_Forest/
│   │   └── ...
│   ├── XGboost/
│   │   └── ...
│   ├── build_silence_dataset.py        # Construction des silences décisionnels
│   ├── predict.py                      # Pipeline de prédiction
│   ├── main.py                         # Orchestration complète du pipeline
│   ├── main_plots.py                   # Visualisation ROC + analyse θ
│   ├── theta_analysis_logistic.py      # Calcul du seuil θ - Logistique
│   └── theta_analysis_xgboost.py       # Calcul du seuil θ - XGBoost
├── output/                             # Modèles entraînés et résultats (pkl non versionnés)
├── frontend/                           # Interface React
├── impact_measurement_kit_/            # Outils de mesure RSE
├── doc/                                # Documentation des features
├── data/                               # Données brutes (non versionnées) et notebook d'analyse
├── pyproject.toml
└── README.md
```