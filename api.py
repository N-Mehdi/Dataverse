"""
FastAPI backend — Lightning Alert
Lance avec : uvicorn api:app --reload
"""

import io
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.build_silence_dataset import build_silence_dataset
from src.predict import load_raw_dataset, build_predictions_from_scores

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = Path("output/model_full_with_xgboost/model_xgboost_full.pkl")
THETA = 0.906

FEATURE_EXCLUDED = {
    "airport_alert_id",
    "alert_group",
    "obs_start",
    "alert_start",
    "decision_time",
    "cg_reference_index",
    "minutes_since_reference_cg",
    "y",
}

app = FastAPI(title="DATAVERSE API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model — chargé une seule fois au démarrage
# ---------------------------------------------------------------------------

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    silence_df = build_silence_dataset(raw_df)

    if len(silence_df) == 0:
        raise ValueError("Aucune ligne générée dans le silence dataset.")

    model = get_model()
    feature_cols = [c for c in silence_df.columns if c not in FEATURE_EXCLUDED]
    X_pred = silence_df[feature_cols]

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_pred)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_pred)
    else:
        raise ValueError("Le modèle ne fournit ni predict_proba ni decision_function.")

    pred_df = build_predictions_from_scores(silence_df, scores)
    pred_df["end_alert"] = pred_df["confidence"] >= THETA
    return pred_df


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if str(df[col].dtype) in (
            "large_utf8",
            "string[pyarrow]",
            "string",
            "object",
        ) or hasattr(df[col].dtype, "pyarrow_dtype"):
            try:
                df[col] = df[col].astype(str)
            except Exception:
                pass
    return df


def load_upload(file: UploadFile) -> pd.DataFrame:
    content = file.file.read()
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    return load_raw_dataset(tmp_path)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/summary")
async def predict_summary(file: UploadFile = File(...)):
    """
    Retourne le tableau récapitulatif (une ligne par alerte) en JSON.
    """
    try:
        raw_df = load_upload(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture fichier : {e}")

    try:
        pred_df = run_pipeline(raw_df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    pred_df = sanitize_df(pred_df)

    # Métadonnées sur le fichier
    meta = {
        "n_lightnings": int(len(raw_df)),
        "n_airports": int(raw_df["airport"].nunique()),
        "n_lightnings_with_alert": int(raw_df["airport_alert_id"].notna().sum()),
    }

    # Récapitulatif par alerte
    summary = (
        pred_df.groupby(["airport", "airport_alert_id"])
        .agg(
            conf_max=("confidence", "max"),
            conf_last=("confidence", "last"),
            n_instants=("confidence", "count"),
            end_alert=("end_alert", "any"),
        )
        .reset_index()
        .sort_values(["airport", "airport_alert_id"])
        .reset_index(drop=True)
    )
    summary["conf_max"] = summary["conf_max"].round(3)
    summary["conf_last"] = summary["conf_last"].round(3)

    return {
        "meta": meta,
        "n_alerts": len(summary),
        "summary": summary.to_dict(orient="records"),
    }


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Retourne le predictions.csv complet en téléchargement.
    """
    try:
        raw_df = load_upload(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture fichier : {e}")

    try:
        pred_df = run_pipeline(raw_df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    pred_df = sanitize_df(pred_df)
    csv_bytes = pred_df.to_csv(index=False).encode("utf-8")

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )
