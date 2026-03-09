"""
predict_xgb.py
--------------
Prédit en temps réel si une alerte orageuse est terminée.

Usage production :
    Appelé à chaque nouvel éclair (ou à intervalles réguliers) pendant une alerte.
    Retourne la probabilité qu'un éclair CG survienne dans les 30 prochaines minutes
    et la décision de lever ou non l'alerte.

Usage en ligne de commande (test) :
    python src/predict_xgb.py <airport> <csv_éclairs> [--now "2024-06-15 14:32:00"]

    csv_éclairs : fichier CSV avec les colonnes du dataset d'origine
                  (date, amplitude, dist, maxis, azimuth, lat, lon, icloud)
                  filtré sur l'alerte en cours.
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from datetime import timezone

from features_snapshot import compute_snapshot_features, FEATURE_COLS, HORIZON

# ── Seuils optimaux par aéroport ─────────────────────────────────────────────
SEUILS_PAR_AIRPORT = {
    "Ajaccio": 0.18,
    "Bastia": 0.10,
    "Biarritz": 0.10,
    "Nantes": 0.10,
    "Pise": 0.11,
}
SEUIL_DEFAUT = 0.15

MODEL_PATH = "models/xgb_model.pkl"


def load_model(path: str = MODEL_PATH):
    bundle = joblib.load(path)
    return bundle["model"]


def predict(
    eclairs_alerte: pd.DataFrame,
    airport: str,
    now: pd.Timestamp,
    model=None,
    model_path: str = MODEL_PATH,
) -> dict:
    """
    Prédit si l'alerte est terminée à l'instant `now`.

    Paramètres
    ----------
    eclairs_alerte : DataFrame des éclairs CG de l'alerte en cours
                     (colonnes : date, amplitude, dist, maxis, azimuth, lat, lon)
    airport        : nom de l'aéroport (ex: "Ajaccio")
    now            : instant courant (pd.Timestamp avec timezone UTC)
    model          : modèle XGBoost déjà chargé (optionnel, sinon chargé depuis model_path)

    Retourne
    --------
    dict avec :
        proba              : P(éclair CG dans les 30 prochaines minutes)
        decision           : "LEVER" ou "MAINTENIR"
        time_since_last_cg : minutes depuis le dernier éclair CG
        seuil              : seuil utilisé pour cet aéroport
        features           : dict des features calculées
    """
    if model is None:
        model = load_model(model_path)

    # Filtrer les éclairs CG uniquement
    if "icloud" in eclairs_alerte.columns:
        cg = eclairs_alerte[eclairs_alerte["icloud"] == False].copy()
    else:
        cg = eclairs_alerte.copy()

    if len(cg) < 2:
        return {
            "proba": 1.0,
            "decision": "MAINTENIR",
            "time_since_last_cg": 0.0,
            "seuil": SEUILS_PAR_AIRPORT.get(airport, SEUIL_DEFAUT),
            "features": {},
            "message": "Pas assez d'éclairs CG pour décider (< 2)",
        }

    cg = cg.sort_values("date").reset_index(drop=True)
    cg["date"] = pd.to_datetime(cg["date"], utc=True)

    # S'assurer que `now` est en UTC
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    t_alert_start = cg["date"].min()
    t_last_cg = cg["date"].max()

    # Vérification : now doit être >= dernier éclair
    if now < t_last_cg:
        now = t_last_cg

    time_since_last_cg = (now - t_last_cg).total_seconds() / 60

    # Calcul des features à l'instant now
    features = compute_snapshot_features(cg, now, t_alert_start, airport)

    # Prédiction
    X = pd.DataFrame([features])[FEATURE_COLS]
    proba = float(model.predict_proba(X)[0, 1])

    seuil = SEUILS_PAR_AIRPORT.get(airport, SEUIL_DEFAUT)
    decision = "LEVER" if proba < seuil else "MAINTENIR"

    return {
        "proba": proba,
        "decision": decision,
        "time_since_last_cg": time_since_last_cg,
        "seuil": seuil,
        "features": features,
        "message": (
            f"P(éclair dans {HORIZON} min) = {proba:.1%} "
            f"{'<' if proba < seuil else '>='} seuil {seuil:.0%} "
            f"→ {decision}"
        ),
    }


def predict_sequence(
    eclairs_alerte: pd.DataFrame,
    airport: str,
    model=None,
    model_path: str = MODEL_PATH,
    interval_min: int = 2,
) -> pd.DataFrame:
    """
    Simule la prédiction à intervalles réguliers pendant toute une alerte.
    Utile pour visualiser l'évolution de P au cours du temps.

    Retourne un DataFrame avec une ligne par instant évalué.
    """
    if model is None:
        model = load_model(model_path)

    if "icloud" in eclairs_alerte.columns:
        cg = eclairs_alerte[eclairs_alerte["icloud"] == False].copy()
    else:
        cg = eclairs_alerte.copy()

    cg = cg.sort_values("date").reset_index(drop=True)
    cg["date"] = pd.to_datetime(cg["date"], utc=True)

    t_start = cg["date"].min()
    t_last_cg = cg["date"].max()
    t_end = t_last_cg + pd.Timedelta(minutes=30)

    rows = []
    t = t_start + pd.Timedelta(minutes=interval_min * 2)
    while t <= t_end:
        hist = cg[cg["date"] <= t]
        if len(hist) < 2:
            t += pd.Timedelta(minutes=interval_min)
            continue

        result = predict(hist, airport, t, model=model)
        rows.append(
            {
                "t": t,
                "elapsed_min": (t - t_start).total_seconds() / 60,
                "time_since_last_cg": result["time_since_last_cg"],
                "proba": result["proba"],
                "decision": result["decision"],
            }
        )
        t += pd.Timedelta(minutes=interval_min)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prédit si une alerte orageuse est terminée."
    )
    parser.add_argument("airport", type=str, help="Nom de l'aéroport (ex: Ajaccio)")
    parser.add_argument("csv", type=str, help="CSV des éclairs de l'alerte en cours")
    parser.add_argument(
        "--now",
        type=str,
        default=None,
        help="Instant courant UTC (ex: '2024-06-15 14:32:00'). Défaut = dernier éclair.",
    )
    parser.add_argument(
        "--sequence",
        action="store_true",
        help="Simule la prédiction sur toute la durée de l'alerte",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help=f"Chemin vers le modèle (défaut: {MODEL_PATH})",
    )
    args = parser.parse_args()

    print(f"Chargement des éclairs depuis {args.csv}...")
    df = pd.read_csv(args.csv)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    model = load_model(args.model)

    if args.sequence:
        print(f"\nSimulation séquentielle pour {args.airport}...")
        seq = predict_sequence(df, args.airport, model=model)
        print(seq.to_string(index=False))

        # Première décision LEVER
        lever = seq[seq["decision"] == "LEVER"]
        if len(lever) > 0:
            first = lever.iloc[0]
            print(
                f"\n→ Première levée possible à t+{first['time_since_last_cg']:.1f} min après dernier éclair"
            )
            print(f"  (soit {first['elapsed_min']:.1f} min après début d'alerte)")
        else:
            print("\n→ Alerte jamais levée dans la simulation")

    else:
        if args.now:
            now = pd.Timestamp(args.now, tz="UTC")
        else:
            cg = df[df["icloud"] == False] if "icloud" in df.columns else df
            now = pd.to_datetime(cg["date"]).max()
            print(f"--now non spécifié, utilisation du dernier éclair : {now}")

        result = predict(df, args.airport, now, model=model)

        print(f"\n{'═' * 50}")
        print(f"  Aéroport              : {args.airport}")
        print(f"  Instant évalué        : {now}")
        print(
            f"  Dernier éclair CG     : {result['time_since_last_cg']:.1f} min avant now"
        )
        print(f"  P(éclair dans 30 min) : {result['proba']:.1%}")
        print(f"  Seuil                 : {result['seuil']:.0%}")
        print(f"  Décision              : {result['decision']}")
        print(f"{'═' * 50}")
        print(f"\n  {result['message']}")
