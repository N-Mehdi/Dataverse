"""
predict.py
----------
Inférence sur une alerte en cours.

Pour une alerte donnée, à un instant t (= X minutes après le dernier
éclair nuage-sol observé), on estime :
  - La probabilité que l'orage soit déjà terminé
  - La courbe de survie complète S(t)
  - La comparaison avec la baseline règle des 30 minutes

Vocabulaire des temps
---------------------
- t_abs  : temps absolu depuis le DÉBUT de l'alerte (premier éclair CG)
- t_now  : = time_since_last_cg, minutes écoulées depuis le DERNIER éclair CG
- t_rel  : temps RELATIF depuis maintenant (axe X du graphique)

S(t_abs) : probabilité que l'alerte soit encore active à t_abs minutes depuis le début.

Usage :
    from predict import Predictor
    p = Predictor("models/rsf_model.pkl")
    result = p.predict(features_dict, time_since_last_cg=15)
    print(result)
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from features import ACTIVE_FEATURES

# Seuils optimisés par aéroport (issus de optimize_threshold.py, plafond 15%)
SEUILS_PAR_AIRPORT = {
    "Ajaccio": 0.84,
    "Bastia": 0.73,
    "Biarritz": 0.77,
    "Nantes": 0.74,
    "Pise": 0.73,
}
SEUIL_DEFAUT = 0.77  # fallback si aéroport inconnu


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------


class Predictor:
    """
    Charge un modèle sauvegardé et prédit la probabilité de fin d'alerte.

    Paramètres
    ----------
    model_path : chemin vers le fichier .pkl sauvegardé par model.py
    seuil      : probabilité minimale pour recommander la levée d'alerte
    """

    def __init__(self, model_path: str, seuil: float = SEUIL_DEFAUT):
        checkpoint = joblib.load(model_path)
        self.model = checkpoint["model"]
        self.scaler = checkpoint["scaler"]
        self.seuil = seuil
        self.model_type = type(self.model).__name__
        print(f"Modèle chargé : {self.model_type}")

    # -----------------------------------------------------------------------
    # Prédiction principale
    # -----------------------------------------------------------------------

    def predict(
        self,
        features: dict,
        time_since_last_cg: float,
        horizon: int = 60,
    ) -> dict:
        """
        Prédit la probabilité de fin d'alerte.

        Paramètres
        ----------
        features           : dict avec les colonnes ACTIVE_FEATURES
        time_since_last_cg : minutes écoulées depuis le dernier éclair CG
                             = position actuelle sur l'axe du temps absolu
        horizon            : nombre de minutes à afficher après maintenant

        Retourne
        --------
        dict avec prob_end, survival_curve (axe = temps depuis maintenant),
        recommendation, gain_vs_baseline
        """
        t_now = time_since_last_cg

        # Axe temps absolu : de 0 jusqu'à t_now + horizon
        t_max_abs = t_now + horizon
        times_abs = np.arange(0, t_max_abs + 1, 1, dtype=float)

        # Construire le vecteur de features dans le bon ordre
        X = pd.DataFrame([features])[ACTIVE_FEATURES]

        # Appliquer la normalisation si présente (RSF)
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X.values)
        else:
            X_scaled = X.values

        # Courbe de survie S(t_abs)
        if self.model_type == "CoxPHFitter":
            S_abs = self._predict_cox(X, times_abs)
        elif self.model_type == "RandomSurvivalForest":
            S_abs = self._predict_rsf(X_scaled, times_abs)
        else:
            raise ValueError(f"Type de modèle inconnu : {self.model_type}")

        # ── Probabilités à l'instant actuel ─────────────────────────────────
        # S(t_now) = P(alerte encore active maintenant)
        s_now = float(np.interp(t_now, times_abs, S_abs, right=0.0))
        prob_active = s_now
        prob_end = 1.0 - s_now

        # ── Courbe conditionnelle pour l'affichage ───────────────────────────
        # S_cond(t_rel) = P(alerte active dans t_rel min | active maintenant)
        #              = S(t_now + t_rel) / S(t_now)
        times_rel = np.arange(0, horizon + 1, 1, dtype=float)
        times_abs_future = t_now + times_rel
        S_future = np.interp(times_abs_future, times_abs, S_abs, right=0.0)

        if s_now > 0:
            S_conditional = np.clip(S_future / s_now, 0, 1)
        else:
            S_conditional = np.zeros_like(times_rel)

        # ── Temps de levée prédit par le modèle (absolu) ────────────────────
        # Premier t_abs où 1 - S(t_abs) >= seuil
        prob_end_abs = 1.0 - S_abs
        idx_lever = np.searchsorted(prob_end_abs, self.seuil)

        if idx_lever < len(times_abs):
            t_lever_abs = float(times_abs[idx_lever])
        else:
            t_lever_abs = float(t_max_abs)  # pas atteint dans l'horizon
        # Temps relatif avant levée depuis maintenant
        t_lever_rel = max(0.0, t_lever_abs - t_now)

        # ── Gain vs baseline ─────────────────────────────────────────────────
        # Baseline : lève toujours à t_abs = 30 min
        # Modèle   : lève à t_abs = t_lever_abs
        # Gain > 0 = modèle lève AVANT la baseline → bon
        gain_vs_baseline = round(30.0 - t_lever_abs, 1)

        # ── Recommandation ───────────────────────────────────────────────────
        recommendation = "LEVER" if prob_end >= self.seuil else "MAINTENIR"

        return {
            "time_since_last_cg": t_now,
            "prob_end": round(prob_end, 4),
            "prob_active": round(prob_active, 4),
            "survival_curve": pd.DataFrame(
                {
                    "time_rel": times_rel,
                    "survival_prob": S_conditional,
                }
            ),
            "recommendation": recommendation,
            "seuil": self.seuil,
            "t_lever_abs": t_lever_abs,  # minutes depuis début alerte
            "t_lever_rel": t_lever_rel,  # minutes depuis maintenant
            "baseline_active": t_now < 30,
            "gain_vs_baseline": gain_vs_baseline,
        }

    # -----------------------------------------------------------------------
    # Prédictions internes
    # -----------------------------------------------------------------------

    def _predict_cox(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        sf = self.model.predict_survival_function(X, times=times)
        return sf.iloc[:, 0].values

    def _predict_rsf(self, X_scaled: np.ndarray, times: np.ndarray) -> np.ndarray:
        sf_funcs = self.model.predict_survival_function(X_scaled)
        sf_func = sf_funcs[0]
        return np.interp(times, sf_func.x, sf_func(sf_func.x))

    # -----------------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------------

    def plot(
        self,
        result: dict,
        airport: str = "",
        true_duration: float = None,
        save_path: str = None,
    ):
        """
        Graphique de la courbe de survie conditionnelle.

        Axe X = temps depuis maintenant (t_rel).
        Ligne orange   = maintenant (x=0).
        Ligne rouge    = baseline 30 min (x = 30 - t_now, si future).
        Ligne verte    = temps prédit de levée par le modèle.
        Ligne violette = fin réelle (si true_duration fournie).
        """
        df = result["survival_curve"]
        t = result["time_since_last_cg"]
        prob = result["prob_end"]
        gain = result["gain_vs_baseline"]

        fig, ax = plt.subplots(figsize=(10, 5))

        # Courbe de survie conditionnelle
        ax.plot(
            df["time_rel"],
            df["survival_prob"],
            color="steelblue",
            linewidth=2,
            label="Courbe de survie conditionnelle",
        )

        # Zone levée probable (S < 1-seuil)
        ax.fill_between(
            df["time_rel"],
            0,
            df["survival_prob"],
            where=df["survival_prob"] < (1.0 - self.seuil),
            alpha=0.15,
            color="green",
            label="Zone levée probable",
        )

        # Maintenant (t_rel = 0)
        ax.axvline(
            x=0,
            color="orange",
            linewidth=2.5,
            label=f"Maintenant (t={t:.0f} min depuis début alerte)",
        )

        # Baseline : temps restant avant la règle 30 min
        t_baseline_rel = 30.0 - t
        if t_baseline_rel > 0:
            ax.axvline(
                x=t_baseline_rel,
                color="red",
                linewidth=1.5,
                linestyle="--",
                label=f"Baseline 30 min (dans {t_baseline_rel:.0f} min)",
            )
        else:
            ax.axvline(
                x=0,
                color="red",
                linewidth=1.5,
                linestyle="--",
                label="Baseline 30 min (déjà dépassée)",
            )

        # Seuil horizontal
        ax.axhline(
            y=1.0 - self.seuil,
            color="green",
            linewidth=1,
            linestyle=":",
            alpha=0.7,
            label=f"Seuil levée ({self.seuil:.0%})",
        )

        # Temps prédit de levée par le modèle
        t_lever_rel = result["t_lever_rel"]
        if t_lever_rel <= df["time_rel"].max():
            ax.axvline(
                x=t_lever_rel,
                color="limegreen",
                linewidth=1.5,
                linestyle="-.",
                label=f"Levée modèle (dans {t_lever_rel:.0f} min)",
            )

        # Durée réelle si fournie
        if true_duration is not None:
            t_real_rel = true_duration - t
            if 0 <= t_real_rel <= df["time_rel"].max():
                ax.axvline(
                    x=t_real_rel,
                    color="purple",
                    linewidth=1.5,
                    linestyle=":",
                    label=f"Fin réelle (dans {t_real_rel:.1f} min)",
                )

        # Titre
        color_rec = "green" if result["recommendation"] == "LEVER" else "red"
        gain_str = f"+{gain:.0f} min" if gain > 0 else f"{gain:.0f} min"
        ax.set_title(
            f"Prédiction fin d'alerte — {airport}\n"
            f"Recommandation : {result['recommendation']}  "
            f"(P(terminé) = {prob:.0%}, seuil {self.seuil:.0%})  |  "
            f"Gain vs baseline : {gain_str}",
            color=color_rec,
            fontsize=11,
        )
        ax.set_xlabel("Temps depuis maintenant (minutes)")
        ax.set_ylabel("P(alerte encore active | active maintenant)")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, df["time_rel"].max())
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure sauvegardée : {save_path}")

        return fig


# ---------------------------------------------------------------------------
# Point d'entrée — test sur une alerte réelle + stats sur 50 alertes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    features_path = sys.argv[1] if len(sys.argv) > 1 else "data/features.parquet"
    df = pd.read_parquet(features_path)

    # ── Test 1 : simulation sur une alerte unique ────────────────────────────
    sample = df.sample(1, random_state=42).iloc[0]
    airport = sample["airport"]
    alert_id = sample["airport_alert_id"]
    duration = sample["duration"]

    print(f"\nAlerte réelle choisie :")
    print(f"  Aéroport     : {airport}")
    print(f"  Alert ID     : {alert_id}")
    print(f"  Durée réelle : {duration:.1f} min  (baseline lève à 30 min)")

    features_dict = {col: sample[col] for col in ACTIVE_FEATURES}

    # Seuil optimisé pour cet aéroport
    seuil = SEUILS_PAR_AIRPORT.get(airport, SEUIL_DEFAUT)
    print(f"  Seuil utilisé : {seuil:.0%} (optimisé pour {airport})")
    predictor = Predictor("models/rsf_model.pkl", seuil=seuil)

    print(f"\n── Simulation minute par minute ────────────────────────────────────")
    print(
        f"{'t_now':>6} │ {'P(terminé)':>11} │ {'t_lever_abs':>11} │ {'Gain':>7} │ {'Baseline':>10} │ Reco"
    )
    print("─" * 68)

    for t in [0, 5, 10, 15, 20, 25, 30]:
        r = predictor.predict(features_dict, time_since_last_cg=t)
        baseline = "LEVER" if not r["baseline_active"] else "MAINTENIR"
        gain_str = (
            f"+{r['gain_vs_baseline']:.1f}"
            if r["gain_vs_baseline"] > 0
            else f"{r['gain_vs_baseline']:.1f}"
        )
        print(
            f"{t:>5}' │ {r['prob_end']:>10.1%} │ "
            f"{r['t_lever_abs']:>9.1f}' │ {gain_str:>7} │ "
            f"{baseline:>10} │ {r['recommendation']}"
        )

    # Graphique à t=0 (début d'alerte)
    result_0 = predictor.predict(features_dict, time_since_last_cg=0)
    fig = predictor.plot(
        result_0,
        airport=airport,
        true_duration=duration,
        save_path="outputs/prediction_reelle.png",
    )
    gain_0 = result_0["gain_vs_baseline"]
    print(f"\nGain vs baseline (à t=0) : {gain_0:+.1f} min")
    print(f"  Modèle lève à t_abs = {result_0['t_lever_abs']:.1f} min")
    print(f"  Baseline lève à     t_abs = 30 min")
    plt.show()

    # ── Test 2 : gain moyen sur 50 alertes ──────────────────────────────────
    print(f"\n── Évaluation sur 50 alertes aléatoires (t_now=0) ─────────────────")
    gains = []
    faux_allclear = 0

    for _, row in df.sample(50, random_state=0).iterrows():
        fd = {col: row[col] for col in ACTIVE_FEATURES}
        s = SEUILS_PAR_AIRPORT.get(row["airport"], SEUIL_DEFAUT)
        p = Predictor("models/rsf_model.pkl", seuil=s)
        r = p.predict(fd, time_since_last_cg=0)
        gains.append(r["gain_vs_baseline"])
        if r["t_lever_abs"] < row["duration"]:
            faux_allclear += 1

    gains = np.array(gains)
    print(f"  Gain moyen        : {gains.mean():+.1f} min")
    print(f"  Gain médian       : {np.median(gains):+.1f} min")
    print(f"  Alertes avec gain : {(gains > 0).sum()}/50  ({(gains > 0).mean():.0%})")
    print(f"  Faux all-clear    : {faux_allclear}/50  ({faux_allclear / 50:.0%})")
