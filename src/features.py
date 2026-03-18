"""
features.py — Classification binaire : lever ou maintenir l'alerte

Logique snapshot :
    t = instant d'un CG inner (<20 km)
    Snapshot valide si : gap(t, CG inner suivant) > 20 min
    Features calculées sur [first_cg - 10 min, t] + ]t, t+20]
    Label y=1 si aucun CG inner après t+20, y=0 sinon
"""

import sys
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

PRE_WINDOW_MIN      = 10
DECISION_WINDOW_MIN = 20
INNER_RADIUS_KM     = 20
BURST_PAUSE_MIN     = 5
OUTER_INNER_LAG_MIN = 10

FEATURES = [
    "elapsed",
    "time_since_last_alert",
    "n_cg_inner",
    "amp_max",
    "amp_mean",
    "amp_trend",
    "dist_min",
    "dist_mean",
    "dist_trend",
    "pause_max",
    "n_bursts",
    "act_var",
    "n_ic_inner",
    "ratio_ic_cg",
    "n_ic_inner_win",
    "n_cg_outer",
    "n_cg_outer_win",
    "n_ic_outer",
    "n_ic_outer_win",
]

# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["zone"] = df["dist"].apply(lambda d: "inner" if d < INNER_RADIUS_KM else "outer")
    df["type"] = df["icloud"].apply(lambda x: "IC" if x else "CG")
    return df

# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _linear_slope(times: np.ndarray, values: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    t = times - times[0]
    tm, vm = t.mean(), values.mean()
    num = ((t - tm) * (values - vm)).sum()
    den = ((t - tm) ** 2).sum()
    return float(num / den) if den > 0 else 0.0


def _burst_features(dates: pd.Series, elapsed_min: float) -> dict:
    if len(dates) < 2:
        return {"pause_max": 0.0, "n_bursts": 0, "act_var": 0.0}
    s = dates.sort_values()
    pauses = s.diff().dt.total_seconds().dropna() / 60
    pause_max = float(pauses.max())
    n_bursts  = int((pauses > BURST_PAUSE_MIN).sum())
    if elapsed_min >= 5:
        n_bins  = max(2, int(elapsed_min // 5))
        t_min   = (s - s.iloc[0]).dt.total_seconds() / 60
        counts  = s.groupby(pd.cut(t_min, bins=n_bins), observed=True).size()
        act_var = float(counts.var()) if len(counts) >= 2 else 0.0
    else:
        act_var = 0.0
    return {"pause_max": pause_max, "n_bursts": n_bursts, "act_var": act_var}

# ---------------------------------------------------------------------------
# Features d'un snapshot
# ---------------------------------------------------------------------------

def _snapshot(t, pre_start, first_cg, all_airport, alert_cg_inner, time_since_last_alert):
    t20 = t + pd.Timedelta(minutes=DECISION_WINDOW_MIN)

    # Historique [pre_start, t]
    hist          = all_airport[(all_airport["date"] >= pre_start) & (all_airport["date"] <= t)]
    cg_inner_hist = hist[(hist["type"] == "CG") & (hist["zone"] == "inner")]
    ic_inner_hist = hist[(hist["type"] == "IC") & (hist["zone"] == "inner")]
    cg_outer_hist = hist[(hist["type"] == "CG") & (hist["zone"] == "outer")]
    ic_outer_hist = hist[(hist["type"] == "IC") & (hist["zone"] == "outer")]

    # Fenêtre ]t, t+20]
    win          = all_airport[(all_airport["date"] > t) & (all_airport["date"] <= t20)]
    ic_inner_win = win[(win["type"] == "IC") & (win["zone"] == "inner")]
    cg_outer_win = win[(win["type"] == "CG") & (win["zone"] == "outer")]
    ic_outer_win = win[(win["type"] == "IC") & (win["zone"] == "outer")]

    # Label
    label   = int(len(alert_cg_inner[alert_cg_inner["date"] > t20]) == 0)
    elapsed = (t - first_cg).total_seconds() / 60

    # CG inner — historique
    n_cg_inner = len(cg_inner_hist)
    if n_cg_inner > 0:
        amps   = cg_inner_hist["amplitude"].abs().values
        dists  = cg_inner_hist["dist"].values
        times  = (cg_inner_hist["date"] - pre_start).dt.total_seconds().values / 60
        amp_max    = float(amps.max())
        amp_mean   = float(amps.mean())
        amp_trend  = _linear_slope(times, amps)
        dist_min   = float(dists.min())
        dist_mean  = float(dists.mean())
        dist_trend = _linear_slope(times, dists)
        burst      = _burst_features(cg_inner_hist["date"], elapsed)
    else:
        amp_max = amp_mean = amp_trend = 0.0
        dist_min = dist_mean = dist_trend = 0.0
        burst = {"pause_max": 0.0, "n_bursts": 0, "act_var": 0.0}

    # IC inner
    n_ic_inner  = len(ic_inner_hist)
    ratio_ic_cg = n_ic_inner / max(n_cg_inner, 1)

    return {
        "y":                     label,
        "elapsed":               round(elapsed, 2),
        "time_since_last_alert": round(time_since_last_alert, 1),
        "n_cg_inner":            n_cg_inner,
        "amp_max":               round(amp_max, 2),
        "amp_mean":              round(amp_mean, 2),
        "amp_trend":             round(amp_trend, 4),
        "dist_min":              round(dist_min, 2),
        "dist_mean":             round(dist_mean, 2),
        "dist_trend":            round(dist_trend, 4),
        "pause_max":             round(burst["pause_max"], 2),
        "n_bursts":              burst["n_bursts"],
        "act_var":               round(burst["act_var"], 2),
        "n_ic_inner":            n_ic_inner,
        "ratio_ic_cg":           round(ratio_ic_cg, 3),
        "n_ic_inner_win":        len(ic_inner_win),
        "n_cg_outer":            len(cg_outer_hist),
        "n_cg_outer_win":        len(cg_outer_win),
        "n_ic_outer":            len(ic_outer_hist),
        "n_ic_outer_win":        len(ic_outer_win),
    }

# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    records   = []
    alert_ids = (
        df[df["airport_alert_id"].notna()][["airport", "airport_alert_id"]]
        .drop_duplicates()
    )

    cg_inner_all = df[
        (df["type"] == "CG") & (df["zone"] == "inner") & df["airport_alert_id"].notna()
    ]
    alert_ends = (
        cg_inner_all.groupby(["airport", "airport_alert_id"])["date"]
        .max()
        .reset_index()
        .rename(columns={"date": "end_date"})
    )

    n_total = len(alert_ids)
    print(f"  Alertes à traiter : {n_total}")

    for i, (_, row) in enumerate(alert_ids.iterrows(), 1):
        if i % 100 == 0:
            print(f"  {i}/{n_total}...")

        airport  = row["airport"]
        alert_id = row["airport_alert_id"]

        alert_cg = df[
            (df["airport"] == airport) &
            (df["airport_alert_id"] == alert_id) &
            (df["type"] == "CG") &
            (df["zone"] == "inner")
        ].sort_values("date").reset_index(drop=True)

        if len(alert_cg) == 0:
            continue

        first_cg  = alert_cg["date"].iloc[0]
        last_cg   = alert_cg["date"].iloc[-1]
        pre_start = first_cg - pd.Timedelta(minutes=PRE_WINDOW_MIN)
        t_max     = last_cg  + pd.Timedelta(minutes=DECISION_WINDOW_MIN)

        all_airport = df[
            (df["airport"] == airport) &
            (df["date"] >= pre_start) &
            (df["date"] <= t_max)
        ]

        prev_ends = alert_ends[
            (alert_ends["airport"] == airport) &
            (alert_ends["airport_alert_id"] < alert_id)
        ]
        if len(prev_ends) > 0:
            tsla = (first_cg - prev_ends["end_date"].max()).total_seconds() / 60
        else:
            tsla = 99999.0

        n_cg = len(alert_cg)
        for idx in range(n_cg):
            t = alert_cg["date"].iloc[idx]
            if idx < n_cg - 1:
                gap = (alert_cg["date"].iloc[idx + 1] - t).total_seconds() / 60
                if gap <= DECISION_WINDOW_MIN:
                    continue

            feats = _snapshot(t, pre_start, first_cg, all_airport, alert_cg, tsla)
            feats["airport"]          = airport
            feats["airport_alert_id"] = alert_id
            feats["t"]                = t
            feats["t20"]              = t + pd.Timedelta(minutes=DECISION_WINDOW_MIN)
            records.append(feats)

    meta = ["airport", "airport_alert_id", "t", "t20", "y"]
    return pd.DataFrame(records)[meta + FEATURES]

# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/segment_alerts_all_airports_train.csv"
    print(f"Chargement de {path} ...")
    df = load_data(path)

    print("Calcul des features ...")
    features = compute_features(df)

    print(f"\n{'='*55}")
    print(f"Snapshots valides  : {len(features)}")
    print(f"Alertes couvertes  : {features['airport_alert_id'].nunique()}")
    print(f"Features           : {len(FEATURES)}")
    print(f"\nLabel :")
    print(features["y"].value_counts())
    print(f"\nPar aéroport :")
    print(features.groupby("airport")["y"].value_counts().unstack(fill_value=0))
    print(f"\nStatistiques features :")
    print(features[FEATURES].describe().round(2))

    out = "data/features_classification.parquet"
    features.to_parquet(out, index=False)
    print(f"\nSauvegardé dans {out}")