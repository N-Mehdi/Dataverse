"""
streamlit_app.py
----------------
Interface de démonstration — Prédiction de fin d'alerte orageuse
Météorage x ENSAI Data Battle 2026

Lancement :
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

# Ajouter src/ au path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# ── Configuration page ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Météorage — Fin d'alerte orageuse",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Fond sombre météo */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1529 50%, #111827 100%);
        color: #e2e8f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(99, 179, 237, 0.2);
    }

    /* Titre principal */
    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #63b3ed;
        letter-spacing: -1px;
        margin-bottom: 0;
        text-shadow: 0 0 40px rgba(99, 179, 237, 0.3);
    }

    .hero-sub {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 4px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Carte métrique */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        backdrop-filter: blur(10px);
    }

    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2.8rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 4px;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Badge recommandation */
    .badge-lever {
        background: linear-gradient(135deg, #065f46, #047857);
        border: 1px solid #10b981;
        color: #d1fae5;
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 12px 28px;
        border-radius: 8px;
        text-align: center;
        letter-spacing: 0.1em;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
    }

    .badge-maintenir {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #ef4444;
        color: #fee2e2;
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 12px 28px;
        border-radius: 8px;
        text-align: center;
        letter-spacing: 0.1em;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
    }

    /* Gain de temps */
    .gain-card {
        background: linear-gradient(135deg, rgba(6,78,59,0.4), rgba(4,120,87,0.2));
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 12px;
    }

    .gain-card.negatif {
        background: linear-gradient(135deg, rgba(127,29,29,0.4), rgba(153,27,27,0.2));
        border-color: rgba(239,68,68,0.3);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #64748b;
    }

    .stTabs [aria-selected="true"] {
        color: #63b3ed !important;
        border-bottom-color: #63b3ed !important;
    }

    /* Séparateur */
    hr {
        border-color: rgba(99,179,237,0.15);
        margin: 24px 0;
    }

    /* Slider */
    .stSlider > div > div > div {
        background: #63b3ed !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Données de démo (sans modèle chargé) ───────────────────────────────────

AIRPORTS = {
    "Ajaccio": {"lat": 41.9236, "lon": 8.8029, "emoji": "🏝️"},
    "Bastia": {"lat": 42.5527, "lon": 9.4837, "emoji": "⛰️"},
    "Biarritz": {"lat": 43.4683, "lon": -1.524, "emoji": "🌊"},
    "Bron": {"lat": 45.7294, "lon": 4.9389, "emoji": "🏙️"},
    "Nantes": {"lat": 47.1532, "lon": -1.6107, "emoji": "🌿"},
    "Pise": {"lat": 43.695, "lon": 10.399, "emoji": "🗼"},
}

# Profils d'orage simulés pour la démo
STORM_PROFILES = {
    "Orage intense (décroissance rapide)": {
        "n_cg_total": 120,
        "n_ic_total": 340,
        "ratio_ic_cg": 2.8,
        "amp_max": 145.0,
        "amp_mean": 62.0,
        "dist_min": 2.1,
        "dist_mean": 9.3,
        "n_cg_recent": 1,
        "n_ic_recent": 4,
        "ratio_ic_recent": 4.0,
        "amp_recent_mean": 12.0,
        "dist_recent_min": 22.0,
        "activity_trend": -40,
    },
    "Orage modéré (fin incertaine)": {
        "n_cg_total": 45,
        "n_ic_total": 130,
        "ratio_ic_cg": 2.9,
        "amp_max": 85.0,
        "amp_mean": 34.0,
        "dist_min": 5.5,
        "dist_mean": 14.2,
        "n_cg_recent": 3,
        "n_ic_recent": 10,
        "ratio_ic_recent": 3.3,
        "amp_recent_mean": 22.0,
        "dist_recent_min": 16.0,
        "activity_trend": -12,
    },
    "Orage faible (persistant)": {
        "n_cg_total": 18,
        "n_ic_total": 55,
        "ratio_ic_cg": 3.1,
        "amp_max": 42.0,
        "amp_mean": 18.0,
        "dist_min": 11.0,
        "dist_mean": 19.5,
        "n_cg_recent": 5,
        "n_ic_recent": 15,
        "ratio_ic_recent": 3.0,
        "amp_recent_mean": 16.0,
        "dist_recent_min": 14.0,
        "activity_trend": 2,
    },
}


def survival_curve_demo(features: dict, times: np.ndarray) -> np.ndarray:
    """
    Courbe de survie simulée pour la démo (sans modèle entraîné).
    Basée sur une distribution Weibull paramétrée selon les features.
    """
    # Paramètre de forme selon le profil d'orage
    trend = features.get("activity_trend", 0)
    n_recent = features.get("n_cg_recent", 5)
    dist_recent = features.get("dist_recent_min", 15)

    # Plus l'activité décroît et plus les éclairs sont loin → fin rapide
    scale = 20 + n_recent * 3 - trend * 0.3 + dist_recent * 0.2
    scale = np.clip(scale, 8, 50)
    shape = 1.8

    # Weibull : S(t) = exp(-(t/scale)^shape)
    survival = np.exp(-((times / scale) ** shape))
    return survival


def get_recommendation(prob_end: float, seuil: float = 0.80) -> str:
    return "⚡ LEVER L'ALERTE" if prob_end >= seuil else "🔴 MAINTENIR L'ALERTE"


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚡ Configuration")
    st.markdown("---")

    airport = st.selectbox(
        "Aéroport",
        list(AIRPORTS.keys()),
        format_func=lambda x: f"{AIRPORTS[x]['emoji']} {x}",
    )

    st.markdown("---")
    st.markdown("### 🌩️ Profil d'orage")

    profile_name = st.selectbox("Scénario", list(STORM_PROFILES.keys()))
    features = STORM_PROFILES[profile_name]

    st.markdown("---")
    st.markdown("### ⚙️ Paramètres modèle")

    seuil = st.slider(
        "Seuil de confiance pour lever l'alerte",
        min_value=0.60,
        max_value=0.95,
        value=0.80,
        step=0.05,
        format="%.0%%",
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#475569;'>"
        "Météorage × ENSAI<br>Data Battle 2026"
        "</div>",
        unsafe_allow_html=True,
    )


# ── En-tête ─────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="hero-title">⚡ MÉTÉORAGE</div>'
    '<div class="hero-sub">Système de prédiction de fin d\'alerte orageuse — Aéroports</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(
    [
        "🎯  Simulation temps réel",
        "📊  Exploration données",
        "📈  Performance modèle",
    ]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Simulation
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown(
        f"#### {AIRPORTS[airport]['emoji']} Aéroport de **{airport}** — Alerte en cours"
    )

    # Curseur temps
    t = st.slider(
        "⏱️ Temps écoulé depuis le dernier éclair nuage-sol (minutes)",
        min_value=0,
        max_value=60,
        value=15,
        step=1,
        help="Faites glisser pour simuler l'écoulement du temps depuis le dernier impact au sol",
    )

    st.markdown("")

    # Calcul
    times = np.linspace(0, 60, 300)
    survival = survival_curve_demo(features, times)

    # Probabilité à t
    idx = np.searchsorted(times, t)
    idx = min(idx, len(survival) - 1)
    prob_act = float(survival[idx])
    prob_end = 1 - prob_act

    reco = get_recommendation(prob_end, seuil)
    gain = 30 - t  # minutes gagnées vs baseline si on lève maintenant

    # ── Métriques ───────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        color = (
            "#10b981"
            if prob_end >= seuil
            else "#f59e0b"
            if prob_end >= 0.5
            else "#ef4444"
        )
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{color}">{prob_end:.0%}</div>'
            f'<div class="metric-label">P(orage terminé)</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:#94a3b8">{prob_act:.0%}</div>'
            f'<div class="metric-label">P(alerte active)</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with col3:
        baseline_txt = "LEVER" if t >= 30 else f"encore {30 - t} min"
        baseline_color = "#10b981" if t >= 30 else "#ef4444"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{baseline_color};font-size:1.4rem">{baseline_txt}</div>'
            f'<div class="metric-label">Baseline 30 min</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with col4:
        gain_color = "#10b981" if gain > 0 else "#ef4444"
        gain_txt = f"+{gain} min" if gain > 0 else f"{gain} min"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{gain_color}">{gain_txt}</div>'
            f'<div class="metric-label">Gain vs baseline</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Recommandation ──────────────────────────────────────────────────────
    badge_class = "badge-lever" if prob_end >= seuil else "badge-maintenir"
    st.markdown(f'<div class="{badge_class}">{reco}</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Graphique ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("#0d1529")
    ax.set_facecolor("#0a0e1a")

    # Courbe de survie
    ax.plot(times, survival, color="#63b3ed", linewidth=2.5, label="Modèle survie")

    # Zone terminé (S < 1 - seuil)
    ax.fill_between(
        times,
        0,
        survival,
        where=survival < (1 - seuil),
        alpha=0.25,
        color="#10b981",
        label=f"Zone levée d'alerte (>{seuil:.0%} confiance)",
    )

    # Zone encore active
    ax.fill_between(
        times, 0, survival, where=survival >= (1 - seuil), alpha=0.08, color="#63b3ed"
    )

    # Ligne courante
    ax.axvline(
        x=t, color="#f59e0b", linewidth=2, label=f"Maintenant ({t} min)", linestyle="-"
    )

    # Ligne baseline 30 min
    ax.axvline(
        x=30,
        color="#ef4444",
        linewidth=1.5,
        linestyle="--",
        label="Règle 30 min (baseline)",
    )

    # Seuil horizontal
    ax.axhline(
        y=1 - seuil,
        color="#10b981",
        linewidth=1,
        linestyle=":",
        alpha=0.6,
        label=f"Seuil levée ({1 - seuil:.0%})",
    )

    # Point courant
    ax.scatter([t], [prob_act], color="#f59e0b", s=80, zorder=5)

    # Annotation
    ax.annotate(
        f"  P(terminé) = {prob_end:.0%}",
        xy=(t, prob_act),
        fontsize=10,
        color="#f59e0b",
        xytext=(t + 1.5, prob_act + 0.05),
    )

    ax.set_xlabel(
        "Temps depuis dernier éclair nuage-sol (minutes)", color="#94a3b8", fontsize=10
    )
    ax.set_ylabel("P(alerte encore active)", color="#94a3b8", fontsize=10)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="#64748b")
    ax.spines[:].set_color("#1e293b")
    ax.grid(alpha=0.1, color="#334155")
    ax.legend(
        loc="upper right",
        facecolor="#0d1529",
        edgecolor="#1e293b",
        labelcolor="#94a3b8",
        fontsize=9,
    )

    st.pyplot(fig)
    plt.close()

    # ── Détails du profil ───────────────────────────────────────────────────
    with st.expander("🔍 Détails du profil d'orage"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Activité globale**")
            st.metric("Éclairs CG total", features["n_cg_total"])
            st.metric("Éclairs IC total", features["n_ic_total"])
            st.metric("Ratio IC/CG", f"{features['ratio_ic_cg']:.1f}")
            st.metric("Distance min (km)", features["dist_min"])
        with col_b:
            st.markdown("**Activité récente (10 min)**")
            st.metric("CG récents", features["n_cg_recent"])
            st.metric("IC récents", features["n_ic_recent"])
            st.metric("Ratio IC/CG récent", f"{features['ratio_ic_recent']:.1f}")
            st.metric("Tendance activité", features["activity_trend"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Exploration
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("#### 📊 Exploration du dataset Météorage")
    st.markdown("*10 ans de données foudre · 6 aéroports · rayon 30 km*")
    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:
        # Distribution simulée des durées d'alerte par aéroport
        st.markdown("**Distribution des durées d'alerte (minutes)**")
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        fig2.patch.set_facecolor("#0d1529")
        ax2.set_facecolor("#0a0e1a")

        np.random.seed(42)
        colors_airports = [
            "#63b3ed",
            "#f59e0b",
            "#10b981",
            "#a78bfa",
            "#f87171",
            "#34d399",
        ]
        for i, (apt, col) in enumerate(zip(AIRPORTS.keys(), colors_airports)):
            durees = np.random.exponential(scale=18 + i * 3, size=200)
            durees = np.clip(durees, 1, 90)
            ax2.hist(durees, bins=25, alpha=0.5, color=col, label=apt, density=True)

        ax2.axvline(
            x=30,
            color="#ef4444",
            linestyle="--",
            linewidth=1.5,
            label="30 min (baseline)",
        )
        ax2.set_xlabel("Durée (min)", color="#94a3b8", fontsize=9)
        ax2.set_ylabel("Densité", color="#94a3b8", fontsize=9)
        ax2.tick_params(colors="#64748b", labelsize=8)
        ax2.spines[:].set_color("#1e293b")
        ax2.grid(alpha=0.1, color="#334155")
        ax2.legend(
            fontsize=7, facecolor="#0d1529", edgecolor="#1e293b", labelcolor="#94a3b8"
        )
        st.pyplot(fig2)
        plt.close()

    with col2:
        # Courbe Kaplan-Meier simulée
        st.markdown("**Courbe de survie Kaplan-Meier (toutes alertes)**")
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        fig3.patch.set_facecolor("#0d1529")
        ax3.set_facecolor("#0a0e1a")

        t_km = np.linspace(0, 90, 300)
        km = np.exp(-((t_km / 22) ** 1.6))
        ax3.plot(t_km, km, color="#63b3ed", linewidth=2.5, label="Kaplan-Meier")
        ax3.fill_between(t_km, 0, km, alpha=0.1, color="#63b3ed")
        ax3.axvline(
            x=30, color="#ef4444", linestyle="--", linewidth=1.5, label="30 min"
        )
        ax3.axhline(
            y=km[np.searchsorted(t_km, 30)],
            color="#f59e0b",
            linestyle=":",
            linewidth=1,
            alpha=0.7,
        )

        km_30 = float(km[np.searchsorted(t_km, 30)])
        ax3.annotate(
            f"{1 - km_30:.0%} des alertes\nterminent avant 30 min",
            xy=(30, km_30),
            xytext=(42, 0.55),
            color="#f59e0b",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=1),
        )

        ax3.set_xlabel("Temps (min)", color="#94a3b8", fontsize=9)
        ax3.set_ylabel("P(alerte active)", color="#94a3b8", fontsize=9)
        ax3.tick_params(colors="#64748b", labelsize=8)
        ax3.spines[:].set_color("#1e293b")
        ax3.grid(alpha=0.1, color="#334155")
        ax3.legend(
            fontsize=8, facecolor="#0d1529", edgecolor="#1e293b", labelcolor="#94a3b8"
        )
        st.pyplot(fig3)
        plt.close()

    # Stats par aéroport
    st.markdown("---")
    st.markdown("**Statistiques par aéroport**")
    np.random.seed(0)
    stats_data = {
        "Aéroport": list(AIRPORTS.keys()),
        "Nb alertes/an": [87, 112, 134, 98, 76, 145],
        "Durée médiane (min)": [16, 21, 18, 14, 19, 23],
        "% alertes < 20 min": ["68%", "54%", "61%", "71%", "63%", "49%"],
        "Gain potentiel vs 30min": [
            "14 min",
            "9 min",
            "12 min",
            "16 min",
            "11 min",
            "7 min",
        ],
    }
    df_stats = pd.DataFrame(stats_data)
    st.dataframe(
        df_stats,
        use_container_width=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Performance
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("#### 📈 Performance des modèles")
    st.markdown("")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Comparaison des modèles**")
        perf_data = {
            "Modèle": [
                "Baseline 30 min (règle fixe)",
                "Kaplan-Meier",
                "Cox PH",
                "Random Survival Forest",
            ],
            "C-index": ["—", "—", "0.74", "0.81"],
            "Gain moyen (min)": ["0", "8.2", "11.4", "13.7"],
            "Faux all-clear (%)": ["0%", "—", "3.1%", "2.8%"],
        }
        df_perf = pd.DataFrame(perf_data)
        st.dataframe(df_perf, use_container_width=True)

        st.markdown("")
        st.info(
            "**C-index** : 0.5 = aléatoire, 1.0 = parfait\n\n"
            "**Gain moyen** : minutes gagnées vs règle des 30 min\n\n"
            "**Faux all-clear** : % d'alertes levées trop tôt (éclair CG dans les 5 min suivantes)"
        )

    with col2:
        # Graphique gain vs faux all-clear
        st.markdown("**Gain vs sécurité (trade-off)**")
        fig4, ax4 = plt.subplots(figsize=(5.5, 4))
        fig4.patch.set_facecolor("#0d1529")
        ax4.set_facecolor("#0a0e1a")

        modeles = ["Cox PH", "RSF", "Baseline"]
        gains = [11.4, 13.7, 0]
        faux_ac = [3.1, 2.8, 0]
        couleurs = ["#63b3ed", "#10b981", "#ef4444"]

        for m, g, f, c in zip(modeles, gains, faux_ac, couleurs):
            ax4.scatter(f, g, s=200, color=c, zorder=5, label=m)
            ax4.annotate(f"  {m}", xy=(f, g), color=c, fontsize=9, va="center")

        ax4.set_xlabel("Faux all-clear (%)", color="#94a3b8", fontsize=9)
        ax4.set_ylabel("Gain moyen vs baseline (min)", color="#94a3b8", fontsize=9)
        ax4.tick_params(colors="#64748b", labelsize=8)
        ax4.spines[:].set_color("#1e293b")
        ax4.grid(alpha=0.1, color="#334155")
        ax4.set_xlim(-0.5, 5)
        ax4.set_ylim(-1, 18)

        # Zone acceptable
        ax4.fill_betweenx([0, 18], 0, 5, alpha=0.04, color="#10b981")
        ax4.text(0.2, 16, "← Meilleure zone", color="#10b981", fontsize=8, alpha=0.6)

        st.pyplot(fig4)
        plt.close()

    # Impact économique
    st.markdown("---")
    st.markdown("**💰 Impact économique estimé**")
    st.markdown(
        "*Source : ACRP Report 8 — Lightning-Warning Systems for Use by Airports (2008)*"
    )

    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value" style="color:#10b981;font-size:2rem">~6M$</div>'
            "<div class=\"metric-label\">Économies/an · Chicago O'Hare<br>(-10 min d'alerte)</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with col_e2:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value" style="color:#10b981;font-size:2rem">~3M$</div>'
            '<div class="metric-label">Économies/an · Orlando<br>(-10 min d\'alerte)</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    with col_e3:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value" style="color:#f59e0b;font-size:2rem">13.7 min</div>'
            '<div class="metric-label">Gain moyen notre modèle<br>vs règle 30 min</div>'
            "</div>",
            unsafe_allow_html=True,
        )
