"""
World Cup 2026 — Social Media Sentiment Dashboard

Multi-tab Streamlit application that presents the full analysis pipeline
results in an interactive, business-friendly format.

Usage:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import TARGET_TEAMS, PROCESSED_DIR
from src.utils import load_dataframe, setup_logger

logger = setup_logger(__name__)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mundial 2026 — Sentiment Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color palette ───────────────────────────────────────────────────────────
COLORS = {
    "positive": "#2ECC71",
    "negative": "#E74C3C",
    "neutral": "#95A5A6",
    "background": "#F8F9FA",
    "text": "#2C3E50",
    "accent": "#3498DB",
}
TEAM_COLORS = {
    "Spain": "#C60B1E",
    "Argentina": "#75AADB",
    "Brazil": "#009739",
    "France": "#002395",
    "England": "#CF142B",
}


# ── Data loading (cached) ──────────────────────────────────────────────────


@st.cache_data(ttl=300)
def load_final_data() -> pd.DataFrame:
    """Load the final processed dataset from disk."""
    path = PROCESSED_DIR / "final.parquet"
    if path.exists():
        return load_dataframe(str(path))
    # Fallback: try earlier stages
    for name in ["topic_ner", "sentiment", "preprocessed"]:
        path = PROCESSED_DIR / f"{name}.parquet"
        if path.exists():
            df = load_dataframe(str(path))
            if not df.empty:
                return df
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_shift_data() -> pd.DataFrame:
    """Load sentiment shift analysis results."""
    path = PROCESSED_DIR / "sentiment_shift.parquet"
    if path.exists():
        return load_dataframe(str(path))
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_matches() -> pd.DataFrame:
    """Load match results."""
    path = PROCESSED_DIR / "match_results.parquet"
    if path.exists():
        return load_dataframe(str(path))
    return pd.DataFrame()


# ── Helper functions ────────────────────────────────────────────────────────


def filter_dataframe(
    df: pd.DataFrame,
    teams: list,
    languages: list,
    date_range: tuple,
) -> pd.DataFrame:
    """Apply sidebar filters to the DataFrame."""
    mask = pd.Series(True, index=df.index)

    if teams and "teams" in df.columns:
        team_mask = df["teams"].apply(
            lambda x: any(t in str(x) for t in teams) if pd.notna(x) else False,
        )
        mask &= team_mask

    if languages and "language" in df.columns:
        mask &= df["language"].isin(languages)

    if date_range and "created_utc" in df.columns:
        dates = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")
        mask &= dates >= pd.Timestamp(date_range[0], tz="UTC")
        mask &= dates <= pd.Timestamp(date_range[1], tz="UTC")

    return df[mask].copy()


def compute_kpi(df: pd.DataFrame, period: str = "all") -> dict:
    """Compute summary KPIs for the current filter state."""
    kpis = {}
    kpis["total_comments"] = len(df)
    if "sentiment_label" in df.columns:
        kpis["pos_pct"] = (df["sentiment_label"] == "positive").mean()
        kpis["neg_pct"] = (df["sentiment_label"] == "negative").mean()
        kpis["neu_pct"] = (df["sentiment_label"] == "neutral").mean()
    return kpis


# ── Sidebar ────────────────────────────────────────────────────────────────

st.sidebar.title("Mundial 2026")
st.sidebar.markdown("Análisis de Sentimiento en Redes Sociales")
st.sidebar.markdown("---")

# Filters
selected_teams = st.sidebar.multiselect(
    "Selecciones",
    options=TARGET_TEAMS,
    default=TARGET_TEAMS,
)

selected_languages = st.sidebar.multiselect(
    "Idioma",
    options=["es", "en"],
    default=["es", "en"],
    format_func=lambda x: "Español" if x == "es" else "English",
)

# Date range — infer from data
df_full = load_final_data()
min_date = None
max_date = None
if not df_full.empty and "created_utc" in df_full.columns:
    dates = pd.to_datetime(df_full["created_utc"], utc=True, errors="coerce")
    min_date = dates.min().date() if not dates.isna().all() else None
    max_date = dates.max().date() if not dates.isna().all() else None

if min_date and max_date:
    date_range = st.sidebar.date_input(
        "Rango de fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
else:
    date_range = None

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Datos**: YouTube (FIFA, ESPN, FOX Soccer)  \n"
    "**Modelo**: pysentimiento + RoBERTa  \n"
    "**Temas**: BERTopic + NER"
)

# ── Tabs ───────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Resumen Ejecutivo",
    "📈 Evolución Temporal",
    "🏆 Comparativa",
    "📋 Temas",
    "⚽ Impacto de Partidos",
])

# ── Data ────────────────────────────────────────────────────────────────────
df = filter_dataframe(
    df_full,
    teams=selected_teams,
    languages=selected_languages,
    date_range=date_range,
)
shift_df = load_shift_data()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Executive Summary
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Resumen Ejecutivo")

    if df.empty:
        st.info(
            "No hay datos disponibles. Ejecuta el pipeline primero:\n"
            "`python -m src.pipeline`"
        )
        st.stop()

    kpis = compute_kpi(df)

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Comentarios analizados", f"{kpis['total_comments']:,}")
    with col2:
        st.metric(
            "Sentimiento Positivo",
            f"{kpis['pos_pct']:.1%}",
            delta=f"{kpis['pos_pct'] - 0.33:.1%} vs. aleatorio",
            delta_color="normal",
        )
    with col3:
        st.metric(
            "Sentimiento Negativo",
            f"{kpis['neg_pct']:.1%}",
            delta=f"{kpis['neg_pct'] - 0.33:.1%} vs. aleatorio",
            delta_color="inverse",
        )
    with col4:
        st.metric(
            "Comentarios con emojis",
            f"{(df['n_emojis'] > 0).mean():.1%}" if "n_emojis" in df.columns else "N/A",
        )

    st.markdown("---")

    # Sentiment donut chart
    col_a, col_b = st.columns(2)
    with col_a:
        if "sentiment_label" in df.columns:
            fig = go.Figure(data=[
                go.Pie(
                    labels=["Positivo", "Negativo", "Neutral"],
                    values=[
                        kpis["pos_pct"] * 100,
                        kpis["neg_pct"] * 100,
                        kpis["neu_pct"] * 100,
                    ],
                    marker_colors=[
                        COLORS["positive"],
                        COLORS["negative"],
                        COLORS["neutral"],
                    ],
                    hole=0.4,
                    textinfo="label+percent",
                )
            ])
            fig.update_layout(title="Distribución General de Sentimiento")
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if "teams" in df.columns and "sentiment_label" in df.columns:
            team_sent = (
                df.groupby("teams")["sentiment_label"]
                .value_counts(normalize=True)
                .unstack()
            )
            if not team_sent.empty:
                fig = go.Figure()
                for col_name in ["positive", "negative", "neutral"]:
                    if col_name in team_sent.columns:
                        fig.add_trace(go.Bar(
                            name=col_name.capitalize(),
                            x=team_sent.index,
                            y=team_sent[col_name],
                            marker_color=COLORS.get(col_name, "#999"),
                        ))
                fig.update_layout(
                    title="Sentimiento por Selección",
                    barmode="group",
                    xaxis_title="Selección",
                    yaxis_title="Proporción",
                )
                st.plotly_chart(fig, use_container_width=True)

    # Topic overview
    if "topic_label" in df.columns:
        st.subheader("Temas Principales")
        topic_counts = (
            df["topic_label"]
            .value_counts()
            .head(8)
            .reset_index()
        )
        topic_counts.columns = ["Tema", "Comentarios"]
        fig = px.bar(
            topic_counts,
            x="Comentarios",
            y="Tema",
            orientation="h",
            title="Top 8 Temas",
            color="Comentarios",
            color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Time Evolution
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Evolución Temporal del Sentimiento")
    st.markdown(
        "La línea discontinua muestra la media móvil de 3 días. "
        "Las líneas verticales marcan los partidos de cada selección."
    )

    if df.empty or "created_utc" not in df.columns:
        st.warning("Datos insuficientes para evolución temporal.")
    else:
        df_plot = df.copy()
        df_plot["date"] = pd.to_datetime(
            df_plot["created_utc"], utc=True,
        ).dt.date

        team_for_plot = st.selectbox(
            "Selecciona una selección",
            options=["Todas"] + TARGET_TEAMS,
        )

        if team_for_plot != "Todas":
            df_plot = df_plot[
                df_plot["teams"].str.contains(team_for_plot, case=False, na=False)
            ]

        daily = (
            df_plot.groupby("date")["sentiment_label"]
            .value_counts(normalize=True)
            .unstack()
        )
        daily = daily.fillna(0)

        if not daily.empty:
            fig = go.Figure()

            for col_name, color_key in [
                ("positive", "positive"),
                ("negative", "negative"),
                ("neutral", "neutral"),
            ]:
                if col_name in daily.columns:
                    fig.add_trace(go.Scatter(
                        x=daily.index,
                        y=daily[col_name],
                        mode="lines",
                        name=col_name.capitalize(),
                        line=dict(color=COLORS[color_key], width=1),
                        opacity=0.5,
                    ))
                    # 3-day rolling average
                    rolling = daily[col_name].rolling(3, min_periods=1).mean()
                    fig.add_trace(go.Scatter(
                        x=daily.index,
                        y=rolling,
                        mode="lines",
                        name=f"{col_name.capitalize()} (media 3d)",
                        line=dict(color=COLORS[color_key], width=2.5, dash="dot"),
                    ))

            # Match annotations
            matches_df = load_matches()
            if not matches_df.empty and team_for_plot != "Todas":
                for _, mrow in matches_df.iterrows():
                    mdate = pd.to_datetime(mrow["utc_date"]).date()
                    if min(daily.index) <= mdate <= max(daily.index):
                        fig.add_vline(
                            x=mdate,
                            line_width=1,
                            line_dash="dash",
                            line_color="gray",
                            opacity=0.5,
                        )

            fig.update_layout(
                title=f"Evolución del Sentimiento{ ' — ' + team_for_plot if team_for_plot != 'Todas' else ''}",
                xaxis_title="Fecha",
                yaxis_title="Proporción",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Team Comparison
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Comparativa entre Selecciones")

    if df.empty or "teams" not in df.columns:
        st.warning("Datos insuficientes para comparativa.")
    else:
        # Radar chart: average metrics per team
        metrics = []
        if "sentiment_positive" in df.columns:
            metrics += ["sentiment_positive", "sentiment_negative", "sentiment_neutral"]
        if "n_emojis" in df.columns:
            metrics.append("n_emojis")

        if metrics:
            radar_df = df.groupby("teams")[metrics].mean().reset_index()

            fig = go.Figure()
            for _, row in radar_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row[m] for m in metrics],
                    theta=metrics,
                    fill="toself",
                    name=row["teams"],
                    line_color=TEAM_COLORS.get(row["teams"], "#3498DB"),
                ))
            fig.update_layout(
                title="Perfil Comparativo por Selección",
                polar=dict(radialaxis=dict(visible=True)),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Grouped bar: sentiment by team
        st.subheader("Distribución de Sentimiento por Selección")
        team_sent = (
            df.groupby("teams")["sentiment_label"]
            .value_counts(normalize=True)
            .unstack()
        )
        fig = go.Figure()
        for col_name in ["positive", "negative", "neutral"]:
            if col_name in team_sent.columns:
                fig.add_trace(go.Bar(
                    name=col_name.capitalize(),
                    x=team_sent.index,
                    y=team_sent[col_name],
                    marker_color=COLORS[col_name],
                ))
        fig.update_layout(
            barmode="group",
            xaxis_title="Selección",
            yaxis_title="Proporción",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Sentiment polarity box plot
        if "sentiment_positive" in df.columns:
            st.subheader("Distribución de Puntuación Positiva")
            fig = px.box(
                df,
                x="teams",
                y="sentiment_positive",
                color="teams",
                color_discrete_map=TEAM_COLORS,
                points="outliers",
            )
            fig.update_layout(xaxis_title="", yaxis_title="Puntuación Positiva")
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: Topics
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Análisis de Temas")
    st.markdown(
        "Temas extraídos con BERTopic (modelo multilingüe). "
        "Cada comentario se asigna al tema más probable. "
        "Los temas se etiquetan automáticamente con sus 3 palabras más representativas."
    )

    if "topic_label" not in df.columns:
        st.warning("Temas no disponibles. Ejecuta topic modeling primero.")
    else:
        col_t1, col_t2 = st.columns([2, 1])

        with col_t1:
            topic_counts = (
                df["topic_label"]
                .value_counts()
                .reset_index()
            )
            topic_counts.columns = ["Tema", "Comentarios"]

            fig = px.bar(
                topic_counts.head(15),
                x="Comentarios",
                y="Tema",
                orientation="h",
                title="Distribución de Temas",
                color="Comentarios",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        with col_t2:
            # Topic × Sentiment heatmap
            cross = pd.crosstab(
                df["topic_label"],
                df["sentiment_label"],
                normalize="index",
            )
            if not cross.empty:
                fig = px.imshow(
                    cross,
                    text_auto=".0%",
                    color_continuous_scale="RdBu_r",
                    title="Tema vs. Sentimiento",
                    aspect="auto",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        # Topic evolution over time
        if "created_utc" in df.columns:
            st.subheader("Evolución de Temas en el Tiempo")
            df_time = df.copy()
            df_time["date"] = pd.to_datetime(
                df_time["created_utc"], utc=True,
            ).dt.date

            topic_freq = (
                df_time.groupby(["date", "topic_label"])
                .size()
                .reset_index(name="count")
            )

            top_topics = (
                topic_freq.groupby("topic_label")["count"]
                .sum()
                .nlargest(6)
                .index
            )
            topic_freq = topic_freq[topic_freq["topic_label"].isin(top_topics)]

            fig = px.line(
                topic_freq,
                x="date",
                y="count",
                color="topic_label",
                title="Frecuencia de Temas Destacados en el Tiempo",
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Brand mentions
        if "brands_mentioned" in df.columns:
            st.subheader("Menciones de Marcas")
            brand_series = (
                df[df["brands_mentioned"] != ""]["brands_mentioned"]
                .str.split(",")
                .explode()
                .str.strip()
                .value_counts()
            )
            if not brand_series.empty:
                fig = px.bar(
                    brand_series.head(10),
                    orientation="h",
                    title="Top 10 Marcas Mencionadas",
                    color=brand_series.head(10).values,
                    color_continuous_scale="Blues",
                )
                fig.update_layout(
                    xaxis_title="Menciones",
                    yaxis_title="",
                    yaxis={"categoryorder": "total ascending"},
                )
                st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: Match Impact
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Impacto de Resultados Deportivos en el Sentimiento")
    st.markdown(
        "Para cada partido, se compara el sentimiento en ventanas de "
        "24h antes vs. 24h después. La prueba de Mann-Whitney U evalúa "
        "si la diferencia es estadísticamente significativa (p < 0.05)."
    )

    if shift_df.empty:
        st.info(
            "No hay datos de impacto de partidos disponibles. "
            "Asegúrate de que football-data.org devuelve resultados "
            "y de que los comentarios están dentro de las ventanas temporales."
        )
    else:
        # Filter by team
        team_shift = st.selectbox(
            "Selecciona selección",
            options=shift_df["team"].unique(),
        )
        team_data = shift_df[shift_df["team"] == team_shift]

        st.subheader(f"Diferencia de Sentimiento — {team_shift}")

        # Before/after bars
        fig = go.Figure()
        for result in team_data["result"].unique():
            rdata = team_data[team_data["result"] == result]
            if rdata.empty:
                continue

            fig.add_trace(go.Bar(
                name=f"{result} (antes)",
                x=[f"{result}"],
                y=rdata["pre_pos_pct"],
                marker_color=COLORS["positive"],
                opacity=0.5,
                legendgroup=result,
            ))
            fig.add_trace(go.Bar(
                name=f"{result} (después)",
                x=[f"{result}"],
                y=rdata["post_pos_pct"],
                marker_color=COLORS["positive"],
                opacity=1.0,
                legendgroup=result,
            ))

        fig.update_layout(
            barmode="group",
            title="% Positivo Antes vs. Después del Partido",
            xaxis_title="Resultado",
            yaxis_title="Proporción Positivo",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        st.subheader("Resultados Detallados")
        display_cols = [
            "team", "result", "n_pre", "n_post",
            "pre_pos_pct", "post_pos_pct",
            "pre_neg_pct", "post_neg_pct",
            "p_value", "significant",
        ]
        display_cols = [c for c in display_cols if c in team_data.columns]
        st.dataframe(
            team_data[display_cols].style.applymap(
                lambda x: "background-color: #d4edda" if x is True else (
                    "background-color: #f8d7da" if x is False else ""
                ),
                subset=["significant"],
            ),
            use_container_width=True,
        )

        # Interpretation
        st.markdown("---")
        st.subheader("Interpretación")
        sig_results = shift_df[shift_df["significant"] == True]
        if not sig_results.empty:
            st.success(
                "Se detectaron cambios estadísticamente significativos "
                "en los siguientes casos:"
            )
            for _, row in sig_results.iterrows():
                direction = (
                    "mejoró" if row["post_pos_pct"] > row["pre_pos_pct"]
                    else "empeoró"
                )
                st.markdown(
                    f"- **{row['team']}** tras {row['result']}: "
                    f"sentimiento {direction} "
                    f"(p = {row['p_value']:.4f})"
                )
        else:
            st.info(
                "Aún no se detectan cambios estadísticamente significativos "
                "(p < 0.05). Esto puede deberse a que el torneo no ha "
                "comenzado o a que el volumen de datos es insuficiente."
            )

# ── Footer ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Autor**: Pablo Huidobro García  \n"
    "Proyecto: [GitHub](https://github.com/your-username/"
    "mundial2026-sentiment-analysis)"
)
