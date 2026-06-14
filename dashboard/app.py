"""
World Cup 2026 — Social Media Sentiment Dashboard

Multi-tab Streamlit application built on top of the full processed
dataset (comentarios_topics_ner).  Requires notebooks 01–04 to have
been executed at least once.

Usage:
    streamlit run dashboard/app.py
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.dashboard_data import load_dashboard_data, load_match_results  # noqa: E402
from src.results_api import get_pre_post_windows  # noqa: E402

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mundial 2026 — Sentiment Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color palette ───────────────────────────────────────────────────────────
POS_COLOR = "#2ECC71"
NEG_COLOR = "#E74C3C"
NEU_COLOR = "#95A5A6"
BG_COLOR = "#F8F9FA"
ACCENT = "#3498DB"
SENT_COLORS = {"POS": POS_COLOR, "NEU": NEU_COLOR, "NEG": NEG_COLOR}

# ── Load data (cached) ──────────────────────────────────────────────────────

df_full = load_dashboard_data()
match_results = load_match_results()

# ── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("Mundial 2026")
st.sidebar.markdown("Analisis de Sentimiento en Redes Sociales")
st.sidebar.markdown("---")

# Team filter — populate from actual data, default to all available
available_teams = (
    sorted(df_full["search_team"].dropna().unique()) if not df_full.empty else []
)
selected_teams = st.sidebar.multiselect(
    "Selecciones",
    options=available_teams,
    default=available_teams,
)

# Language filter
selected_languages = st.sidebar.multiselect(
    "Idioma",
    options=["es", "en"],
    default=["es", "en"],
    format_func=lambda x: "Espanol" if x == "es" else "English",
)

# Date range
min_date, max_date = None, None
if not df_full.empty and "published_at" in df_full.columns:
    dates = pd.to_datetime(df_full["published_at"], utc=True, errors="coerce")
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

# ── Filtering ───────────────────────────────────────────────────────────────

df = df_full.copy()
if not df.empty:
    if selected_teams:
        df = df[df["search_team"].isin(selected_teams)]
    if selected_languages:
        df = df[df["language"].isin(selected_languages)]
    if date_range and len(date_range) == 2 and "published_at" in df.columns:
        dts = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        lo = pd.Timestamp(date_range[0], tz="UTC")
        hi = pd.Timestamp(date_range[1], tz="UTC")
        df = df[(dts >= lo) & (dts <= hi)]

# ── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Resumen Ejecutivo",
        "Evolucion Temporal",
        "Comparativa",
        "Temas",
        "Impacto de Partidos",
    ]
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Resumen ejecutivo
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Resumen Ejecutivo")

    if df.empty:
        st.info(
            "No hay datos procesados todavia. "
            "Ejecuta los notebooks 01-04 para generar los datos."
        )
        st.stop()

    n_total = len(df)
    pos_pct = (df["sentiment_bert"] == "POS").mean()
    neg_pct = (df["sentiment_bert"] == "NEG").mean()
    neu_pct = (df["sentiment_bert"] == "NEU").mean()

    # KPI row (deltas only if we have enough data for a split)
    half = n_total // 2
    if half > 10 and "published_at" in df.columns:
        dts = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        first_half = df.iloc[:half]
        second_half = df.iloc[half:]
        delta_pos = (second_half["sentiment_bert"] == "POS").mean() - (
            first_half["sentiment_bert"] == "POS"
        ).mean()
        delta_neg = (second_half["sentiment_bert"] == "NEG").mean() - (
            first_half["sentiment_bert"] == "NEG"
        ).mean()
    else:
        delta_pos = None
        delta_neg = None

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Comentarios analizados", f"{n_total:,}")
    with col2:
        st.metric(
            "Sentimiento Positivo",
            f"{pos_pct:.1%}",
            delta=f"{delta_pos:+.1%}" if delta_pos is not None else None,
            delta_color="normal",
        )
    with col3:
        st.metric(
            "Sentimiento Negativo",
            f"{neg_pct:.1%}",
            delta=f"{delta_neg:+.1%}" if delta_neg is not None else None,
            delta_color="inverse",
        )
    with col4:
        emoji_pct = (df["n_emojis"] > 0).mean() if "n_emojis" in df.columns else 0
        st.metric("Con emojis", f"{emoji_pct:.1%}")

    # Auto-generated summary
    teams_str = ", ".join(sorted(df["search_team"].unique()))
    top_neg_topic = ""
    if "topic_label" in df.columns and "sentiment_bert" in df.columns:
        neg_topic_ct = (
            df[df["sentiment_bert"] == "NEG"]
            .groupby("topic_label")
            .size()
            .sort_values(ascending=False)
        )
        neg_topic_ct = neg_topic_ct[neg_topic_ct.index != "Outliers / Other"]
        if not neg_topic_ct.empty:
            top_neg_topic = neg_topic_ct.index[0]
            top_neg_pct = neg_topic_ct.iloc[0] / (df["sentiment_bert"] == "NEG").sum()
            top_neg_topic = f", principalmente relacionado con **{top_neg_topic}** ({top_neg_pct:.0%} de los negativos)"
        else:
            top_neg_topic = "."

    st.markdown(
        f"De los **{n_total:,}** comentarios analizados sobre **{teams_str}**, "
        f"el **{neg_pct:.1%}** expresa sentimiento negativo{top_neg_topic} "
        f"El **{pos_pct:.1%}** es positivo y el **{neu_pct:.1%}** neutral."
    )

    st.markdown("---")

    # Sentiment donut
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Positivo", "Negativo", "Neutral"],
                    values=[pos_pct * 100, neg_pct * 100, neu_pct * 100],
                    marker_colors=[POS_COLOR, NEG_COLOR, NEU_COLOR],
                    hole=0.4,
                    textinfo="label+percent",
                )
            ]
        )
        fig.update_layout(title="Distribucion General de Sentimiento")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if "search_team" in df.columns:
            team_sent = (
                df.groupby("search_team")["sentiment_bert"]
                .value_counts(normalize=True)
                .unstack()
            )
            if not team_sent.empty:
                fig = go.Figure()
                for col_name in ["POS", "NEU", "NEG"]:
                    if col_name in team_sent.columns:
                        fig.add_trace(
                            go.Bar(
                                name=col_name,
                                x=team_sent.index,
                                y=team_sent[col_name],
                                marker_color=SENT_COLORS[col_name],
                            )
                        )
                fig.update_layout(
                    title="Sentimiento por Seleccion",
                    barmode="group",
                    xaxis_title="Seleccion",
                    yaxis_title="Proporcion",
                )
                st.plotly_chart(fig, use_container_width=True)

    # Topic overview on executive summary
    if "topic_label" in df.columns:
        st.subheader("Temas Principales")
        topic_counts = df["topic_label"].value_counts().head(8).reset_index()
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

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Evolucion temporal
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Evolucion Temporal del Sentimiento")

    if df.empty or "published_at" not in df.columns:
        st.warning("Datos insuficientes para evolucion temporal.")
    else:
        df_plot = df.copy()
        df_plot["date"] = pd.to_datetime(df_plot["published_at"], utc=True)

        # Dynamic granularity: hourly if span <= 3 days, daily otherwise
        span = (df_plot["date"].max() - df_plot["date"].min()).total_seconds()
        if span <= 3 * 86400:
            freq_label = "hour"
            df_plot["period"] = df_plot["date"].dt.floor("h")
        else:
            freq_label = "day"
            df_plot["period"] = df_plot["date"].dt.floor("D")

        team_for_plot = st.selectbox(
            "Selecciona una seleccion",
            options=["Todas"] + selected_teams,
        )

        if team_for_plot != "Todas":
            df_plot = df_plot[df_plot["search_team"] == team_for_plot]

        daily = (
            df_plot.groupby("period")["sentiment_bert"]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
        )

        if not daily.empty:
            fig = go.Figure()
            for col_name in ["POS", "NEU", "NEG"]:
                if col_name in daily.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=daily.index,
                            y=daily[col_name],
                            mode="lines",
                            name=col_name,
                            line=dict(color=SENT_COLORS[col_name], width=2),
                        )
                    )

            # Match annotations
            mr = match_results
            if not mr.empty:
                # Filter to teams in current selection
                if team_for_plot != "Todas":
                    mr = mr[mr["team"] == team_for_plot]
                for _, mrow in mr.iterrows():
                    mdate = pd.to_datetime(mrow["match_date"])
                    if mdate.tz is None:
                        mdate = mdate.tz_localize("UTC")
                    if daily.index.min() <= mdate <= daily.index.max():
                        outcome = mrow["outcome"]
                        opp = mrow["opponent"]
                        score = mrow["score"]
                        fig.add_vline(
                            x=mdate,
                            line_width=1.5,
                            line_dash="dash",
                            line_color="gray",
                            opacity=0.6,
                            annotation_text=f"{outcome} vs {opp} ({score})",
                            annotation_position="top",
                        )

            fig.update_layout(
                title=f"Evolucion del Sentimiento{' — ' + team_for_plot if team_for_plot != 'Todas' else ''}",
                xaxis_title="Fecha" if freq_label == "day" else "Fecha (hora)",
                yaxis_title="Proporcion",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay suficientes datos para la escala temporal seleccionada.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Comparativa entre selecciones
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Comparativa entre Selecciones")

    if df.empty or "search_team" not in df.columns:
        st.info(
            "No hay datos procesados todavia. "
            "Ejecuta los notebooks 01-04 para generar los datos."
        )
        st.stop()

    unique_teams = df["search_team"].unique()
    if len(unique_teams) <= 1:
        st.info(
            "Actualmente solo hay datos para una seleccion. "
            "La comparativa se enriquecera a medida que la recoleccion "
            "diaria incorpore mas equipos."
        )

    # Grouped bar: sentiment distribution per team
    st.subheader("Distribucion de Sentimiento por Seleccion")
    team_sent = (
        df.groupby("search_team")["sentiment_bert"]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )
    fig = go.Figure()
    for col_name in ["POS", "NEU", "NEG"]:
        if col_name in team_sent.columns:
            fig.add_trace(
                go.Bar(
                    name=col_name,
                    x=team_sent.index,
                    y=team_sent[col_name],
                    marker_color=SENT_COLORS[col_name],
                )
            )
    fig.update_layout(
        barmode="group",
        xaxis_title="Seleccion",
        yaxis_title="Proporcion",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart (only if 2+ teams)
    if len(unique_teams) >= 2:
        st.subheader("Perfil Comparativo")
        metrics_agg = {
            "POS_pct": (df["sentiment_bert"] == "POS").mean(),
            "NEG_pct": (df["sentiment_bert"] == "NEG").mean(),
            "NEU_pct": (df["sentiment_bert"] == "NEU").mean(),
        }
        radar_df = (
            df.groupby("search_team")["sentiment_bert"]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
        )
        all_dimensions = [c for c in ["POS", "NEU", "NEG"] if c in radar_df.columns]

        fig = go.Figure()
        for team in radar_df.index:
            fig.add_trace(
                go.Scatterpolar(
                    r=[radar_df.loc[team, d] for d in all_dimensions],
                    theta=[
                        {"POS": "Positivo", "NEU": "Neutral", "NEG": "Negativo"}[d]
                        for d in all_dimensions
                    ],
                    fill="toself",
                    name=str(team),
                )
            )
        fig.update_layout(
            title="Perfil Comparativo por Seleccion",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Basic stats table
    st.subheader("Estadisticas por Seleccion")
    stats_list = []
    for team in unique_teams:
        tdf = df[df["search_team"] == team]
        stats_list.append(
            {
                "Seleccion": team,
                "Comentarios": len(tdf),
                "Positivo": f"{(tdf['sentiment_bert'] == 'POS').mean():.1%}",
                "Negativo": f"{(tdf['sentiment_bert'] == 'NEG').mean():.1%}",
                "Neutral": f"{(tdf['sentiment_bert'] == 'NEU').mean():.1%}",
                "% Emojis": f"{(tdf['n_emojis'] > 0).mean():.1%}"
                if "n_emojis" in tdf.columns
                else "N/A",
            }
        )
    st.dataframe(pd.DataFrame(stats_list), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Temas
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Analisis de Temas")

    if "topic_label" not in df.columns:
        st.warning(
            "Temas no disponibles. Ejecuta topic modeling primero (notebook 04)."
        )
        st.stop()

    st.markdown(
        "Temas extraidos con BERTopic (modelo multilinguee). "
        "Los temas se etiquetan con sus 3 palabras mas representativas."
    )

    # Topic count bar (top 10 excluding -1 or labeled)
    topic_counts = df["topic_label"].value_counts().reset_index()
    topic_counts.columns = ["Tema", "Comentarios"]
    # Separate outlier for clarity
    outlier_rows = topic_counts[topic_counts["Tema"] == "Outliers / Other"]
    main_rows = topic_counts[topic_counts["Tema"] != "Outliers / Other"].head(10)
    plot_topics = pd.concat([main_rows, outlier_rows], ignore_index=True)

    fig = px.bar(
        plot_topics,
        x="Comentarios",
        y="Tema",
        orientation="h",
        title="Distribucion de Temas (top 10 + Sin clasificar)",
        color="Comentarios",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    # Topic x sentiment stacked bar
    st.subheader("Sentimiento por Tema")
    ct = pd.crosstab(df["topic_label"], df["sentiment_bert"], normalize="index")
    ct = ct[[c for c in ["POS", "NEU", "NEG"] if c in ct.columns]]

    fig = go.Figure()
    for col_name in ["POS", "NEU", "NEG"]:
        if col_name in ct.columns:
            fig.add_trace(
                go.Bar(
                    name=col_name,
                    y=ct.index,
                    x=ct[col_name],
                    orientation="h",
                    marker_color=SENT_COLORS[col_name],
                )
            )
    fig.update_layout(
        barmode="stack",
        title="Proporcion de Sentimiento por Tema",
        xaxis_title="Proporcion",
        yaxis_title="",
        height=max(300, len(ct) * 25),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Topic selector — show sample comments
    topic_list = sorted(df["topic_label"].unique())
    selected_topic = st.selectbox(
        "Selecciona un tema para ver comentarios de ejemplo", topic_list
    )
    topic_samples = df[df["topic_label"] == selected_topic][
        ["text_clean", "sentiment_bert", "language", "search_team"]
    ].head(10)
    st.dataframe(topic_samples, use_container_width=True)

    # Entities table
    st.subheader("Jugadores mas mencionados")
    if "players_mentioned" in df.columns:
        all_players = df["players_mentioned"].str.split(",").explode().str.strip()
        all_players = all_players[all_players != ""]
        if not all_players.empty:
            player_counts = Counter(all_players)
            player_df = pd.DataFrame(
                player_counts.most_common(15), columns=["Jugador", "Menciones"]
            )
            # Sentiment breakdown per player
            player_sent_rows = []
            for player in player_df["Jugador"].head(10):
                mask = df["players_mentioned"].str.contains(player, na=False)
                subset = df[mask]
                player_sent_rows.append(
                    {
                        "Jugador": player,
                        "Menciones": mask.sum(),
                        "POS": f"{(subset['sentiment_bert'] == 'POS').mean():.0%}",
                        "NEU": f"{(subset['sentiment_bert'] == 'NEU').mean():.0%}",
                        "NEG": f"{(subset['sentiment_bert'] == 'NEG').mean():.0%}",
                    }
                )
            st.dataframe(pd.DataFrame(player_sent_rows), use_container_width=True)
        else:
            st.info("No se detectaron jugadores en los datos actuales.")
    else:
        st.info("Columna de jugadores no disponible.")

    # Brands
    if "brands_mentioned" in df.columns:
        st.subheader("Marcas patrocinadoras")
        all_brands = df["brands_mentioned"].str.split(",").explode().str.strip()
        all_brands = all_brands[all_brands != ""]
        if not all_brands.empty():
            brand_counts = Counter(all_brands)
            brand_df = pd.DataFrame(
                brand_counts.most_common(10), columns=["Marca", "Menciones"]
            )
            st.dataframe(brand_df, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — Impacto de resultados
# ═════════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("Impacto de Resultados Deportivos en el Sentimiento")

    mr = match_results
    if mr.empty:
        st.info(
            "No hay datos de resultados disponibles. "
            "Asegurate de que FOOTBALL_DATA_API_KEY esta configurada "
            "y los partidos han finalizado."
        )
        st.stop()

    if df.empty:
        st.info(
            "No hay datos procesados todavia. "
            "Ejecuta los notebooks 01-04 para generar los datos."
        )
        st.stop()

    st.markdown(
        "Para cada partido finalizado, se compara el sentimiento en ventanas "
        "de 24h antes vs. 24h despues. "
        "La prueba de Mann-Whitney U evalua si la diferencia es "
        "estadisticamente significativa (p < 0.05)."
    )

    # Summary table of finished matches
    st.subheader("Partidos finalizados")
    mr_display = mr[
        ["team", "opponent", "match_date", "outcome", "score", "stage"]
    ].copy()
    mr_display["match_date"] = pd.to_datetime(mr_display["match_date"]).dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    st.dataframe(mr_display, use_container_width=True)

    # For each match, compute pre/post sentiment
    st.subheader("Sentimiento antes vs. despues")

    for _, mrow in mr.iterrows():
        mdate = pd.to_datetime(mrow["match_date"])
        team = mrow["team"]
        opponent = mrow["opponent"]
        outcome = mrow["outcome"]
        score = mrow["score"]

        pre_start, pre_end, post_start, post_end = get_pre_post_windows(mdate)

        with st.expander(
            f"{team} vs {opponent} ({score}, {outcome}) — {mdate.strftime('%Y-%m-%d')}"
        ):
            # Filter comments for this team within windows
            if "published_at" in df.columns:
                dts = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
                team_mask = df["search_team"] == team

                pre_mask = team_mask & (dts >= pre_start) & (dts < pre_end)
                post_mask = team_mask & (dts >= post_start) & (dts <= post_end)

                pre_df = df[pre_mask]
                post_df = df[post_mask]

                if len(pre_df) < 3 or len(post_df) < 3:
                    st.info(
                        f"Datos insuficientes para esta ventana "
                        f"(pre: {len(pre_df)}, post: {len(post_df)} comentarios). "
                        "Se necesitan al menos 3 por ventana."
                    )
                    continue

                # Pre/post sentiment distribution
                pre_dist = pre_df["sentiment_bert"].value_counts(normalize=True)
                post_dist = post_df["sentiment_bert"].value_counts(normalize=True)

                col1, col2 = st.columns(2)
                with col1:
                    fig_pre = go.Figure(
                        data=[
                            go.Pie(
                                labels=["Positivo", "Negativo", "Neutral"],
                                values=[
                                    pre_dist.get("POS", 0) * 100,
                                    pre_dist.get("NEG", 0) * 100,
                                    pre_dist.get("NEU", 0) * 100,
                                ],
                                marker_colors=[POS_COLOR, NEG_COLOR, NEU_COLOR],
                                hole=0.4,
                                textinfo="label+percent",
                            )
                        ]
                    )
                    fig_pre.update_layout(
                        title=f"Antes (n={len(pre_df)})",
                        height=300,
                    )
                    st.plotly_chart(fig_pre, use_container_width=True)

                with col2:
                    fig_post = go.Figure(
                        data=[
                            go.Pie(
                                labels=["Positivo", "Negativo", "Neutral"],
                                values=[
                                    post_dist.get("POS", 0) * 100,
                                    post_dist.get("NEG", 0) * 100,
                                    post_dist.get("NEU", 0) * 100,
                                ],
                                marker_colors=[POS_COLOR, NEG_COLOR, NEU_COLOR],
                                hole=0.4,
                                textinfo="label+percent",
                            )
                        ]
                    )
                    fig_post.update_layout(
                        title=f"Despues (n={len(post_df)})",
                        height=300,
                    )
                    st.plotly_chart(fig_post, use_container_width=True)

                # Pre/post grouped bar
                bar_df = pd.DataFrame(
                    {
                        "Periodo": ["Antes", "Despues"],
                        "POS": [
                            pre_dist.get("POS", 0),
                            post_dist.get("POS", 0),
                        ],
                        "NEU": [
                            pre_dist.get("NEU", 0),
                            post_dist.get("NEU", 0),
                        ],
                        "NEG": [
                            pre_dist.get("NEG", 0),
                            post_dist.get("NEG", 0),
                        ],
                    }
                )
                fig_bar = go.Figure()
                for col_name in ["POS", "NEU", "NEG"]:
                    fig_bar.add_trace(
                        go.Bar(
                            name=col_name,
                            x=bar_df["Periodo"],
                            y=bar_df[col_name],
                            marker_color=SENT_COLORS[col_name],
                        )
                    )
                fig_bar.update_layout(
                    barmode="group",
                    title="Comparacion Antes / Despues",
                    yaxis_title="Proporcion",
                    height=300,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Statistical test (Mann-Whitney)
                try:
                    from scipy.stats import mannwhitneyu

                    # Map sentiment to numeric score
                    sent_map = {"POS": 2, "NEU": 1, "NEG": 0}
                    pre_scores = pre_df["sentiment_bert"].map(sent_map).values
                    post_scores = post_df["sentiment_bert"].map(sent_map).values
                    stat, p_value = mannwhitneyu(
                        pre_scores, post_scores, alternative="two-sided"
                    )
                    significant = p_value < 0.05
                    delta_pos_pct = post_dist.get("POS", 0) - pre_dist.get("POS", 0)

                    interpretation = (
                        "diferencia estadisticamente significativa"
                        if significant
                        else "diferencia NO significativa"
                    )
                    direction = (
                        f"El sentimiento positivo {'aumento' if delta_pos_pct > 0 else 'disminuyo'} "
                        f"en {abs(delta_pos_pct):.1%} despues del partido."
                    )

                    st.markdown(
                        f"**Prueba Mann-Whitney U**: {interpretation} (p = {p_value:.4f})"
                    )
                    st.markdown(direction)
                except ImportError:
                    st.info(
                        "scipy no esta disponible. Instala scipy para ver "
                        "los resultados de la prueba estadistica."
                    )

# ── Footer ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Autor**: Pablo Huidobro Garcia  \n"
    "Proyecto: [GitHub](https://github.com/your-username/"
    "mundial2026-sentiment-analysis)"
)
