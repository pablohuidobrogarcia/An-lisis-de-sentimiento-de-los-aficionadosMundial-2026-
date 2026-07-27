"""
Mundial 2026 — Social Media Sentiment Dashboard

Editorial-style dashboard with 5 sections:
1. Hero KPIs
2. Streamgraph (daily sentiment volume)
3. Treemap (topics)
4. Player Network
5. Ridgeline (sentiment by phase)

Usage:
    streamlit run dashboard/app.py
"""

import json
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pyvis.network import Network

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import PROCESSED_DIR  # noqa: E402

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mundial 2026 — Sentiment Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme palette ──────────────────────────────────────────────────────
DARK_BG = "#1e1e2e"
CARD_BG = "#282840"
POS_COLOR = "#2ECC71"
NEG_COLOR = "#E74C3C"
NEU_COLOR = "#95A5A6"
TEAM_COLORS = {
    "Spain": "#c60b1e",
    "Argentina": "#75aadb",
    "Brazil": "#009739",
    "France": "#002395",
    "England": "#cf142b",
}
SENT_COLORS = {"POS": POS_COLOR, "NEU": NEU_COLOR, "NEG": NEG_COLOR}

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #1e1e2e;
    color: #ffffff;
}
[data-testid="stSidebar"] {
    background-color: #1e1e2e;
    border-right: 1px solid #333333;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
.stSelectbox label, .stMultiSelect label {
    color: #cccccc !important;
}
[data-testid="stMetric"] {
    background-color: #282840;
    border-radius: 8px;
    padding: 12px;
    border: 1px solid #3a3a5c;
}
[data-testid="stMetric"] label {
    color: #888888 !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
}
.kpi-card {
    background-color: #282840;
    border: 1px solid #3a3a5c;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.kpi-card .kpi-value {
    font-size: 36px;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.2;
}
.kpi-card .kpi-label {
    font-size: 14px;
    color: #888888;
    margin-top: 4px;
}
.kpi-card .kpi-sublabel {
    font-size: 12px;
    color: #555555;
    margin-top: 2px;
}
.stTab [data-baseweb="tab-list"] {
    gap: 0;
}
.stTab [data-baseweb="tab"] {
    color: #888888;
    font-size: 14px;
}
.stTab [aria-selected="true"] {
    color: #ffffff;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Load data (cached) ──────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    sent_path = (
        PROCESSED_DIR / "comentarios_sentimiento" / "comentarios_sentimiento.parquet"
    )
    ner_path = (
        PROCESSED_DIR / "comentarios_topics_ner" / "comentarios_topics_ner.parquet"
    )

    df = pd.DataFrame()
    if sent_path.exists():
        df = pd.read_parquet(str(sent_path))

    if ner_path.exists():
        ner = pd.read_parquet(str(ner_path))
        ner_cols = [
            "comment_id",
            "topic_label",
            "players_mentioned",
            "brands_mentioned",
            "emojis",
            "n_emojis",
        ]
        df = df.merge(
            ner[ner_cols].drop_duplicates(subset=["comment_id"]),
            on="comment_id",
            how="left",
        )

    return df


df = load_data()

# ── Sidebar filters ─────────────────────────────────────────────────────────

st.sidebar.title("Mundial 2026")
st.sidebar.markdown("Analisis de Sentimiento en Redes Sociales")
st.sidebar.markdown("---")

available_teams = sorted(df["search_team"].dropna().unique()) if not df.empty else []
selected_teams = st.sidebar.multiselect(
    "Selecciones",
    options=available_teams,
    default=available_teams,
)

selected_languages = st.sidebar.multiselect(
    "Idioma",
    options=["es", "en"],
    default=["es", "en"],
    format_func=lambda x: "Espanol" if x == "es" else "English",
)

min_date, max_date = None, None
if not df.empty and "published_at" in df.columns:
    dates = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
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

raw = df.copy()
if not raw.empty:
    if selected_teams:
        raw = raw[raw["search_team"].isin(selected_teams)]
    if selected_languages:
        raw = raw[raw["language"].isin(selected_languages)]
    if date_range and len(date_range) == 2 and "published_at" in raw.columns:
        dts = pd.to_datetime(raw["published_at"], utc=True, errors="coerce")
        lo = pd.Timestamp(date_range[0], tz="UTC")
        hi = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1)
        raw = raw[(dts >= lo) & (dts < hi)]

df_ner = raw.dropna(subset=["topic_label"])
df_full = raw  # full dataset for sections needing full timeline

# ── Phase definitions ───────────────────────────────────────────────────────

PHASES = [
    ("Fase de Grupos", "2026-06-13", "2026-07-03"),
    ("Octavos", "2026-07-04", "2026-07-09"),
    ("Cuartos", "2026-07-10", "2026-07-12"),
    ("Semifinales", "2026-07-14", "2026-07-15"),
    ("3er Puesto", "2026-07-18", "2026-07-18"),
    ("Final", "2026-07-19", "2026-07-19"),
]


def classify_phase(dt):
    if pd.isna(dt):
        return None
    for name, start, end in PHASES:
        s = pd.Timestamp(start, tz="UTC")
        e = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        if s <= dt <= e:
            return name
    return None


# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════

t1, t2, t3, t4, t5 = st.tabs(
    [
        "Hero KPIs",
        "Streamgraph",
        "Treemap",
        "Red de Menciones",
        "Ridgeline",
    ]
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Hero KPIs
# ═════════════════════════════════════════════════════════════════════════════

with t1:
    st.header("Mundial 2026 — Radiografia de la Conversacion")

    if df_full.empty:
        st.info("No hay datos procesados todavia.")
        st.stop()

    total_comments = len(df_full)
    n_players = 0
    if "players_mentioned" in df_full.columns:
        all_players = (
            df_full["players_mentioned"].dropna().str.split(",").explode().str.strip()
        )
        all_players = all_players[all_players != ""]
        n_players = all_players.nunique()
        top_player = (
            all_players.value_counts().index[0] if not all_players.empty else "—"
        )
        top_player_count = (
            all_players.value_counts().iloc[0] if not all_players.empty else 0
        )
    else:
        top_player = "—"
        top_player_count = 0

    if "published_at" in df_full.columns:
        dts = pd.to_datetime(df_full["published_at"], utc=True, errors="coerce")
        peak_day = dts.dt.date.value_counts().index[0]
        peak_day_count = dts.dt.date.value_counts().iloc[0]
    else:
        peak_day = "—"
        peak_day_count = 0

    max_likes = (
        int(df_full["like_count"].max()) if "like_count" in df_full.columns else 0
    )

    kpis = [
        ("201,694", "Comentarios", "analizados en total"),
        (str(n_players), "Jugadores", "detectados via NER"),
        (f"{peak_day_count:,}", f"Pico: {peak_day}", "comentarios en un dia"),
        (f"{top_player_count:,}", f"Mas mencionado: {top_player}", "menciones"),
        (f"{max_likes:,}", "Like record", "en un solo comentario"),
    ]

    cols = st.columns(5)
    for col, (value, label, sublabel) in zip(cols, kpis):
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-sublabel">{sublabel}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Streamgraph
# ═════════════════════════════════════════════════════════════════════════════

with t2:
    st.header("El Pulso del Torneo: Sentimiento Dia a Dia")

    if df_full.empty or "published_at" not in df_full.columns:
        st.warning("Datos insuficientes.")
    else:
        daily = df_full.copy()
        daily["date"] = pd.to_datetime(daily["published_at"], utc=True).dt.floor("D")

        daily_agg = (
            daily.groupby("date")
            .agg(
                total=("comment_id", "count"),
                pos=("sentiment_bert", lambda x: (x == "POS").sum()),
                neu=("sentiment_bert", lambda x: (x == "NEU").sum()),
                neg=("sentiment_bert", lambda x: (x == "NEG").sum()),
            )
            .reset_index()
        )

        daily_agg["pos_pct"] = daily_agg["pos"] / daily_agg["total"]
        daily_agg["neg_pct"] = daily_agg["neg"] / daily_agg["total"]
        daily_agg["neu_pct"] = daily_agg["neu"] / daily_agg["total"]

        fig = go.Figure()

        # Phase backgrounds
        phase_colors_bg = {
            "Fase de Grupos": "rgba(52, 152, 219, 0.08)",
            "Octavos": "rgba(155, 89, 182, 0.08)",
            "Cuartos": "rgba(230, 126, 34, 0.08)",
            "Semifinales": "rgba(26, 188, 156, 0.08)",
            "3er Puesto": "rgba(241, 196, 15, 0.08)",
            "Final": "rgba(231, 76, 60, 0.08)",
        }
        for phase_name, start, end in PHASES:
            s = pd.Timestamp(start, tz="UTC")
            e = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
            fig.add_vrect(
                x0=s,
                x1=e,
                fillcolor=phase_colors_bg.get(phase_name, "rgba(255,255,255,0.03)"),
                layer="below",
                line_width=0,
            )

        # POS stream (bottom)
        fig.add_trace(
            go.Scatter(
                x=daily_agg["date"],
                y=daily_agg["pos_pct"],
                mode="lines",
                fill="tonexty",
                name="POS",
                line={"width": 0},
                fillcolor=POS_COLOR,
                stackgroup="one",
                hovertemplate="%{x|%d %b}<br>POS: %{y:.1%}<extra></extra>",
            )
        )

        # NEU stream (middle)
        fig.add_trace(
            go.Scatter(
                x=daily_agg["date"],
                y=daily_agg["neu_pct"],
                mode="lines",
                fill="tonexty",
                name="NEU",
                line={"width": 0},
                fillcolor=NEU_COLOR,
                stackgroup="one",
                hovertemplate="%{x|%d %b}<br>NEU: %{y:.1%}<extra></extra>",
            )
        )

        # NEG stream (top)
        fig.add_trace(
            go.Scatter(
                x=daily_agg["date"],
                y=daily_agg["neg_pct"],
                mode="lines",
                fill="tonexty",
                name="NEG",
                line={"width": 0},
                fillcolor=NEG_COLOR,
                stackgroup="one",
                hovertemplate="%{x|%d %b}<br>NEG: %{y:.1%}<extra></extra>",
            )
        )

        # Phase annotations
        phase_annot_dates = {
            "Fase de Grupos": "2026-06-23",
            "Octavos": "2026-07-06",
            "Cuartos": "2026-07-11",
            "Semifinales": "2026-07-14",
            "Final": "2026-07-19",
        }
        for phase_name, anno_date in phase_annot_dates.items():
            fig.add_annotation(
                x=pd.Timestamp(anno_date, tz="UTC"),
                y=0.98,
                yref="y",
                text=f"<b>{phase_name}</b>",
                showarrow=False,
                font={"size": 11, "color": "#aaaaaa"},
                yshift=10,
            )

        # Peak NEG annotation
        peak_neg_row = daily_agg.loc[daily_agg["neg_pct"].idxmax()]
        fig.add_annotation(
            x=peak_neg_row["date"],
            y=peak_neg_row["neg_pct"],
            text=f"Pico NEG: {peak_neg_row['neg_pct']:.0%}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            ax=0,
            ay=-40,
            font={"size": 12, "color": NEG_COLOR},
            bgcolor=DARK_BG,
            bordercolor=NEG_COLOR,
            borderwidth=1,
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_BG,
            font={"color": "#ffffff"},
            title={
                "text": "El Pulso del Torneo: Sentimiento Dia a Dia",
                "x": 0.5,
                "font": {"size": 24},
            },
            xaxis={"title": "", "showgrid": False, "zeroline": False},
            yaxis={
                "title": "Proporcion",
                "showgrid": True,
                "gridcolor": "#333333",
                "zeroline": False,
                "tickformat": ".0%",
            },
            hovermode="x unified",
            legend={"orientation": "h", "y": 1.02, "x": 0.5, "xanchor": "center"},
            margin={"l": 50, "r": 30, "t": 80, "b": 40},
            height=500,
        )

        st.plotly_chart(fig, width="stretch")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Treemap
# ═════════════════════════════════════════════════════════════════════════════

with t3:
    st.header("De Que Habla la Aficion")

    if df_ner.empty or "topic_label" not in df_ner.columns:
        st.warning("Temas no disponibles. Ejecuta topic modeling primero.")
    else:
        try:
            topic_agg = (
                df_ner.groupby("topic_label")
                .agg(
                    volumen=("comment_id", "count"),
                    pos_pct=("sentiment_bert", lambda x: (x == "POS").mean()),
                    neu_pct=("sentiment_bert", lambda x: (x == "NEU").mean()),
                    neg_pct=("sentiment_bert", lambda x: (x == "NEG").mean()),
                )
                .reset_index()
            )

            topic_agg["sentiment_score"] = topic_agg["pos_pct"] - topic_agg["neg_pct"]

            topic_samples = (
                df_ner.groupby("topic_label")["text_clean"]
                .apply(
                    lambda x: "<br>---<br>".join(
                        '"' + str(t)[:120] + '"'
                        for t in x.dropna().sample(min(3, len(x)), random_state=42)
                    )
                )
                .reset_index()
            )

            topic_agg = topic_agg.merge(topic_samples, on="topic_label")
            topic_agg["hover_text"] = topic_agg.apply(
                lambda r: (
                    f"<b>{r['topic_label']}</b><br>"
                    f"Volumen: {r['volumen']:,} comentarios<br>"
                    f"POS: {r['pos_pct']:.0%} | NEU: {r['neu_pct']:.0%} | NEG: {r['neg_pct']:.0%}<br>"
                    f"Score: {r['sentiment_score']:+.2f}<br>"
                    f"<br><b>Ejemplos:</b><br>{r['text_clean']}"
                ),
                axis=1,
            )

            topic_agg = topic_agg.sort_values("volumen", ascending=False)

            topic_agg["main_topic"] = topic_agg["topic_label"].str.split(" / ").str[0]
            topic_agg["sub_keywords"] = (
                topic_agg["topic_label"].str.split(" / ", n=1).str[1].fillna("")
            )

            st.write(
                f"**Datos:** {len(topic_agg)} topicos, {topic_agg['volumen'].sum():,} comentarios"
            )

            fig = go.Figure(
                go.Treemap(
                    labels=topic_agg["topic_label"],
                    parents=[""] * len(topic_agg),
                    values=topic_agg["volumen"],
                    customdata=topic_agg[
                        ["main_topic", "sub_keywords"]
                    ].values.tolist(),
                    texttemplate="<b>%{customdata[0]}</b><br><span style='font-size:10px'>%{customdata[1]}</span><br><span style='font-size:11px'>%{value:,} comentarios</span>",
                    textfont={"size": 13},
                    marker={
                        "colors": topic_agg["sentiment_score"].tolist(),
                        "colorscale": [
                            [0.0, "#E74C3C"],
                            [0.5, "#95A5A6"],
                            [1.0, "#2ECC71"],
                        ],
                        "cmin": -0.5,
                        "cmax": 0.5,
                        "line": {"width": 0},
                    },
                )
            )

            fig.update_layout(
                paper_bgcolor=DARK_BG,
                plot_bgcolor=DARK_BG,
                font={"color": "#ffffff"},
                margin={"l": 5, "r": 5, "t": 5, "b": 5},
                height=600,
            )

            st.plotly_chart(fig, width="stretch")

            with st.expander("Ver tabla de topicos"):
                st.dataframe(
                    topic_agg[
                        [
                            "topic_label",
                            "volumen",
                            "pos_pct",
                            "neu_pct",
                            "neg_pct",
                            "sentiment_score",
                        ]
                    ],
                    width="stretch",
                )

        except Exception as e:
            st.error(f"Error al generar treemap: {e}")
            import traceback

            st.code(traceback.format_exc(), language="python")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Red de Menciones
# ═════════════════════════════════════════════════════════════════════════════

KNOWN_PLAYERS = {
    "Spain": [
        "Pedri",
        "Gavi",
        "Lamine Yamal",
        "Nico Williams",
        "Rodri",
        "Unai Simon",
        "Dani Carvajal",
        "Aymeric Laporte",
        "Alvaro Morata",
        "Fermin Lopez",
        "Mikel Merino",
        "Dani Olmo",
    ],
    "Argentina": [
        "Lionel Messi",
        "Angel Di Maria",
        "Julian Alvarez",
        "Enzo Fernandez",
        "Alexis Mac Allister",
        "Lautaro Martinez",
        "Emiliano Martinez",
        "Nicolas Otamendi",
        "Cristian Romero",
        "Rodrigo De Paul",
        "Leandro Paredes",
        "Nahuel Molina",
    ],
    "Brazil": [
        "Neymar",
        "Vinicius Jr",
        "Rodrygo",
        "Endrick",
        "Raphinha",
        "Casemiro",
        "Marquinhos",
        "Alisson",
        "Gabriel Martinelli",
        "Bruno Guimaraes",
        "Joao Pedro",
        "Lucas Paqueta",
    ],
    "France": [
        "Kylian Mbappe",
        "Antoine Griezmann",
        "Eduardo Camavinga",
        "Aurelien Tchouameni",
        "Mike Maignan",
        "Dayot Upamecano",
        "Ousmane Dembele",
        "Randal Kolo Muani",
        "Olivier Giroud",
        "Theo Hernandez",
        "Jules Kounde",
        "Adrien Rabiot",
    ],
    "England": [
        "Harry Kane",
        "Jude Bellingham",
        "Bukayo Saka",
        "Declan Rice",
        "Phil Foden",
        "Mason Mount",
        "Jack Grealish",
        "Marcus Rashford",
        "Jordan Pickford",
        "Kyle Walker",
        "John Stones",
        "Cole Palmer",
    ],
}


def _normalize(name):
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


PLAYER_TEAM = {}
for _team, _players in KNOWN_PLAYERS.items():
    for _p in _players:
        PLAYER_TEAM[_p] = _team
        PLAYER_TEAM[_normalize(_p)] = _team


def _parse_players(val):
    import ast

    if pd.isna(val):
        return set()
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return set()
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return set(parsed)
        except (ValueError, SyntaxError):
            return {p.strip() for p in val.split(",") if p.strip()}
    if isinstance(val, (list, tuple)):
        return set(val)
    return set()


def build_player_network(df: pd.DataFrame) -> dict:
    mention_counts = Counter()
    cooccur = defaultdict(int)
    n_with_mentions = 0

    for val in df["players_mentioned"]:
        players = _parse_players(val)
        if not players:
            continue
        n_with_mentions += 1
        for p in players:
            mention_counts[p] += 1
        for p1, p2 in combinations(sorted(players), 2):
            cooccur[(p1, p2)] += 1

    top_players = {p for p, _ in mention_counts.most_common(30)}

    edges = []
    for (p1, p2), weight in cooccur.items():
        if p1 in top_players and p2 in top_players and weight >= 3:
            edges.append((p1, p2, weight))

    net = Network(height="700px", width="100%", bgcolor=DARK_BG, font_color="#ffffff")
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.01,
        damping=0.09,
    )

    max_mentions = mention_counts.most_common(1)[0][1] if mention_counts else 1
    for player, count in mention_counts.most_common(30):
        team = PLAYER_TEAM.get(player) or PLAYER_TEAM.get(_normalize(player))
        color = TEAM_COLORS.get(team, "#888888")
        title_parts = [f"<b>{player}</b>"]
        if team:
            title_parts.append(f"Team: {team}")
        title_parts.append(f"Mentions: {count:,}")
        net.add_node(
            player,
            label=player,
            title="<br>".join(title_parts),
            color=color,
            size=5 + (count / max_mentions) * 30,
            borderWidth=0,
        )

    for p1, p2, weight in edges:
        net.add_edge(
            p1,
            p2,
            value=weight,
            title=f"{p1} &amp; {p2}<br>Co-mentions: {weight}",
            color={"color": "#ffffff", "opacity": 0.3 + min(weight / 50, 0.7)},
        )

    net.set_options(
        json.dumps(
            {
                "nodes": {
                    "font": {"size": 14, "face": "Arial", "strokeWidth": 0},
                    "borderWidth": 0,
                },
                "edges": {
                    "smooth": False,
                    "width": 1,
                },
                "physics": {
                    "barnesHut": {
                        "gravitationalConstant": -8000,
                        "centralGravity": 0.3,
                        "springLength": 150,
                        "springConstant": 0.01,
                        "damping": 0.09,
                    },
                    "stabilization": {"iterations": 100},
                },
                "interaction": {
                    "hover": True,
                    "tooltipDelay": 0,
                    "zoomView": True,
                    "dragView": True,
                },
            }
        )
    )

    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as f:
        net.save_graph(f.name)
        with open(f.name, encoding="utf-8") as f2:
            html = f2.read()

    return {
        "html": html,
        "n_with_mentions": n_with_mentions,
        "n_unique_players": len(mention_counts),
        "n_edges": sum(
            1
            for (p1, p2), w in cooccur.items()
            if p1 in top_players and p2 in top_players and w >= 3
        ),
    }


with t4:
    st.header("Red de Menciones: Quien se compara con quien")
    st.markdown(
        "Cada vez que dos jugadores aparecen mencionados en el **mismo comentario**, "
        "se dibuja una arista entre ellos. El grosor refleja la frecuencia de co-mencion. "
        "Agrupacion automatica por seleccion (colores) mediante simulacion fisica force-directed."
    )

    if df_ner.empty or "players_mentioned" not in df_ner.columns:
        st.info("Datos de NER no disponibles. Ejecuta topic modeling primero.")
    else:
        with st.spinner("Construyendo red de co-menciones..."):
            try:
                result = build_player_network(df_ner)
                st.components.v1.html(result["html"], height=730, scrolling=True)
            except Exception as e:  # noqa: BLE001
                st.error(f"Error al generar la red: {e}")
                st.stop()

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Comentarios con menciones", f"{result['n_with_mentions']:,}")
        with col2:
            st.metric("Jugadores distintos detectados", result["n_unique_players"])
        with col3:
            st.metric("Pares de co-mencion (>=3)", result["n_edges"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — Ridgeline
# ═════════════════════════════════════════════════════════════════════════════

with t5:
    st.header("La Presion de la Eliminacion Directa")

    if df_full.empty or "published_at" not in df_full.columns:
        st.warning("Datos insuficientes.")
    else:
        ridge_df = df_full.copy()
        ridge_df["date"] = pd.to_datetime(ridge_df["published_at"], utc=True)
        ridge_df["fase"] = ridge_df["date"].apply(classify_phase)

        SENT_MAP = {"NEG": 0, "NEU": 0.5, "POS": 1}
        ridge_df["sent_num"] = ridge_df["sentiment_bert"].map(SENT_MAP)

        order = [
            "Fase de Grupos",
            "Octavos",
            "Cuartos",
            "Semifinales",
            "3er Puesto",
            "Final",
        ]
        y_spacing = 0.7
        colors_ridge = {
            "Fase de Grupos": "#3498DB",
            "Octavos": "#9B59B6",
            "Cuartos": "#E67E22",
            "Semifinales": "#1ABC9C",
            "3er Puesto": "#F39C12",
            "Final": "#E74C3C",
        }

        fig = go.Figure()

        for i, phase in enumerate(order):
            sub = ridge_df[ridge_df["fase"] == phase]
            vals = sub["sent_num"].dropna()
            if len(vals) == 0:
                continue

            fig.add_trace(
                go.Violin(
                    y=[i * y_spacing] * len(vals),
                    x=vals,
                    name=phase,
                    orientation="h",
                    side="positive",
                    width=1.2,
                    points=False,
                    line={"color": colors_ridge[phase], "width": 2},
                    fillcolor=colors_ridge[phase],
                    opacity=0.6,
                    scalemode="width",
                    bandwidth=0.08,
                    hovertemplate=(
                        f"<b>{phase}</b><br>"
                        f"n={len(sub):,}<br>"
                        f"POS: {(sub['sentiment_bert']=='POS').mean():.0%}<br>"
                        f"NEU: {(sub['sentiment_bert']=='NEU').mean():.0%}<br>"
                        f"NEG: {(sub['sentiment_bert']=='NEG').mean():.0%}<extra></extra>"
                    ),
                )
            )

        fig.add_annotation(
            x=0.5,
            y=1.08,
            xref="paper",
            yref="paper",
            text="Distribucion de sentimiento por fase del torneo. "
            "El pico NEG se desplaza de izquierda a derecha a medida que avanza.",
            showarrow=False,
            font={"size": 14, "color": "#aaaaaa"},
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_BG,
            font={"color": "#ffffff"},
            title={
                "text": "La Presion de la Eliminacion Directa",
                "x": 0.5,
                "font": {"size": 24},
            },
            xaxis={
                "title": "",
                "tickvals": [0, 0.5, 1],
                "ticktext": ["<b>NEG</b>", "<b>NEU</b>", "<b>POS</b>"],
                "range": [-0.3, 1.3],
                "showgrid": True,
                "gridcolor": "#333333",
                "zeroline": False,
            },
            yaxis={
                "tickvals": [i * y_spacing for i in range(len(order))],
                "ticktext": order,
                "showgrid": False,
                "zeroline": False,
            },
            hovermode="y unified",
            showlegend=False,
            margin={"l": 130, "r": 30, "t": 100, "b": 40},
            height=550,
        )

        st.plotly_chart(fig, width="stretch")

        # Summary table
        st.markdown("---")
        st.subheader("Distribucion por Fase")
        summary_rows = []
        for phase in order:
            sub = ridge_df[ridge_df["fase"] == phase]
            n = len(sub)
            if n == 0:
                continue
            pos = (sub["sentiment_bert"] == "POS").mean() * 100
            neu = (sub["sentiment_bert"] == "NEU").mean() * 100
            neg = (sub["sentiment_bert"] == "NEG").mean() * 100
            summary_rows.append(
                {
                    "Fase": phase,
                    "Comentarios": f"{n:,}",
                    "POS": f"{pos:.1f}%",
                    "NEU": f"{neu:.1f}%",
                    "NEG": f"{neg:.1f}%",
                }
            )
        st.dataframe(pd.DataFrame(summary_rows), width="stretch")


# ── Footer ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Autor**: Pablo Huidobro Garcia  \n"
    "Proyecto: [GitHub](https://github.com/your-username/"
    "mundial2026-sentiment-analysis)"
)
