"""
Dashboard de Mortalidad - Medellín 2012–2021
Autores: Camilo González & Rubén Esguerra
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA Y PREPARACIÓN DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

MESES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
}

OPS_COLORS = {
    "Enfermedades del sistema circulatorio":                    "#E63946",
    "Neoplasias (Tumores)":                                     "#457B9D",
    "Todas las demas enfermedades":                             "#2A9D8F",
    "Enfermedades Transmisibles":                               "#E9C46A",
    "Causas externas":                                          "#F4A261",
    "Ciertas afecciones originadas en el periodo perinatal":    "#A8DADC",
    "Signos sintomas y afecciones mal definidas":               "#6D6875",
}

ETAREO_ORDER = [
    "< 1 año", "1-4", "5-9", "10-14", "15-19", "20-24", "25-29",
    "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64",
    "65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95-99", "100 y mas"
]


def load_data(path: str = "defunciones_clean.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["MES_NOMBRE"] = df["MES"].map(MESES)
    df["ETAREO_QUIN"] = pd.Categorical(
        df["ETAREO_QUIN"],
        categories=[e for e in ETAREO_ORDER if e in df["ETAREO_QUIN"].dropna().unique()],
        ordered=True
    )
    return df


DF = load_data()

# Opciones para filtros
ANOS      = sorted(DF["ANO"].unique())
SEXOS     = sorted(DF["SEXO"].dropna().unique())
OPS_GRPS  = sorted(DF["NOM_667_OPS_GRUPO"].dropna().unique())
COMUNAS   = sorted(DF["COMUNA_RES"].dropna().unique())


# ─────────────────────────────────────────────────────────────────────────────
# 2. HELPERS DE FILTRADO
# ─────────────────────────────────────────────────────────────────────────────

def apply_filters(anos, sexos, ops_grupos, comunas):
    mask = pd.Series([True] * len(DF), index=DF.index)
    if anos:
        mask &= DF["ANO"].isin(anos)
    if sexos:
        mask &= DF["SEXO"].isin(sexos)
    if ops_grupos:
        mask &= DF["NOM_667_OPS_GRUPO"].isin(ops_grupos)
    if comunas:
        mask &= DF["COMUNA_RES"].isin(comunas)
    return DF[mask].copy()


# ─────────────────────────────────────────────────────────────────────────────
# 3. FIGURAS
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_TEMPLATE = "plotly_white"


def fig_ops_bar(df: pd.DataFrame) -> go.Figure:
    counts = (
        df["NOM_667_OPS_GRUPO"].value_counts().reset_index()
        .rename(columns={"NOM_667_OPS_GRUPO": "Grupo", "count": "Defunciones"})
        .sort_values("Defunciones")
    )
    fig = px.bar(
        counts, x="Defunciones", y="Grupo", orientation="h",
        color="Grupo",
        color_discrete_map=OPS_COLORS,
        text="Defunciones",
        template=PLOTLY_TEMPLATE,
        title="Distribución por Causa de Muerte (Grupo OPS)"
    )
    fig.update_traces(texttemplate="%{x:,}", textposition="outside")
    fig.update_layout(
        showlegend=False,
        yaxis_title="",
        xaxis_title="Defunciones",
        margin=dict(l=10, r=30, t=50, b=10),
    )
    return fig


def fig_evolucion_anual(df: pd.DataFrame) -> go.Figure:
    evol = (
        df.groupby(["ANO", "NOM_667_OPS_GRUPO"])
        .size().reset_index(name="Defunciones")
    )
    fig = px.line(
        evol, x="ANO", y="Defunciones", color="NOM_667_OPS_GRUPO",
        markers=True,
        color_discrete_map=OPS_COLORS,
        template=PLOTLY_TEMPLATE,
        title="Evolución Anual de Defunciones por Grupo OPS",
        labels={"NOM_667_OPS_GRUPO": "Grupo", "ANO": "Año"}
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.45, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=50, b=130),
        xaxis=dict(dtick=1),
    )
    return fig


def fig_ops_sexo(df: pd.DataFrame) -> go.Figure:
    dfg = (
        df[df["SEXO"].isin(["Masculino", "Femenino"])]
        .groupby(["NOM_667_OPS_GRUPO", "SEXO"])
        .size().reset_index(name="Defunciones")
    )
    fig = px.bar(
        dfg, x="NOM_667_OPS_GRUPO", y="Defunciones", color="SEXO",
        barmode="group",
        color_discrete_map={"Masculino": "#1565C0", "Femenino": "#C2185B"},
        template=PLOTLY_TEMPLATE,
        title="Defunciones por Causa y Sexo",
        labels={"NOM_667_OPS_GRUPO": "Grupo OPS", "SEXO": "Sexo"}
    )
    fig.update_layout(
        xaxis_tickangle=-35,
        legend_title_text="Sexo",
        margin=dict(l=10, r=10, t=50, b=130),
        xaxis_title="",
    )
    return fig


def fig_heatmap_etareo(df: pd.DataFrame) -> go.Figure:
    df_e = df[df["ETAREO_QUIN"].notna()].copy()
    pivot = pd.crosstab(df_e["NOM_667_OPS_GRUPO"], df_e["ETAREO_QUIN"])
    # normalise by row → % within each OPS group
    pivot_pct = (pivot.div(pivot.sum(axis=1), axis=0) * 100).round(1)
    # keep ordered columns
    cols_ordered = [c for c in ETAREO_ORDER if c in pivot_pct.columns]
    pivot_pct = pivot_pct[cols_ordered]

    fig = go.Figure(go.Heatmap(
        z=pivot_pct.values,
        x=pivot_pct.columns.tolist(),
        y=pivot_pct.index.tolist(),
        colorscale="YlOrRd",
        colorbar=dict(title="% en<br>grupo OPS"),
        text=pivot_pct.values,
        texttemplate="%{text:.1f}%",
        hovertemplate="Grupo: %{y}<br>Etario: %{x}<br>Porcentaje: %{z:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        title="Heatmap: Grupo OPS × Grupo Etario (% por fila)",
        template=PLOTLY_TEMPLATE,
        xaxis_tickangle=-45,
        margin=dict(l=10, r=10, t=50, b=10),
        height=360,
    )
    return fig


def fig_top10_comunas(df: pd.DataFrame) -> go.Figure:
    df_c = df[~df["COMUNA_RES"].isin(["Sin informacion", "sin informacion"])].dropna(subset=["COMUNA_RES"])
    top10 = df_c["COMUNA_RES"].value_counts().head(10).index
    df_t = df_c[df_c["COMUNA_RES"].isin(top10)]
    dfg = (
        df_t.groupby(["COMUNA_RES", "NOM_667_OPS_GRUPO"])
        .size().reset_index(name="Defunciones")
    )
    fig = px.bar(
        dfg, x="COMUNA_RES", y="Defunciones", color="NOM_667_OPS_GRUPO",
        color_discrete_map=OPS_COLORS,
        template=PLOTLY_TEMPLATE,
        title="Top 10 Comunas — Defunciones por Causa",
        labels={"COMUNA_RES": "Comuna", "NOM_667_OPS_GRUPO": "Grupo"}
    )
    fig.update_layout(
        xaxis_tickangle=-35,
        legend=dict(orientation="h", yanchor="bottom", y=-0.55, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=50, b=160),
        xaxis_title="",
    )
    return fig


def fig_evolucion_mensual(df: pd.DataFrame) -> go.Figure:
    evol = df.groupby(["ANO", "MES"]).size().reset_index(name="Defunciones")
    evol["Periodo"] = evol["ANO"].astype(str) + "-" + evol["MES"].astype(str).str.zfill(2)
    evol = evol.sort_values(["ANO", "MES"])
    fig = px.area(
        evol, x="Periodo", y="Defunciones",
        template=PLOTLY_TEMPLATE,
        title="Evolución Mensual Total de Defunciones",
        color_discrete_sequence=["#457B9D"]
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        margin=dict(l=10, r=10, t=50, b=80),
    )
    return fig


def fig_seg_social(df: pd.DataFrame) -> go.Figure:
    df_s = df[df["SEG_SOCIAL"] != "Sin info"]
    dfg = df_s.groupby(["NOM_667_OPS_GRUPO", "SEG_SOCIAL"]).size().reset_index(name="Defunciones")
    fig = px.bar(
        dfg, x="NOM_667_OPS_GRUPO", y="Defunciones", color="SEG_SOCIAL",
        barmode="stack",
        template=PLOTLY_TEMPLATE,
        title="Seguridad Social × Causa de Muerte",
        labels={"NOM_667_OPS_GRUPO": "Grupo OPS", "SEG_SOCIAL": "Régimen"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        xaxis_tickangle=-35,
        xaxis_title="",
        margin=dict(l=10, r=10, t=50, b=130),
        legend=dict(orientation="h", yanchor="bottom", y=-0.55, xanchor="left", x=0),
    )
    return fig


def fig_edad_hist(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df.dropna(subset=["EDAD_SIMPLE"]),
        x="EDAD_SIMPLE", nbins=30,
        color="SEXO" if df["SEXO"].isin(["Masculino", "Femenino"]).any() else None,
        color_discrete_map={"Masculino": "#1565C0", "Femenino": "#C2185B", "Indeterminado": "#aaa"},
        template=PLOTLY_TEMPLATE,
        title="Distribución de Edad al Fallecimiento",
        labels={"EDAD_SIMPLE": "Edad (años)", "count": "Frecuencia", "SEXO": "Sexo"}
    )
    fig.update_layout(
        bargap=0.05,
        legend_title_text="Sexo",
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def fig_nivel_edu(df: pd.DataFrame) -> go.Figure:
    df_e = df[~df["NIVEL_EDU_GRUPO"].isin(["Sin info"])].copy()
    orden = ["Básica", "Media", "Técnico/Tecnológico", "Superior"]
    dfg = df_e.groupby(["NOM_667_OPS_GRUPO", "NIVEL_EDU_GRUPO"]).size().reset_index(name="Defunciones")
    dfg["NIVEL_EDU_GRUPO"] = pd.Categorical(dfg["NIVEL_EDU_GRUPO"], categories=orden, ordered=True)
    dfg = dfg.sort_values("NIVEL_EDU_GRUPO")
    fig = px.bar(
        dfg, y="NOM_667_OPS_GRUPO", x="Defunciones", color="NIVEL_EDU_GRUPO",
        barmode="stack", orientation="h",
        template=PLOTLY_TEMPLATE,
        title="Nivel Educativo × Causa de Muerte",
        labels={"NOM_667_OPS_GRUPO": "Grupo OPS", "NIVEL_EDU_GRUPO": "Nivel Edu."},
        color_discrete_sequence=px.colors.sequential.RdYlGn
    )
    fig.update_layout(
        yaxis_title="",
        margin=dict(l=10, r=10, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. KPIs
# ─────────────────────────────────────────────────────────────────────────────

def kpi_card(title: str, value: str, icon: str, color: str = "#457B9D") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={"fontSize": "2rem"}),
                html.Div([
                    html.P(title, className="text-muted mb-0",
                           style={"fontSize": "0.78rem", "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
                    html.H4(value, className="mb-0 fw-bold", style={"color": color}),
                ], className="ms-3")
            ], className="d-flex align-items-center")
        ]),
        className="shadow-sm border-0 mb-3",
        style={"borderLeft": f"5px solid {color} !important", "borderRadius": "10px"}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

SIDEBAR_STYLE = {
    "position": "sticky",
    "top": "70px",
    "height": "calc(100vh - 80px)",
    "overflowY": "auto",
    "padding": "1.5rem 1rem",
    "backgroundColor": "#f8f9fa",
    "borderRight": "1px solid #dee2e6",
}

NAVBAR = dbc.Navbar(
    dbc.Container([
        html.A(
            dbc.Row([
                dbc.Col(html.Img(src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Red_Cross.svg",
                                 height="32px")),
                dbc.Col(dbc.NavbarBrand("Dashboard de Mortalidad · Medellín 2012–2021",
                                       className="ms-2 fw-bold fs-5")),
            ], align="center", className="g-0"),
            href="#", style={"textDecoration": "none"}
        ),
        dbc.NavbarText("Ciencia de Datos · Portafolio", className="text-white-50 small d-none d-md-block"),
    ], fluid=True),
    color="#1d3557", dark=True, sticky="top", className="shadow-sm mb-0"
)

SIDEBAR = html.Div([
    html.H6("🎛️ FILTROS GLOBALES", className="text-uppercase fw-bold text-secondary mb-3",
            style={"letterSpacing": "0.1em", "fontSize": "0.75rem"}),

    html.Label("📅 Año", className="fw-semibold small"),
    dcc.Dropdown(
        id="filter-ano",
        options=[{"label": str(a), "value": a} for a in ANOS],
        value=[], multi=True, placeholder="Todos los años",
        className="mb-3"
    ),

    html.Label("⚧ Sexo", className="fw-semibold small"),
    dcc.Dropdown(
        id="filter-sexo",
        options=[{"label": s, "value": s} for s in SEXOS],
        value=[], multi=True, placeholder="Todos",
        className="mb-3"
    ),

    html.Label("🏥 Grupo OPS", className="fw-semibold small"),
    dcc.Dropdown(
        id="filter-ops",
        options=[{"label": g, "value": g} for g in OPS_GRPS],
        value=[], multi=True, placeholder="Todos los grupos",
        className="mb-3"
    ),

    html.Label("🏙️ Comuna", className="fw-semibold small"),
    dcc.Dropdown(
        id="filter-comuna",
        options=[{"label": c, "value": c} for c in COMUNAS],
        value=[], multi=True, placeholder="Todas las comunas",
        className="mb-4"
    ),

    dbc.Button("↺ Limpiar filtros", id="btn-reset", color="outline-secondary",
               size="sm", className="w-100"),

    html.Hr(),
    html.P("📊 Dataset: 145,377 registros\n📅 Período: 2012–2021\n📍 Ciudad: Medellín, Colombia",
           className="text-muted small", style={"whiteSpace": "pre-line", "lineHeight": "1.8"}),
], style=SIDEBAR_STYLE)


def tab_general():
    return dbc.Container([
        # KPIs row
        dbc.Row([
            dbc.Col(html.Div(id="kpi-total"),    xs=12, sm=6, lg=3),
            dbc.Col(html.Div(id="kpi-ops"),      xs=12, sm=6, lg=3),
            dbc.Col(html.Div(id="kpi-edad"),     xs=12, sm=6, lg=3),
            dbc.Col(html.Div(id="kpi-comuna"),   xs=12, sm=6, lg=3),
        ], className="mb-2"),

        # Charts row
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id="chart-ops-bar", config={"displayModeBar": False}))],
                             className="shadow-sm border-0"), md=7),
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id="chart-edad-hist", config={"displayModeBar": False}))],
                             className="shadow-sm border-0"), md=5),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id="chart-seg-social", config={"displayModeBar": False}))],
                             className="shadow-sm border-0"), md=12),
        ])
    ], fluid=True, className="pt-3")


def tab_temporal():
    return dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id="chart-evol-anual", config={"displayModeBar": False}))],
                             className="shadow-sm border-0"), md=12)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id="chart-evol-mensual", config={"displayModeBar": False}))],
                             className="shadow-sm border-0"), md=12)
        ])
    ], fluid=True, className="pt-3")


def tab_demografico():
    return dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id="chart-ops-sexo", config={"displayModeBar": False}))],
                             className="shadow-sm border-0"), md=12)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id="chart-heatmap-etareo", config={"displayModeBar": False}))],
                             className="shadow-sm border-0"), md=12)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id="chart-nivel-edu", config={"displayModeBar": False}))],
                             className="shadow-sm border-0"), md=12)
        ]),
    ], fluid=True, className="pt-3")


def tab_geografico():
    return dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id="chart-top10-comunas", config={"displayModeBar": False}))],
                             className="shadow-sm border-0"), md=12)
        ])
    ], fluid=True, className="pt-3")


TABS = dbc.Tabs([
    dbc.Tab(tab_general(),     label="📊 Vista General",       tab_id="tab-general"),
    dbc.Tab(tab_temporal(),    label="📈 Análisis Temporal",   tab_id="tab-temporal"),
    dbc.Tab(tab_demografico(), label="👥 Análisis Demográfico",tab_id="tab-demografico"),
    dbc.Tab(tab_geografico(),  label="🗺️ Análisis Geográfico", tab_id="tab-geografico"),
], id="main-tabs", active_tab="tab-general", className="mb-0")

MAIN_CONTENT = html.Div(TABS, style={"padding": "0 1.5rem 2rem"})

APP_LAYOUT = html.Div([
    NAVBAR,
    dbc.Row([
        dbc.Col(SIDEBAR, xs=12, md=3, xl=2, className="px-0 d-none d-md-block"),
        dbc.Col(MAIN_CONTENT, xs=12, md=9, xl=10),
    ], className="g-0"),
])


# ─────────────────────────────────────────────────────────────────────────────
# 6. APP
# ─────────────────────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    title="Dashboard Mortalidad Medellín",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
server = app.server  # for gunicorn / Render

app.layout = APP_LAYOUT


# ─────────────────────────────────────────────────────────────────────────────
# 7. CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

FILTER_INPUTS = [
    Input("filter-ano",    "value"),
    Input("filter-sexo",   "value"),
    Input("filter-ops",    "value"),
    Input("filter-comuna", "value"),
]


@app.callback(
    Output("filter-ano",    "value"),
    Output("filter-sexo",   "value"),
    Output("filter-ops",    "value"),
    Output("filter-comuna", "value"),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(_):
    return [], [], [], []


@app.callback(
    Output("kpi-total",  "children"),
    Output("kpi-ops",    "children"),
    Output("kpi-edad",   "children"),
    Output("kpi-comuna", "children"),
    *FILTER_INPUTS,
)
def update_kpis(anos, sexos, ops_grupos, comunas):
    df = apply_filters(anos, sexos, ops_grupos, comunas)
    total      = f"{len(df):,}"
    top_ops    = df["NOM_667_OPS_GRUPO"].mode()[0] if len(df) else "—"
    top_ops_s  = (top_ops[:28] + "…") if len(top_ops) > 30 else top_ops
    edad_prom  = f"{df['EDAD_SIMPLE'].mean():.1f} años" if df["EDAD_SIMPLE"].notna().any() else "—"
    top_comuna = df["COMUNA_RES"].mode()[0] if len(df) else "—"

    return (
        kpi_card("Total Defunciones",    total,       "💀", "#E63946"),
        kpi_card("Causa más frecuente",  top_ops_s,   "🏥", "#457B9D"),
        kpi_card("Edad promedio",        edad_prom,   "🎂", "#2A9D8F"),
        kpi_card("Comuna líder",         top_comuna,  "🏙️", "#E9C46A"),
    )


@app.callback(Output("chart-ops-bar", "figure"),    *FILTER_INPUTS)
def upd_ops_bar(anos, sexos, ops_grupos, comunas):
    return fig_ops_bar(apply_filters(anos, sexos, ops_grupos, comunas))


@app.callback(Output("chart-edad-hist", "figure"),  *FILTER_INPUTS)
def upd_edad_hist(anos, sexos, ops_grupos, comunas):
    return fig_edad_hist(apply_filters(anos, sexos, ops_grupos, comunas))


@app.callback(Output("chart-seg-social", "figure"), *FILTER_INPUTS)
def upd_seg(anos, sexos, ops_grupos, comunas):
    return fig_seg_social(apply_filters(anos, sexos, ops_grupos, comunas))


@app.callback(Output("chart-evol-anual", "figure"),   *FILTER_INPUTS)
def upd_evol_anual(anos, sexos, ops_grupos, comunas):
    return fig_evolucion_anual(apply_filters(anos, sexos, ops_grupos, comunas))


@app.callback(Output("chart-evol-mensual", "figure"), *FILTER_INPUTS)
def upd_evol_mensual(anos, sexos, ops_grupos, comunas):
    return fig_evolucion_mensual(apply_filters(anos, sexos, ops_grupos, comunas))


@app.callback(Output("chart-ops-sexo", "figure"),        *FILTER_INPUTS)
def upd_ops_sexo(anos, sexos, ops_grupos, comunas):
    return fig_ops_sexo(apply_filters(anos, sexos, ops_grupos, comunas))


@app.callback(Output("chart-heatmap-etareo", "figure"),  *FILTER_INPUTS)
def upd_heatmap(anos, sexos, ops_grupos, comunas):
    return fig_heatmap_etareo(apply_filters(anos, sexos, ops_grupos, comunas))


@app.callback(Output("chart-nivel-edu", "figure"),        *FILTER_INPUTS)
def upd_nivel_edu(anos, sexos, ops_grupos, comunas):
    return fig_nivel_edu(apply_filters(anos, sexos, ops_grupos, comunas))


@app.callback(Output("chart-top10-comunas", "figure"),    *FILTER_INPUTS)
def upd_comunas(anos, sexos, ops_grupos, comunas):
    return fig_top10_comunas(apply_filters(anos, sexos, ops_grupos, comunas))


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
