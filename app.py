# =============================================================================
# Dashboard: Análisis de Mortalidad en Medellín (2012–2021)
# Variable objetivo: NOM_667_OPS_GRUPO
# Stack: Dash + Plotly + scikit-learn + Dash Bootstrap Components
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)

# =============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

df = pd.read_csv("defunciones_clean.csv")

# Paleta de colores institucional
OPS_COLORS = {
    "Enfermedades del sistema circulatorio":                "#1B4F72",
    "Neoplasias (Tumores)":                                 "#2E86C1",
    "Todas las demas enfermedades":                         "#5DADE2",
    "Enfermedades Transmisibles":                           "#85C1E9",
    "Causas externas":                                      "#7F8C8D",
    "Ciertas afecciones originadas en el periodo perinatal":"#AAB7B8",
    "Signos sintomas y afecciones mal definidas":           "#D5D8DC",
}

COLOR_SEQ = list(OPS_COLORS.values())

# Orden de grupos OPS por frecuencia
OPS_ORDER = df["NOM_667_OPS_GRUPO"].value_counts().index.tolist()

# =============================================================================
# 2. PREPARACIÓN PARA MODELOS
# =============================================================================

FEATURES = ["SEXO", "EDAD_SIMPLE", "EST_CIVIL", "SEG_SOCIAL", "NIVEL_EDU_GRUPO", "ANO", "MES"]
TARGET   = "NOM_667_OPS_GRUPO"

CAT_FEATURES = ["SEXO", "EST_CIVIL", "SEG_SOCIAL", "NIVEL_EDU_GRUPO"]
NUM_FEATURES = ["EDAD_SIMPLE", "ANO", "MES"]

df_model = df[FEATURES + [TARGET]].copy()
df_model["EDAD_SIMPLE"] = df_model["EDAD_SIMPLE"].fillna(df_model["EDAD_SIMPLE"].median())

# Label encoders
encoders = {}
for col in CAT_FEATURES:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    encoders[col] = le

# Encoder para target
le_target = LabelEncoder()
df_model[TARGET] = le_target.fit_transform(df_model[TARGET])

X = df_model[FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Random Forest ----
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    class_weight="balanced_subsample",   # manejo desbalance 55:1
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# ---- Decision Tree ----
dt_model = DecisionTreeClassifier(
    max_depth=8,
    class_weight="balanced",             # manejo desbalance 55:1
    random_state=42
)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# ---- Métricas (alineadas con notebook final) ----
def get_metrics(y_true, y_pred, name):
    return {
        "Modelo":         name,
        "Accuracy":       round(accuracy_score(y_true, y_pred) * 100, 2),
        "F1 Weighted ★":  round(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "F1 Macro":       round(f1_score(y_true, y_pred, average="macro",    zero_division=0) * 100, 2),
        "Recall Macro":   round(recall_score(y_true, y_pred, average="macro",zero_division=0) * 100, 2),
        "Precision W":    round(precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
    }

metrics_df = pd.DataFrame([
    get_metrics(y_test, rf_pred, "Random Forest"),
    get_metrics(y_test, dt_pred, "Árbol de Decisión"),
])

# Feature importances
feat_imp = pd.DataFrame({
    "Variable":    FEATURES,
    "Importancia": rf_model.feature_importances_
}).sort_values("Importancia", ascending=True)

# =============================================================================
# 3. FUNCIONES DE GRÁFICOS
# =============================================================================

def fig_ops_dist():
    counts = df["NOM_667_OPS_GRUPO"].value_counts().reset_index()
    counts.columns = ["Grupo OPS", "Cantidad"]
    counts["Porcentaje"] = (counts["Cantidad"] / counts["Cantidad"].sum() * 100).round(1)
    fig = px.bar(
        counts, x="Cantidad", y="Grupo OPS", orientation="h",
        color="Grupo OPS", color_discrete_map=OPS_COLORS,
        text=counts["Porcentaje"].apply(lambda x: f"{x}%"),
        title="Distribución de Causas de Muerte (Grupo OPS)"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_BASE, showlegend=False, height=380)
    return fig

def fig_sexo():
    counts = df["SEXO"].value_counts().reset_index()
    counts.columns = ["Sexo", "Cantidad"]
    fig = px.bar(counts, x="Sexo", y="Cantidad", color="Sexo",
                 color_discrete_map={"Masculino": "#2E86C1", "Femenino": "#F4A3E1", "Indeterminado": "#AAB7B8"},
                 text="Cantidad", title="Distribución por Sexo")
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_BASE, showlegend=False, height=350)
    return fig

def fig_edad():
    fig = px.histogram(df.dropna(subset=["EDAD_SIMPLE"]), x="EDAD_SIMPLE",
                       nbins=30, title="Distribución de Edad al Fallecimiento",
                       color_discrete_sequence=["#457B9D"])
    fig.update_layout(**LAYOUT_BASE, height=350)
    fig.update_xaxes(title="Edad (años)")
    fig.update_yaxes(title="Frecuencia")
    return fig

def fig_seg_social():
    counts = df[df["SEG_SOCIAL"] != "Sin info"]["SEG_SOCIAL"].value_counts().reset_index()
    counts.columns = ["Régimen", "Cantidad"]
    fig = px.bar(counts, x="Régimen", y="Cantidad", color="Régimen",
                 color_discrete_sequence=["#1B4F72","#2E86C1","#5DADE2","#85C1E9","#AED6F1"],
                 text="Cantidad", title="Distribución por Régimen de Seguridad Social")
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_BASE, showlegend=False, height=350)
    return fig

def fig_edu():
    orden = ["Básica", "Media", "Técnico/Tecnológico", "Superior", "Sin info"]
    counts = df["NIVEL_EDU_GRUPO"].value_counts().reindex(orden).dropna().reset_index()
    counts.columns = ["Nivel", "Cantidad"]
    fig = px.bar(counts, x="Nivel", y="Cantidad", color="Nivel",
                 color_discrete_sequence=["#1B4F72","#2E86C1","#5DADE2","#85C1E9","#D6EAF8"],
                 text="Cantidad", title="Distribución por Nivel Educativo")
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_BASE, showlegend=False, height=350)
    return fig

def fig_anual():
    anual = df.groupby("ANO").size().reset_index(name="Defunciones")
    fig = px.line(anual, x="ANO", y="Defunciones", markers=True,
                  title="Total de Defunciones por Año",
                  color_discrete_sequence=["#2E86C1"])
    fig.update_layout(**LAYOUT_BASE, height=350)
    fig.update_xaxes(dtick=1)
    return fig

# ---- Bivariado ----
def fig_ops_sexo():
    ct = pd.crosstab(df["NOM_667_OPS_GRUPO"], df["SEXO"], normalize="index") * 100
    ct = ct.reset_index()
    ct_melted = ct.melt(id_vars="NOM_667_OPS_GRUPO", var_name="Sexo", value_name="Porcentaje")
    fig = px.bar(ct_melted, x="NOM_667_OPS_GRUPO", y="Porcentaje", color="Sexo",
                 barmode="stack",
                 color_discrete_map={"Masculino": "#2E86C1", "Femenino": "#F4A3E1", "Indeterminado": "#AAB7B8"},
                 title="Distribución de Sexo por Grupo OPS (%)")
    fig.update_xaxes(title="", tickangle=-25)
    fig.update_layout(**LAYOUT_BASE, height=420,
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#2C3E50")))
    return fig

def fig_ops_edad():
    df_plot = df.dropna(subset=["EDAD_SIMPLE"])
    fig = px.box(df_plot, x="NOM_667_OPS_GRUPO", y="EDAD_SIMPLE",
                 color="NOM_667_OPS_GRUPO", color_discrete_map=OPS_COLORS,
                 title="Distribución de Edad por Grupo OPS")
    fig.update_xaxes(title="", tickangle=-25)
    fig.update_yaxes(title="Edad (años)")
    fig.update_layout(**LAYOUT_BASE, showlegend=False, height=430)
    return fig

def fig_ops_anual():
    evol = df.groupby(["ANO", "NOM_667_OPS_GRUPO"]).size().reset_index(name="Defunciones")
    fig = px.line(evol, x="ANO", y="Defunciones", color="NOM_667_OPS_GRUPO",
                  color_discrete_map=OPS_COLORS, markers=True,
                  title="Evolución Anual de Defunciones por Grupo OPS")
    fig.update_xaxes(dtick=1, title="Año")
    fig.update_layout(**LAYOUT_BASE, height=430,
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#2C3E50")))
    return fig

def fig_ops_seg():
    df_s = df[df["SEG_SOCIAL"] != "Sin info"]
    ct = pd.crosstab(df_s["NOM_667_OPS_GRUPO"], df_s["SEG_SOCIAL"], normalize="index") * 100
    ct_melted = ct.reset_index().melt(id_vars="NOM_667_OPS_GRUPO", var_name="Régimen", value_name="Porcentaje")
    fig = px.bar(ct_melted, x="NOM_667_OPS_GRUPO", y="Porcentaje", color="Régimen",
                 barmode="stack",
                 color_discrete_sequence=["#1B4F72","#2E86C1","#5DADE2","#85C1E9","#AED6F1"],
                 title="Seguridad Social por Grupo OPS (%)")
    fig.update_xaxes(title="", tickangle=-25)
    fig.update_layout(**LAYOUT_BASE, height=420,
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#2C3E50")))
    return fig

def fig_heatmap_edad_ops():
    # Etareos ordenados correctamente
    orden_etareo = [
        "<1","1-4","5-9","10-14","15-19","20-24","25-29","30-34",
        "35-39","40-44","45-49","50-54","55-59","60-64","65-69",
        "70-74","75-79","80-84","85-89","90-94","95-99","100 y más"
    ]
    df_e = df[df["ETAREO_QUIN"].notna() & ~df["ETAREO_QUIN"].isin(["Sin informacion","sin informacion"])]
    ct = pd.crosstab(df_e["ETAREO_QUIN"], df_e["NOM_667_OPS_GRUPO"], normalize="columns") * 100
    # Filtrar solo filas que existan
    idx_valid = [e for e in orden_etareo if e in ct.index]
    ct = ct.reindex(idx_valid)

    fig = px.imshow(ct, aspect="auto", color_continuous_scale="RdYlBu_r",
                    title="Grupo Etario vs Grupo OPS (% por causa)",
                    labels=dict(color="%"))
    fig.update_layout(**LAYOUT_BASE, height=480)
    return fig

# ---- Modelos ----
def fig_conf_matrix(model_name):
    preds = rf_pred if model_name == "Random Forest" else dt_pred
    labels = le_target.classes_
    cm = confusion_matrix(y_test, preds)
    cm_pct = (cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100).round(1)
    # Etiquetas cortas
    short = [l[:20]+"…" if len(l) > 20 else l for l in labels]
    fig = px.imshow(cm_pct, x=short, y=short,
                    color_continuous_scale="Blues", aspect="auto",
                    title=f"Matriz de Confusión — {model_name} (%)",
                    labels=dict(color="%"))
    fig.update_layout(**LAYOUT_BASE, height=460)
    return fig

def fig_feat_imp():
    fig = px.bar(feat_imp, x="Importancia", y="Variable", orientation="h",
                 color="Importancia", color_continuous_scale="Teal",
                 title="Importancia de Variables — Random Forest")
    fig.update_layout(**LAYOUT_BASE, showlegend=False, height=380)
    return fig

# Layout base para gráficos
LAYOUT_BASE = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F8F9FA",
    font=dict(family="DM Sans, sans-serif", size=12, color="#2C3E50"),
    title_font=dict(size=14, color="#1A2B3C", family="DM Sans, sans-serif"),
    margin=dict(l=20, r=20, t=58, b=20),
)
_LEGEND = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#2C3E50"))

# =============================================================================
# 4. APP LAYOUT
# =============================================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&family=IBM+Plex+Mono&display=swap",
        dbc.icons.BOOTSTRAP,
    ],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.title = "Dashboard · Mortalidad Medellín"
server = app.server

# ── CSS inyectado inline (sin archivo externo) ─────────────────────────────
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* ── Reset & base ─────────────────────────────── */
            *, *::before, *::after { box-sizing: border-box; }
            body {
                font-family: "DM Sans", sans-serif !important;
                background-color: #F0F4F8 !important;
                color: #1E2A38 !important;
                margin: 0;
            }

            /* ── Sidebar ──────────────────────────────────── */
            .sidebar-brand-title {
                font-family: "DM Serif Display", serif;
                font-size: 1.05rem;
                color: #FFFFFF;
                margin: 0;
                line-height: 1.2;
            }
            .sidebar-brand-sub {
                font-size: 0.68rem;
                color: #8FA3BD;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            .sidebar-section-label {
                font-size: 0.62rem;
                font-weight: 600;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                color: #4E6480;
                padding: 16px 12px 6px 12px;
            }

            /* ── Nav buttons ──────────────────────────────── */
            .nav-btn {
                display: flex !important;
                align-items: center;
                width: 100% !important;
                text-align: left !important;
                padding: 10px 14px !important;
                margin-bottom: 3px !important;
                border: none !important;
                border-radius: 8px !important;
                font-size: 0.875rem !important;
                font-weight: 400 !important;
                font-family: "DM Sans", sans-serif !important;
                color: #A8BCCF !important;
                background: transparent !important;
                transition: background 0.18s ease, color 0.18s ease, transform 0.1s ease !important;
                cursor: pointer;
                gap: 10px;
                position: relative;
                box-shadow: none !important;
            }
            .nav-btn:hover {
                background: rgba(46, 134, 193, 0.12) !important;
                color: #FFFFFF !important;
                transform: translateX(2px);
            }
            .nav-btn-active {
                background: rgba(46, 134, 193, 0.22) !important;
                color: #FFFFFF !important;
                font-weight: 600 !important;
                border-left: 3px solid #2E86C1 !important;
                padding-left: 11px !important;
            }
            .nav-btn-active:hover {
                background: rgba(46, 134, 193, 0.30) !important;
                transform: translateX(0);
            }
            .nav-btn .bi {
                font-size: 0.95rem;
                flex-shrink: 0;
            }

            /* ── Section titles ───────────────────────────── */
            .section-title {
                font-family: "DM Serif Display", serif !important;
                font-size: 1.75rem !important;
                color: #1A2B3C !important;
                font-weight: 400 !important;
                margin-bottom: 4px !important;
                letter-spacing: -0.01em;
            }
            .section-subtitle {
                font-size: 0.92rem;
                color: #5A7290;
                margin-bottom: 28px;
                font-weight: 400;
            }

            /* ── Cards ────────────────────────────────────── */
            .stat-card {
                background: #FFFFFF !important;
                border: 1px solid #E4ECF4 !important;
                border-radius: 12px !important;
                box-shadow: 0 1px 4px rgba(0,0,0,0.05), 0 4px 16px rgba(0,0,0,0.04) !important;
                transition: box-shadow 0.2s ease !important;

                position: relative;
                z-index: 1;
            }

            .stat-card:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.08), 0 8px 24px rgba(0,0,0,0.06) !important;
                z-index: 10;
            }

            /* ── KPI cards ────────────────────────────────── */
            .kpi-card {
                background: #FFFFFF;
                border-radius: 14px;
                border: 1px solid #E4ECF4;
                padding: 22px 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                transition: box-shadow 0.2s ease, transform 0.2s ease;
                position: relative;
                overflow: hidden;
            }
            .kpi-card:hover {
                box-shadow: 0 6px 20px rgba(0,0,0,0.09);
                transform: translateY(-3px);
            }
            .kpi-card::before {
                content: "";
                position: absolute;
                top: 0; left: 0;
                width: 4px; height: 100%;
                border-radius: 14px 0 0 14px;
            }
            .kpi-blue::before  { background: #2E86C1; }
            .kpi-teal::before  { background: #2A9D8F; }
            .kpi-amber::before { background: #E9A128; }
            .kpi-slate::before { background: #546E8A; }

            .kpi-value {
                font-family: "DM Serif Display", serif;
                font-size: 2.2rem;
                line-height: 1;
                margin-bottom: 4px;
            }
            .kpi-label {
                font-size: 0.78rem;
                font-weight: 500;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                color: #7A93AD;
            }
            .kpi-icon {
                font-size: 1.4rem;
                margin-bottom: 10px;
                display: block;
            }

            /* ── Context alert (intro) ────────────────────── */
            .context-alert {
                background: #F7FAFD !important;
                border: 1px solid #D4E6F5 !important;
                border-left: 4px solid #2E86C1 !important;
                border-radius: 10px !important;
                color: #2C3E50 !important;
                padding: 20px 24px !important;
            }

            /* ── Divider ──────────────────────────────────── */
            .sidebar-divider {
                border-color: #1E3050 !important;
                margin: 12px 0 !important;
            }

            /* ── Table ────────────────────────────────────── */
            .tabla-vars { width: 100%; border-collapse: collapse; }
            .tabla-vars th {
                background: #EAF0F6;
                color: #3D5266;
                font-size: 0.72rem;
                font-weight: 600;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                border-top: none;
                border-bottom: 2px solid #C8DEF5;
                padding: 10px 14px;
            }
            .tabla-vars td {
                color: #3D5266;
                font-size: 0.84rem;
                border-bottom: 1px solid #EBF0F5;
                padding: 9px 14px;
                vertical-align: middle;
                background: #FFFFFF;
            }
            .tabla-vars tbody tr:nth-child(even) td { background: #F7FAFD; }
            .tabla-vars tbody tr:hover td {
                background: #EFF6FF;
                transition: background 0.15s ease;
            }

            /* ── Scrollbar ────────────────────────────────── */
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: transparent; }
            ::-webkit-scrollbar-thumb { background: #B0C4D8; border-radius: 3px; }
            .sidebar-author-link:hover {
                color: #FFFFFF !important;
                background: rgba(46,134,193,0.12) !important;
            }
            /* ── Gradient cards (problema / objetivos) ────── */
            .gradient-card-red {
                background: linear-gradient(135deg, #8B1A1A 0%, #C0392B 50%, #922B21 100%);
                border-radius: 14px; padding: 32px 36px; margin-bottom: 24px;
                box-shadow: 0 4px 20px rgba(192,57,43,0.25);
            }
            .gradient-card-green {
                background: linear-gradient(135deg, #1A5C3A 0%, #27AE60 55%, #1E8449 100%);
                border-radius: 14px; padding: 32px 36px; margin-bottom: 28px;
                box-shadow: 0 4px 20px rgba(39,174,96,0.22);
            }
            /* ── Fix dropdowns Dash ───────────────────── */
            .Select-menu-outer,
            .Select-menu,
            .Select-control,
            .Select {
                z-index: 9999 !important;
            }

            .VirtualizedSelectOption {
                z-index: 9999 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
# ---------- Sidebar ----------
def make_nav_btn(icon, label, section_id, active=False):
    return dbc.Button(
        [
            html.I(className=f"bi {icon}"),
            html.Span(label),
        ],
        id=f"btn-{section_id}",
        n_clicks=0,
        className=f"nav-btn btn btn-link {'nav-btn-active' if active else ''}",
    )

sidebar = html.Div([

    # ── Brand ─────────────────────────────────────────────────────────────
    html.Div([
        html.Div(
            html.I(className="bi bi-heart-pulse-fill",
                   style={"fontSize": "1.5rem", "color": "#2E86C1"}),
            style={
                "width": "40px", "height": "40px",
                "background": "rgba(46,134,193,0.12)",
                "borderRadius": "10px",
                "display": "flex", "alignItems": "center", "justifyContent": "center",
                "flexShrink": "0",
            }
        ),
        html.Div([
            html.P("Mortalidad Medellín", className="sidebar-brand-title"),
            html.Span("2012 – 2021", className="sidebar-brand-sub"),
        ]),
    ], style={
        "display": "flex", "alignItems": "center",
        "gap": "12px", "padding": "4px 4px 8px 4px",
    }),

    html.Hr(className="sidebar-divider"),

    # ── Nav ───────────────────────────────────────────────────────────────
    html.Div("Navegación", className="sidebar-section-label"),
    html.Div([
        make_nav_btn("bi-house-fill",     "Introducción", "intro",      True),
        make_nav_btn("bi-question-circle", "Problema",     "problema"),
        make_nav_btn("bi-bullseye",        "Objetivos",    "objetivos"),
        make_nav_btn("bi-bar-chart-line",  "Univariado",   "univariado"),
        make_nav_btn("bi-diagram-3",       "Bivariado",    "bivariado"),
        make_nav_btn("bi-cpu",             "Modelado",     "modelo"),
        make_nav_btn("bi-check2-circle",   "Conclusiones", "conclusiones"),
    ], style={"padding": "0 4px"}),

    html.Hr(className="sidebar-divider"),

    # ── Footer info ───────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.I(className="bi bi-database me-2",
                   style={"color": "#4E6480", "fontSize": "0.75rem"}),
            html.Span("145,377 registros",
                      style={"fontSize": "0.75rem", "color": "#5A7290"}),
        ], style={"marginBottom": "5px"}),
        html.Div([
            html.I(className="bi bi-columns-gap me-2",
                   style={"color": "#4E6480", "fontSize": "0.75rem"}),
            html.Span("11 variables",
                      style={"fontSize": "0.75rem", "color": "#5A7290"}),
        ], style={"marginBottom": "14px"}),
        html.Hr(style={"borderColor": "#1E3050", "margin": "0 0 12px 0"}),
        html.Div([
            html.I(className="bi bi-people-fill me-2",
                   style={"color": "#4E6480", "fontSize": "0.72rem"}),
            html.Span("Autores", style={
                "fontSize": "0.62rem", "fontWeight": "600", "letterSpacing": "0.08em",
                "textTransform": "uppercase", "color": "#4E6480",
            }),
        ], style={"marginBottom": "8px"}),
        html.A([
            html.I(className="bi bi-github me-2",
                   style={"fontSize": "0.72rem"}),
            "Camilo González",
        ], href="https://github.com/spidermil0", target="_blank", style={
            "display": "block", "fontSize": "0.76rem", "color": "#7A93AD",
            "textDecoration": "none", "padding": "4px 2px", "borderRadius": "5px",
            "transition": "color 0.15s ease, background 0.15s ease",
            "marginBottom": "2px",
        }, className="sidebar-author-link"),
        html.A([
            html.I(className="bi bi-github me-2",
                   style={"fontSize": "0.72rem"}),
            "Rubén Esguerra",
        ], href="https://github.com/RubenEsg", target="_blank", style={
            "display": "block", "fontSize": "0.76rem", "color": "#7A93AD",
            "textDecoration": "none", "padding": "4px 2px", "borderRadius": "5px",
            "transition": "color 0.15s ease, background 0.15s ease",
        }, className="sidebar-author-link"),
    ], style={"padding": "4px 12px"}),

], style={
    "position": "fixed",
    "top": "0", "left": "0",
    "height": "100vh", "width": "224px",
    "backgroundColor": "#0F1E2E",
    "padding": "24px 16px",
    "borderRight": "1px solid #182C40",
    "display": "flex", "flexDirection": "column",
    "overflowY": "auto",
})

# ---------- Secciones ----------

def section_intro():
    # ── KPI data ───────────────────────────────────────────────────────────────
    kpis = [
        ("145,377",  "Registros totales",   "bi-database-fill",      "#2E86C1",  "kpi-blue"),
        ("10 años",  "Período analizado",   "bi-calendar2-range",    "#2A9D8F",  "kpi-teal"),
        ("7 grupos", "Categorías OPS",      "bi-diagram-3-fill",     "#E9A128",  "kpi-amber"),
        ("11",       "Variables analizadas","bi-table",               "#546E8A",  "kpi-slate"),
    ]
    kpi_cards = dbc.Row([
        dbc.Col(
            html.Div([
                html.I(className=f"bi {icon} kpi-icon", style={"color": color}),
                html.Div(val, className="kpi-value", style={"color": color}),
                html.Div(label, className="kpi-label"),
            ], className=f"kpi-card {cls}"),
            width=6, md=3, className="mb-3"
        )
        for val, label, icon, color, cls in kpis
    ], className="mb-4 g-3")

    # ── Variables table ────────────────────────────────────────────────────────
    vars_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Variable"),
            html.Th("Tipo"),
            html.Th("Descripción"),
        ])),
        html.Tbody([
            html.Tr([html.Td("NOM_667_OPS_GRUPO"), html.Td(dbc.Badge("Objetivo", color="light", pill=True, style={"background":"#FDECEA","color":"#922B21","fontWeight":"600","border":"1px solid #F1948A","fontSize":"0.75rem"})), html.Td("Grupo de causa de muerte — 7 categorías OPS")]),
            html.Tr([html.Td("ANO / MES"),          html.Td(dbc.Badge("Temporal", color="light", pill=True, style={"background":"#D4E6F5","color":"#1A4A7A","fontWeight":"500","fontSize":"0.75rem"})), html.Td("Año (2012–2021) y mes de fallecimiento")]),
            html.Tr([html.Td("SEXO"),               html.Td(dbc.Badge("Demog.",   color="light", pill=True, style={"background":"#D5EAE7","color":"#1A4A3A","fontWeight":"500","fontSize":"0.75rem"})), html.Td("Masculino / Femenino / Indeterminado")]),
            html.Tr([html.Td("EDAD_SIMPLE"),        html.Td(dbc.Badge("Numérica", color="light", pill=True, style={"background":"#FDEBD0","color":"#7D4A00","fontWeight":"500","fontSize":"0.75rem"})), html.Td("Edad en años al momento del fallecimiento")]),
            html.Tr([html.Td("ETAREO_QUIN"),        html.Td(dbc.Badge("Categ.",   color="light", pill=True, style={"background":"#EAF0F6","color":"#3D5266","fontWeight":"500","fontSize":"0.75rem"})), html.Td("Grupo etario quinquenal")]),
            html.Tr([html.Td("EST_CIVIL"),          html.Td(dbc.Badge("Categ.",   color="light", pill=True, style={"background":"#EAF0F6","color":"#3D5266","fontWeight":"500","fontSize":"0.75rem"})), html.Td("Estado civil del fallecido")]),
            html.Tr([html.Td("SEG_SOCIAL"),         html.Td(dbc.Badge("Categ.",   color="light", pill=True, style={"background":"#EAF0F6","color":"#3D5266","fontWeight":"500","fontSize":"0.75rem"})), html.Td("Régimen de seguridad social")]),
            html.Tr([html.Td("NIVEL_EDU_GRUPO"),    html.Td(dbc.Badge("Categ.",   color="light", pill=True, style={"background":"#EAF0F6","color":"#3D5266","fontWeight":"500","fontSize":"0.75rem"})), html.Td("Nivel educativo agrupado")]),
            html.Tr([html.Td("COMUNA_RES"),         html.Td(dbc.Badge("Geog.",    color="light", pill=True, style={"background":"#E8EAF6","color":"#2C3A7A","fontWeight":"500","fontSize":"0.75rem"})), html.Td("Comuna de residencia (22 comunas)")]),
        ])
    ], bordered=False, size="sm", className="tabla-vars")

    return html.Div([

        # ── Header ──────────────────────────────────────────────────────────
        html.Div([
            html.H2("Análisis de Mortalidad en Medellín", className="section-title"),
            html.P(
                "Exploración estadística de 145,377 defunciones registradas en Medellín "
                "entre 2012 y 2021, clasificadas según grupos de la Organización "
                "Panamericana de la Salud (OPS).",
                className="section-subtitle"
            ),
        ], style={"marginBottom": "8px"}),

        # ── KPI cards ────────────────────────────────────────────────────────
        kpi_cards,

        # ── Contexto ─────────────────────────────────────────────────────────
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("Contexto del problema",
                                  style={"fontSize": "0.7rem", "fontWeight": "600",
                                         "letterSpacing": "0.08em", "textTransform": "uppercase",
                                         "color": "#2E86C1", "display": "block", "marginBottom": "8px"}),
                        html.P(
                            "La mortalidad urbana es un indicador clave del estado de salud pública de una ciudad. "
                            "Medellín, con más de 2.5 millones de habitantes, registra anualmente decenas de miles de "
                            "defunciones clasificadas según la OPS. Comprender qué variables demográficas, "
                            "socioeconómicas y temporales se asocian con cada tipo de causa de muerte permite "
                            "diseñar políticas de salud más focalizadas y eficientes.",
                            style={"fontSize": "0.88rem", "color": "#3D5266", "lineHeight": "1.65", "marginBottom": "0"}
                        ),
                    ])
                ], md=8),
                dbc.Col([
                    html.Div([
                        html.Span("Fuente de datos",
                                  style={"fontSize": "0.7rem", "fontWeight": "600",
                                         "letterSpacing": "0.08em", "textTransform": "uppercase",
                                         "color": "#2A9D8F", "display": "block", "marginBottom": "8px"}),
                        html.P("Sistema de Estadísticas Vitales de Medellín",
                               style={"fontSize": "0.88rem", "fontWeight": "600", "color": "#1E2A38", "marginBottom": "6px"}),
                        html.P("Registros: 2012 – 2021 · Limpieza aplicada · Variable objetivo: NOM_667_OPS_GRUPO",
                               style={"fontSize": "0.78rem", "color": "#5A7290", "marginBottom": "0"}),
                    ])
                ], md=4),
            ])
        ], className="stat-card", style={"padding": "24px 28px", "marginBottom": "28px"}),

        # ── Variables table ──────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.I(className="bi bi-table me-2",
                       style={"color": "#2E86C1", "fontSize": "1rem"}),
                html.Span("Variables del Dataset Post-Limpieza",
                          style={"fontFamily": "DM Sans, sans-serif", "fontWeight": "600",
                                 "fontSize": "1rem", "color": "#1A2B3C"}),
            ], style={"marginBottom": "16px", "display": "flex", "alignItems": "center"}),
            vars_table,
        ], className="stat-card", style={"padding": "24px 28px"}),

    ])


def section_problema():
    return html.Div([
        html.H2("El Problema de Investigación", className="section-title"),
        html.P("¿Qué factores explican la distribución de causas de muerte en una ciudad?",
               className="section-subtitle"),

        # ── Pregunta central con degradado rojo ──────────────────────────────
        html.Div([
            html.Div([
                html.Span("Pregunta Central", style={
                    "fontSize": "0.68rem", "fontWeight": "600", "letterSpacing": "0.1em",
                    "textTransform": "uppercase", "color": "#E07070", "display": "block",
                    "marginBottom": "12px",
                }),
                html.P(
                    "¿Qué factores demográficos, socioeconómicos y temporales determinan "
                    "el grupo de causa de muerte (NOM_667_OPS_GRUPO) de una defunción registrada en Medellín?",
                    style={"fontSize": "1.15rem", "fontFamily": "DM Serif Display, serif",
                           "color": "#FFFFFF", "lineHeight": "1.6", "marginBottom": "0"},
                ),
            ], style={"position": "relative", "zIndex": "1"}),
        ], style={
            "background": "linear-gradient(135deg, #8B1A1A 0%, #C0392B 50%, #922B21 100%)",
            "borderRadius": "14px",
            "padding": "32px 36px",
            "marginBottom": "24px",
            "boxShadow": "0 4px 20px rgba(192,57,43,0.25)",
        }),

        # ── KPI highlights del problema ───────────────────────────────────────
        dbc.Row([
            dbc.Col(html.Div([
                html.Div("~28%", style={"fontFamily":"DM Serif Display, serif",
                    "fontSize":"1.9rem","color":"#C0392B","lineHeight":"1"}),
                html.Div("enf. circulatorias", style={"fontSize":"0.7rem","color":"#7A93AD",
                    "textTransform":"uppercase","letterSpacing":"0.05em","marginTop":"5px"}),
                html.Div("causa más frecuente", style={"fontSize":"0.72rem","color":"#A8BCCF","marginTop":"2px"}),
            ], className="stat-card", style={"padding":"20px","textAlign":"center"}), md=3, className="mb-3"),

            dbc.Col(html.Div([
                html.Div("65%", style={"fontFamily":"DM Serif Display, serif",
                    "fontSize":"1.9rem","color":"#2E86C1","lineHeight":"1"}),
                html.Div("mortalidad en 3 grupos", style={"fontSize":"0.7rem","color":"#7A93AD",
                    "textTransform":"uppercase","letterSpacing":"0.05em","marginTop":"5px"}),
                html.Div("circ. · neoplasias · externas", style={"fontSize":"0.72rem","color":"#A8BCCF","marginTop":"2px"}),
            ], className="stat-card", style={"padding":"20px","textAlign":"center"}), md=3, className="mb-3"),

            dbc.Col(html.Div([
                html.Div("55:1", style={"fontFamily":"DM Serif Display, serif",
                    "fontSize":"1.9rem","color":"#E9A128","lineHeight":"1"}),
                html.Div("ratio de desbalance", style={"fontSize":"0.7rem","color":"#7A93AD",
                    "textTransform":"uppercase","letterSpacing":"0.05em","marginTop":"5px"}),
                html.Div("clase mayor vs. menor", style={"fontSize":"0.72rem","color":"#A8BCCF","marginTop":"2px"}),
            ], className="stat-card", style={"padding":"20px","textAlign":"center"}), md=3, className="mb-3"),

            dbc.Col(html.Div([
                html.Div("3×", style={"fontFamily":"DM Serif Display, serif",
                    "fontSize":"1.9rem","color":"#2A9D8F","lineHeight":"1"}),
                html.Div("hombres en causas externas", style={"fontSize":"0.7rem","color":"#7A93AD",
                    "textTransform":"uppercase","letterSpacing":"0.05em","marginTop":"5px"}),
                html.Div("vs. mujeres", style={"fontSize":"0.72rem","color":"#A8BCCF","marginTop":"2px"}),
            ], className="stat-card", style={"padding":"20px","textAlign":"center"}), md=3, className="mb-3"),
        ], className="g-3 mb-3"),

        # ── Mini timeline COVID ───────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("Contexto temporal", style={
                    "fontSize":"0.68rem","fontWeight":"600","letterSpacing":"0.1em",
                    "textTransform":"uppercase","color":"#5A7290","display":"block","marginBottom":"16px",
                }),
                html.Div([
                    # 2019
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-circle-fill",
                                   style={"color":"#2A9D8F","fontSize":"0.65rem"}),
                        ], style={"marginBottom":"8px"}),
                        html.Div("2019", style={"fontFamily":"DM Serif Display, serif",
                            "fontSize":"1.1rem","color":"#1A2B3C","fontWeight":"400"}),
                        html.Div("Patrón estable", style={"fontSize":"0.72rem","color":"#7A93AD","marginTop":"3px"}),
                        html.Div("Circ. y neoplasias dominan", style={"fontSize":"0.68rem","color":"#A8BCCF","marginTop":"2px"}),
                    ], style={"textAlign":"center","flex":"1"}),

                    # flecha
                    html.Div([
                        html.Div(style={"height":"2px","background":"linear-gradient(90deg,#2A9D8F,#E9A128)","margin":"0 8px","marginTop":"8px"}),
                    ], style={"flex":"1","display":"flex","flexDirection":"column","justifyContent":"flex-start","paddingTop":"2px"}),

                    # 2020
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-exclamation-circle-fill",
                                   style={"color":"#E9A128","fontSize":"0.8rem"}),
                        ], style={"marginBottom":"8px"}),
                        html.Div("2020", style={"fontFamily":"DM Serif Display, serif",
                            "fontSize":"1.1rem","color":"#1A2B3C","fontWeight":"400"}),
                        html.Div("Inicio pandemia", style={"fontSize":"0.72rem","color":"#7A93AD","marginTop":"3px"}),
                        html.Div("↑ causas respiratorias", style={"fontSize":"0.68rem","color":"#E9A128","marginTop":"2px"}),
                    ], style={"textAlign":"center","flex":"1"}),

                    # flecha
                    html.Div([
                        html.Div(style={"height":"2px","background":"linear-gradient(90deg,#E9A128,#C0392B)","margin":"0 8px","marginTop":"8px"}),
                    ], style={"flex":"1","display":"flex","flexDirection":"column","justifyContent":"flex-start","paddingTop":"2px"}),

                    # 2021
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-circle-fill",
                                   style={"color":"#C0392B","fontSize":"0.65rem"}),
                        ], style={"marginBottom":"8px"}),
                        html.Div("2021", style={"fontFamily":"DM Serif Display, serif",
                            "fontSize":"1.1rem","color":"#1A2B3C","fontWeight":"400"}),
                        html.Div("Impacto sostenido", style={"fontSize":"0.72rem","color":"#7A93AD","marginTop":"3px"}),
                        html.Div("Alteración del patrón habitual", style={"fontSize":"0.68rem","color":"#C0392B","marginTop":"2px"}),
                    ], style={"textAlign":"center","flex":"1"}),

                ], style={"display":"flex","alignItems":"flex-start","gap":"4px"}),
            ]),
        ], className="stat-card", style={"padding":"24px 28px","marginBottom":"24px"}),

        # ── Dos columnas ─────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(html.Div([
                html.Div([
                    
                    html.I(className="bi bi-search me-2",
                           style={"color": "#2E86C1", "fontSize": "0.95rem"}),
                    html.Span("Sub-preguntas analíticas", style={
                        "fontWeight": "600", "fontSize": "0.9rem", "color": "#1A2B3C",
                    }),
                ], style={"marginBottom": "16px", "display": "flex", "alignItems": "center"}),
                html.Ul([
                    html.Li(t, style={"marginBottom": "10px", "fontSize": "0.87rem",
                                      "color": "#3D5266", "lineHeight": "1.55"})
                    for t in [
                        "¿Cómo varía la distribución de causas de muerte según sexo y edad?",
                        "¿Existen diferencias según el régimen de seguridad social o nivel educativo?",
                        "¿Ha cambiado la proporción de causas de muerte entre 2012 y 2021?",
                        "¿Es posible predecir el grupo OPS con las variables disponibles?",
                    ]
                ], style={"paddingLeft": "18px", "marginBottom": "0"}),
            ], className="stat-card", style={"padding": "24px 28px", "height": "100%"}),
            md=6, className="mb-3"),

            dbc.Col(html.Div([
                html.Div([
                    html.I(className="bi bi-exclamation-circle me-2",
                           style={"color": "#546E8A", "fontSize": "0.95rem"}),
                    html.Span("Relevancia del problema", style={
                        "fontWeight": "600", "fontSize": "0.9rem", "color": "#1A2B3C",
                    }),
                ], style={"marginBottom": "16px", "display": "flex", "alignItems": "center"}),
                html.Ul([
                    html.Li(t, style={"marginBottom": "10px", "fontSize": "0.87rem",
                                      "color": "#3D5266", "lineHeight": "1.55"})
                    for t in [
                        "Enfermedades circulatorias y neoplasias representan más del 50% de las muertes.",
                        "La mortalidad por causas externas afecta desproporcionadamente a hombres jóvenes.",
                        "El régimen de seguridad social refleja desigualdades en acceso a salud.",
                        "Modelos predictivos pueden apoyar sistemas de alerta temprana en salud pública.",
                    ]
                ], style={"paddingLeft": "18px", "marginBottom": "0"}),
            ], className="stat-card", style={"padding": "24px 28px", "height": "100%"}),
            md=6, className="mb-3"),
        ], className="g-3"),
    ])


def section_objetivos():
    esp = [
        ("bi-1-circle-fill", "#2E86C1", "Caracterizar la distribución de causas de muerte según grupos OPS en Medellín."),
        ("bi-2-circle-fill", "#2A9D8F", "Explorar relaciones entre causas de muerte y variables demográficas (sexo, edad, estado civil)."),
        ("bi-3-circle-fill", "#E9A128", "Analizar el comportamiento temporal de la mortalidad entre 2012 y 2021."),
        ("bi-4-circle-fill", "#546E8A", "Comparar el desempeño predictivo de Random Forest y Árbol de Decisión."),
        ("bi-5-circle-fill", "#2A9D8F", "Construir una herramienta interactiva de predicción del grupo OPS."),
    ]
    return html.Div([
        html.H2("Objetivos del Análisis", className="section-title"),
        html.P("Qué queremos descubrir, medir y construir con este dataset.",
               className="section-subtitle"),

        # ── Objetivo general con degradado verde ─────────────────────────────
        html.Div([
            html.Span("Objetivo General", style={
                "fontSize": "0.68rem", "fontWeight": "600", "letterSpacing": "0.1em",
                "textTransform": "uppercase", "color": "#A8DFCA", "display": "block",
                "marginBottom": "12px",
            }),
            html.P(
                "Desarrollar un dashboard analítico que permita explorar los patrones de mortalidad "
                "en Medellín e implementar modelos de clasificación capaces de predecir el grupo OPS "
                "de una defunción a partir de variables demográficas y socioeconómicas.",
                style={"fontSize": "1.12rem", "fontFamily": "DM Serif Display, serif",
                       "color": "#FFFFFF", "lineHeight": "1.65", "marginBottom": "0"},
            ),
        ], style={
            "background": "linear-gradient(135deg, #1A5C3A 0%, #27AE60 55%, #1E8449 100%)",
            "borderRadius": "14px",
            "padding": "32px 36px",
            "marginBottom": "28px",
            "boxShadow": "0 4px 20px rgba(39,174,96,0.22)",
        }),

        # ── Pipeline del proyecto ─────────────────────────────────────────────
        html.Div("Pipeline del proyecto", style={
            "fontSize": "0.7rem", "fontWeight": "600", "letterSpacing": "0.08em",
            "textTransform": "uppercase", "color": "#5A7290", "marginBottom": "16px",
        }),
        html.Div([
            dbc.Row([
                *[dbc.Col(html.Div([
                    html.Div(
                        html.I(className=f"bi {ico}", style={"fontSize":"1.6rem","color":col}),
                        style={"width":"52px","height":"52px","borderRadius":"12px",
                               "background":bg,"display":"flex","alignItems":"center",
                               "justifyContent":"center","margin":"0 auto 10px auto"}),
                    html.Div(label, style={"fontSize":"0.78rem","fontWeight":"600",
                                          "color":"#1A2B3C","textAlign":"center","marginBottom":"2px"}),
                    html.Div(sub, style={"fontSize":"0.68rem","color":"#7A93AD","textAlign":"center"}),
                ]), md=True, className="mb-2")
                for ico, col, bg, label, sub in [
                    ("bi-bar-chart-line","#2E86C1","#EBF5FB","EDA","Distribuciones"),
                    ("bi-diagram-3",     "#2A9D8F","#E8F8F5","Relaciones","Variables cruzadas"),
                    ("bi-clock-history", "#E9A128","#FEF9E7","Temporalidad","2012 – 2021"),
                    ("bi-cpu",           "#546E8A","#EAECEE","Modelado","RF vs DT"),
                    ("bi-bullseye",      "#2A9D8F","#E8F8F5","Predicción","Herramienta interactiva"),
                ]],
                # flechas entre pasos
                *[],
            ], className="g-2 align-items-start"),
        ], className="stat-card", style={"padding":"24px 28px","marginBottom":"28px"}),

        # ── Objetivos específicos ─────────────────────────────────────────────
        html.Div("Objetivos Específicos", style={
            "fontSize": "0.7rem", "fontWeight": "600", "letterSpacing": "0.08em",
            "textTransform": "uppercase", "color": "#5A7290", "marginBottom": "16px",
        }),
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Div([
                        html.I(className=f"bi {icon}",
                               style={"color": color, "fontSize": "1.4rem", "flexShrink": "0"}),
                    ], style={"display": "flex", "alignItems": "center", "marginRight": "14px"}),
                    html.P(desc, style={"fontSize": "0.87rem", "color": "#3D5266",
                                        "lineHeight": "1.55", "marginBottom": "0"}),
                ], className="stat-card", style={
                    "padding": "18px 20px", "display": "flex",
                    "alignItems": "center", "height": "100%",
                }),
            md=6, className="mb-3")
            for i, (icon, color, desc) in enumerate(esp)
        ], className="g-3"),
    ])


def section_univariado():
    opciones = [
        {"label": "Distribución por Grupo OPS",        "value": "ops"},
        {"label": "Distribución por Sexo",             "value": "sexo"},
        {"label": "Distribución de Edad",              "value": "edad"},
        {"label": "Seguridad Social",                  "value": "seg"},
        {"label": "Nivel Educativo",                   "value": "edu"},
        {"label": "Defunciones por Año",               "value": "anual"},
    ]
    return html.Div([
        html.H2("Análisis Univariado", className="section-title"),
        html.P("Distribución individual de las variables más relevantes del dataset.",
               className="section-subtitle"),

        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div("Variable a explorar", style={
                        "fontSize": "0.7rem", "fontWeight": "600", "letterSpacing": "0.08em",
                        "textTransform": "uppercase", "color": "#5A7290", "marginBottom": "8px",
                    }),
                    dcc.Dropdown(
                        id="dd-univariado",
                        options=opciones,
                        value="ops",
                        clearable=False,
                        persistence=True,
                        persistence_type="session",
                        style={"fontFamily": "DM Sans, sans-serif", "fontSize": "0.88rem"},
                    ),
                ], md=5),
            ], className="mb-0"),
        ], className="stat-card", style={"padding": "20px 24px", "marginBottom": "24px"}),

        dbc.Card(dbc.CardBody(
            dcc.Graph(id="graph-univariado", config={"displayModeBar": False})
        ), className="stat-card mb-3"),

        html.Div(id="desc-univariado"),
    ])

def section_bivariado():
    opciones = [
        {"label": "Grupo OPS × Sexo",              "value": "sexo"},
        {"label": "Grupo OPS × Edad (boxplot)",    "value": "edad"},
        {"label": "Evolución anual por Grupo OPS", "value": "anual"},
        {"label": "Grupo OPS × Seguridad Social",  "value": "seg"},
        {"label": "Grupo Etario × Grupo OPS (heatmap)", "value": "heatmap"},
    ]
    return html.Div([
        html.H2("Análisis Bivariado", className="section-title"),
        html.P("Relaciones entre la variable objetivo (Grupo OPS) y las demás variables del dataset.",
               className="section-subtitle"),

        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div("Análisis a visualizar", style={
                        "fontSize": "0.7rem", "fontWeight": "600", "letterSpacing": "0.08em",
                        "textTransform": "uppercase", "color": "#5A7290", "marginBottom": "8px",
                    }),
                    dcc.Dropdown(
                        id="dd-bivariado",
                        options=opciones,
                        value="sexo",
                        clearable=False,
                        persistence=True,
                        persistence_type="session",
                        style={"fontFamily": "DM Sans, sans-serif", "fontSize": "0.88rem"},
                    ),
                ], md=6),
            ], className="mb-0"),
        ], className="stat-card", style={"padding": "20px 24px", "marginBottom": "24px"}),

        dbc.Card(dbc.CardBody(
            dcc.Graph(id="graph-bivariado", config={"displayModeBar": False})
        ), className="stat-card mb-3"),

        html.Div(id="desc-bivariado"),
    ])

def section_modelo():

    # ── helpers ──────────────────────────────────────────────────────────────
    METR_COLS = ["Modelo", "Accuracy", "F1 Weighted ★", "F1 Macro", "Recall Macro", "Precision W"]

    def mk_select(id_, opts, placeholder):
        return dcc.Dropdown(
            id=id_, options=[{"label": o, "value": o} for o in opts],
            placeholder=placeholder, clearable=False, className="mb-3"
        )

    # ── Tabla comparativa ────────────────────────────────────────────────────
    best_f1 = metrics_df["F1 Weighted ★"].max()
    metric_rows = []
    for _, row in metrics_df.iterrows():
        is_best = row["F1 Weighted ★"] == best_f1
        metric_rows.append(html.Tr([
            html.Td(html.Strong(row["Modelo"]) if is_best else row["Modelo"]),
            html.Td(f"{row['Accuracy']}%"),
            html.Td(
                html.Strong(f"{row['F1 Weighted ★']}%", style={"color": "#1a6b3c"}),
                style={"background": "#d4edda"}
            ),
            html.Td(f"{row['F1 Macro']}%"),
            html.Td(f"{row['Recall Macro']}%"),
            html.Td(f"{row['Precision W']}%"),
        ]))

    metrics_table = dbc.Table([
        html.Thead(html.Tr([html.Th(c) for c in METR_COLS])),
        html.Tbody(metric_rows),
    ], striped=True, hover=True, size="sm")

    # ── Gráfico barras comparativo ────────────────────────────────────────────
    metr_bar_cols = ["Accuracy", "F1 Weighted ★", "F1 Macro", "Recall Macro"]
    fig_bars = go.Figure()
    for i, (_, row) in enumerate(metrics_df.iterrows()):
        fig_bars.add_trace(go.Bar(
            name=row["Modelo"],
            x=metr_bar_cols,
            y=[row[c] for c in metr_bar_cols],
            marker_color=["#1B4F72", "#2E86C1"][i],
            text=[f"{row[c]:.1f}%" for c in metr_bar_cols],
            textposition="outside",
        ))
    fig_bars.update_layout(
        **LAYOUT_BASE,
        barmode="group", height=340,
        title="Comparación de métricas por modelo",
        yaxis=dict(title="%", range=[0, 60]),
        legend=dict(orientation="h", y=-0.25, font=dict(color="#2C3E50")),
        annotations=[dict(
            text="★ Métrica principal — dataset desbalanceado (ratio 55:1)",
            xref="paper", yref="paper", x=0, y=1.08,
            showarrow=False, font=dict(size=10, color="#5D6D7E"),
        )],
    )

    # ── KPI cards comparativas ────────────────────────────────────────────────
    rf_row  = metrics_df[metrics_df["Modelo"] == "Random Forest"].iloc[0]
    dt_row  = metrics_df[metrics_df["Modelo"] == "Árbol de Decisión"].iloc[0]

    def kpi_pair(label, col, is_main=False):
        rf_val = rf_row[col]
        dt_val = dt_row[col]
        winner = "RF" if rf_val >= dt_val else "DT"
        border = "3px solid #2E86C1" if is_main else "1px solid #E4ECF4"
        bg     = "#F0F7FF" if is_main else "#FFFFFF"
        return dbc.Col(html.Div([
            html.Div(label, style={
                "fontSize": "0.67rem", "fontWeight": "600", "letterSpacing": "0.08em",
                "textTransform": "uppercase", "color": "#5A7290" if not is_main else "#1A5C8A",
                "marginBottom": "14px",
            }),
            dbc.Row([
                dbc.Col([
                    html.Div(f"{rf_val:.1f}%", style={
                        "fontFamily": "DM Serif Display, serif", "fontSize": "1.9rem",
                        "color": "#1B4F72" if winner == "RF" else "#7F8C8D", "lineHeight": "1",
                    }),
                    html.Div("Random Forest", style={"fontSize": "0.72rem", "color": "#7A93AD", "marginTop": "4px"}),
                ], width=6),
                dbc.Col([
                    html.Div(f"{dt_val:.1f}%", style={
                        "fontFamily": "DM Serif Display, serif", "fontSize": "1.9rem",
                        "color": "#1B4F72" if winner == "DT" else "#7F8C8D", "lineHeight": "1",
                    }),
                    html.Div("Árbol de Decisión", style={"fontSize": "0.72rem", "color": "#7A93AD", "marginTop": "4px"}),
                ], width=6),
            ]),
            *([] if not is_main else [
                html.Div("★ Métrica principal", style={
                    "fontSize": "0.65rem", "color": "#2E86C1", "marginTop": "10px", "fontWeight": "600",
                })
            ]),
        ], style={
            "background": bg, "borderRadius": "12px", "padding": "18px 20px",
            "border": border, "height": "100%",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.05)" if is_main else "none",
        }), className="mb-3")

    kpi_grid = dbc.Row([
        kpi_pair("F1-Score Weighted",  "F1 Weighted ★", is_main=True),
        kpi_pair("Accuracy",           "Accuracy"),
        kpi_pair("F1 Macro",           "F1 Macro"),
        kpi_pair("Recall Macro",       "Recall Macro"),
    ], className="g-3 mb-4")

    # ── Nota F1 weighted ─────────────────────────────────────────────────────
    nota_f1 = html.Div([
        html.Div([
            html.I(className="bi bi-info-circle-fill me-2",
                   style={"color": "#2E86C1", "fontSize": "0.9rem"}),
            html.Span("¿Por qué F1-Score Weighted?", style={
                "fontWeight": "600", "fontSize": "0.85rem", "color": "#1A2B3C",
            }),
        ], style={"marginBottom": "8px", "display": "flex", "alignItems": "center"}),
        html.P(
            "Con un ratio de desbalance 55:1, el Accuracy puede reportar valores altos "
            "simplemente prediciendo siempre la clase mayoritaria. El F1-Score Weighted "
            "promedia el F1 de cada clase ponderando por su soporte real, penalizando "
            "los errores en clases minoritarias sin ignorar las mayoritarias.",
            style={"fontSize": "0.83rem", "color": "#3D5266", "lineHeight": "1.6", "marginBottom": "0"},
        ),
    ], style={
        "background": "#F0F7FF", "borderRadius": "10px", "padding": "16px 20px",
        "border": "1px solid #C8DEF5", "borderLeft": "4px solid #2E86C1",
        "marginBottom": "24px",
    })

    # ── TAB 1: Métricas ───────────────────────────────────────────────────────
    tab_metricas = html.Div([
        kpi_grid,
        nota_f1,

        # Barras + Feature importance
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_bars, config={"displayModeBar": False})
            ), className="stat-card"), md=7, className="mb-4"),
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_feat_imp(), config={"displayModeBar": False})
            ), className="stat-card"), md=5, className="mb-4"),
        ]),

        # Matriz de confusión
        dbc.Card(dbc.CardBody([
            html.Label("Matriz de confusión — normalizada por fila (% por clase real):",
                       className="text-muted small mb-2 d-block"),
            dcc.Dropdown(
                id="dd-model-cm",
                options=[
                    {"label": "Random Forest",     "value": "Random Forest"},
                    {"label": "Árbol de Decisión", "value": "Árbol de Decisión"},
                ],
                value="Random Forest", clearable=False, className="mb-3",
                style={"maxWidth": "280px"},
            ),
            dcc.Graph(id="graph-cm", config={"displayModeBar": False}),
            html.Small(
                "Normalización por fila compensa el desbalance — cada fila suma 100%.",
                className="text-muted mt-1 d-block"
            ),
        ]), className="stat-card mb-4"),

        # ── Conclusión del modelado ──────────────────────────────────────────
        html.Div([
            html.Div([
                html.Div([
                    html.I(className="bi bi-patch-check-fill me-2",
                           style={"fontSize": "1.1rem", "color": "#A8DFCA"}),
                    html.Span("Modelo seleccionado", style={
                        "fontSize": "0.68rem", "fontWeight": "600", "letterSpacing": "0.1em",
                        "textTransform": "uppercase", "color": "#A8DFCA",
                    }),
                ], style={"marginBottom": "10px", "display": "flex", "alignItems": "center"}),
                html.P("Random Forest", style={
                    "fontFamily": "DM Serif Display, serif", "fontSize": "1.6rem",
                    "color": "#FFFFFF", "marginBottom": "4px",
                }),
                html.P("class_weight = 'balanced_subsample'", style={
                    "fontFamily": "IBM Plex Mono, monospace", "fontSize": "0.78rem",
                    "color": "#A8DFCA", "marginBottom": "20px",
                }),
            ]),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div(f"{rf_row['F1 Weighted ★']:.2f}%",
                             style={"fontFamily": "DM Serif Display, serif", "fontSize": "2rem",
                                    "color": "#FFFFFF", "lineHeight": "1"}),
                    html.Div("F1 Weighted ★", style={"fontSize": "0.7rem", "color": "#A8DFCA",
                                                      "textTransform": "uppercase", "letterSpacing": "0.06em"}),
                ]), md=3),
                dbc.Col(html.Div([
                    html.Div(f"{rf_row['F1 Macro']:.2f}%",
                             style={"fontFamily": "DM Serif Display, serif", "fontSize": "2rem",
                                    "color": "#FFFFFF", "lineHeight": "1"}),
                    html.Div("F1 Macro", style={"fontSize": "0.7rem", "color": "#A8DFCA",
                                                 "textTransform": "uppercase", "letterSpacing": "0.06em"}),
                ]), md=3),
                dbc.Col(html.Div([
                    html.Div(f"{rf_row['Recall Macro']:.2f}%",
                             style={"fontFamily": "DM Serif Display, serif", "fontSize": "2rem",
                                    "color": "#FFFFFF", "lineHeight": "1"}),
                    html.Div("Recall Macro", style={"fontSize": "0.7rem", "color": "#A8DFCA",
                                                     "textTransform": "uppercase", "letterSpacing": "0.06em"}),
                ]), md=3),
                dbc.Col(html.P(
                    "Random Forest superó al Árbol de Decisión en todas las métricas relevantes. "
                    "El uso de 'balanced_subsample' — que ajusta pesos clase a clase en cada árbol "
                    "del ensamble — resultó más robusto que 'balanced' en un árbol único frente "
                    "a clases con soporte muy bajo.",
                    style={"fontSize": "0.82rem", "color": "#D4E6F5", "lineHeight": "1.65", "marginBottom": "0"},
                ), md=3),
            ], className="g-3"),
        ], style={
            "background": "linear-gradient(135deg, #0F2E4A 0%, #1A4A7A 60%, #154360 100%)",
            "borderRadius": "14px", "padding": "28px 32px",
            "boxShadow": "0 4px 20px rgba(26,74,122,0.25)",
        }),
    ], className="pt-3")

    # ── TAB 2: Predicción ────────────────────────────────────────────────────
    def lbl(texto):
        return html.Div(texto, style={
            "fontSize": "0.7rem", "fontWeight": "600", "letterSpacing": "0.06em",
            "textTransform": "uppercase", "color": "#5A7290", "marginBottom": "6px",
        })

    tab_prediccion = html.Div([
        # Banner contextual
        html.Div([
            html.Div([
                html.I(className="bi bi-check-circle-fill me-2",
                       style={"color": "#2A9D8F", "fontSize": "0.95rem"}),
                html.Span("Modelo en producción: ", style={"fontWeight": "600", "color": "#1A2B3C"}),
                html.Span("Random Forest", style={"fontWeight": "700", "color": "#1B4F72"}),
                html.Span(" — F1 Weighted 37.52% · Recall Macro 46.30%",
                          style={"color": "#5A7290", "fontSize": "0.85rem"}),
            ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "4px"}),
        ], style={
            "background": "#F0F7FF", "border": "1px solid #C8DEF5",
            "borderLeft": "4px solid #2E86C1", "borderRadius": "10px",
            "padding": "14px 18px", "marginBottom": "20px",
        }),

        html.Div([
            # ── Selector de modelo ──────────────────────────────────────────
            html.Div([
                lbl("Modelo a utilizar"),
                dcc.RadioItems(
                    id="pred-modelo",
                    options=[
                        {"label": "  Random Forest",     "value": "rf"},
                        {"label": "  Árbol de Decisión", "value": "dt"},
                    ],
                    value="rf", inline=True,
                    inputStyle={"marginRight": "6px"},
                    labelStyle={
                        "marginRight": "24px", "fontSize": "0.88rem",
                        "color": "#1A2B3C", "cursor": "pointer",
                    },
                ),
            ], style={"marginBottom": "24px"}),

            html.Hr(style={"borderColor": "#E4ECF4", "margin": "0 0 24px 0"}),

            # ── Inputs ─────────────────────────────────────────────────────
            dbc.Row([
                dbc.Col([
                    lbl("Sexo"),
                    mk_select("pred-sexo", ["Masculino","Femenino","Indeterminado"], "Seleccionar…"),
                    lbl("Estado Civil"),
                    mk_select("pred-estcivil", ["Soltero/a","Casado/a","Viudo/a","Unión libre","Separado/a","Sin info"], "Seleccionar…"),
                    lbl("Seguridad Social"),
                    mk_select("pred-segsocial", ["Contributivo","Subsidiado","Excepción","Particular","Vinculado","Sin info"], "Seleccionar…"),
                ], md=4),

                dbc.Col([
                    lbl("Nivel Educativo"),
                    mk_select("pred-edu", ["Básica","Media","Técnico/Tecnológico","Superior","Sin info"], "Seleccionar…"),
                    lbl("Edad (años)"),
                    dcc.Slider(id="pred-edad", min=0, max=110, step=1, value=68,
                               marks={0:"0", 20:"20", 40:"40", 60:"60", 80:"80", 110:"110"},
                               tooltip={"placement":"bottom","always_visible":True},
                               className="mb-4"),
                    lbl("Año de defunción"),
                    dcc.Slider(id="pred-ano", min=2012, max=2021, step=1, value=2019,
                               marks={y: str(y) for y in range(2012, 2022, 2)},
                               tooltip={"placement":"bottom","always_visible":True},
                               className="mb-4"),
                ], md=4),

                dbc.Col([
                    lbl("Mes"),
                    dcc.Slider(id="pred-mes", min=1, max=12, step=1, value=6,
                               marks={1:"Ene",3:"Mar",6:"Jun",9:"Sep",12:"Dic"},
                               tooltip={"placement":"bottom","always_visible":True},
                               className="mb-4"),
                    html.Div(style={"height": "16px"}),
                    dbc.Button(
                        [html.I(className="bi bi-cpu me-2"), "Ejecutar predicción"],
                        id="btn-predecir",
                        style={
                            "width": "100%", "background": "#1B4F72", "border": "none",
                            "borderRadius": "8px", "padding": "12px",
                            "fontFamily": "DM Sans, sans-serif", "fontWeight": "600",
                            "fontSize": "0.9rem", "letterSpacing": "0.02em",
                            "transition": "background 0.18s ease",
                        },
                    ),
                ], md=4),
            ]),

            html.Div(id="pred-output", style={"marginTop": "28px"}),
        ], className="stat-card", style={"padding": "28px 32px"}),
    ], className="pt-3")

    # ── Layout con Tabs ──────────────────────────────────────────────────────
    return html.Div([
        html.H2("Modelado Predictivo", className="section-title"),
        html.P("Clasificación de NOM_667_OPS_GRUPO con dos modelos de ensamble y árbol simple.",
               className="section-subtitle"),

        # ── Card institucional de contexto ───────────────────────────────────
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Span("Pipeline de modelado", style={
                        "fontSize": "0.68rem", "fontWeight": "600", "letterSpacing": "0.1em",
                        "textTransform": "uppercase", "color": "#5A7290", "display": "block",
                        "marginBottom": "10px",
                    }),
                    html.P(
                        "Se implementaron dos clasificadores supervisados para predecir el grupo OPS "
                        "de una defunción. Dado el severo desbalance (ratio 55:1), ambos modelos "
                        "utilizan class_weight para compensar clases minoritarias.",
                        style={"fontSize": "0.87rem", "color": "#3D5266", "lineHeight": "1.65",
                               "marginBottom": "12px"},
                    ),
                    dbc.Row([
                        dbc.Col(html.Div([
                            html.Code("Random Forest", style={"fontSize": "0.82rem", "color": "#1B4F72",
                                                               "background": "#EBF5FB", "padding": "2px 8px",
                                                               "borderRadius": "4px"}),
                            html.Span(" → class_weight='balanced_subsample'",
                                      style={"fontSize": "0.8rem", "color": "#5A7290", "marginLeft": "6px"}),
                        ]), md=6),
                        dbc.Col(html.Div([
                            html.Code("Árbol de Decisión", style={"fontSize": "0.82rem", "color": "#1B4F72",
                                                                    "background": "#EBF5FB", "padding": "2px 8px",
                                                                    "borderRadius": "4px"}),
                            html.Span(" → class_weight='balanced'",
                                      style={"fontSize": "0.8rem", "color": "#5A7290", "marginLeft": "6px"}),
                        ]), md=6),
                    ]),
                ], md=8),
                dbc.Col([
                    html.Div([
                        html.Div("Split 80/20 estratificado", style={"fontSize": "0.8rem", "color": "#5A7290", "marginBottom": "6px"}),
                        html.Div([
                            html.Span("Train ", style={"fontSize": "0.72rem", "color": "#7A93AD", "textTransform": "uppercase"}),
                            html.Span("116,301", style={"fontFamily": "DM Serif Display, serif",
                                                         "fontSize": "1.4rem", "color": "#1A2B3C"}),
                        ], style={"marginBottom": "4px"}),
                        html.Div([
                            html.Span("Test  ", style={"fontSize": "0.72rem", "color": "#7A93AD", "textTransform": "uppercase"}),
                            html.Span("29,076", style={"fontFamily": "DM Serif Display, serif",
                                                        "fontSize": "1.4rem", "color": "#1A2B3C"}),
                        ]),
                    ])
                ], md=4),
            ]),
        ], className="stat-card", style={"padding": "24px 28px", "marginBottom": "28px"}),

        dbc.Tabs([
            dbc.Tab(tab_metricas,   label="Métricas de modelos",
                    tab_id="tab-metricas",   className="border border-top-0 p-3",
                    label_style={"fontFamily": "DM Sans, sans-serif", "fontWeight": "500",
                                 "fontSize": "0.88rem", "color": "#5A7290", "letterSpacing": "0.02em"}),
            dbc.Tab(tab_prediccion, label="Predicción interactiva",
                    tab_id="tab-prediccion", className="border border-top-0 p-3",
                    label_style={"fontFamily": "DM Sans, sans-serif", "fontWeight": "500",
                                 "fontSize": "0.88rem", "color": "#5A7290", "letterSpacing": "0.02em"}),
        ], id="tabs-modelo", active_tab="tab-metricas"),
    ])
def section_conclusiones():
    hallazgos = [
        ("bi-heart-pulse",    "#C0392B", "Causas circulatorias dominantes",
         "Las enfermedades del sistema circulatorio representan la causa de muerte más frecuente "
         "(~28%), seguidas de neoplasias (~22%) y causas externas (~18%). Estas tres categorías "
         "concentran más del 65% de la mortalidad total en el período analizado."),
        ("bi-gender-ambiguous","#2E86C1", "Desigualdad por sexo",
         "Los hombres presentan mayor mortalidad por causas externas (violencia, accidentes), "
         "mientras que en mujeres predominan las enfermedades circulatorias y neoplasias. "
         "Esta brecha se mantiene estable durante toda la década."),
        ("bi-person-lines-fill","#2A9D8F", "La edad como factor determinante",
         "La edad explica el 60.7% de la importancia del modelo predictivo. Las causas perinatales "
         "y congénitas se concentran en menores de 5 años; las crónicas y degenerativas en mayores "
         "de 60. Las causas externas impactan desproporcionadamente a la población de 15–44 años."),
        ("bi-graph-up-arrow",  "#E9A128", "Tendencias temporales 2012–2021",
         "La mortalidad total presenta fluctuaciones anuales moderadas. Se detecta un incremento "
         "notable en 2020–2021 asociado a causas respiratorias, coherente con el contexto de la "
         "pandemia de COVID-19, que alteró la distribución habitual de causas de muerte."),
        ("bi-shield-check",    "#546E8A", "Desigualdades socioeconómicas",
         "El régimen subsidiado y el bajo nivel educativo están sobre-representados en causas "
         "externas y mal definidas. Las personas sin seguridad social presentan un perfil de "
         "mortalidad más temprana, reflejando inequidades estructurales en el acceso a salud."),
        ("bi-cpu-fill",        "#1A4A7A", "Alcance y límites del modelado",
         "Random Forest obtuvo F1 Weighted de 37.52%, superando al Árbol de Decisión (34.51%). "
         "El rendimiento moderado refleja el solapamiento real entre clases: las tres causas "
         "mayoritarias comparten un perfil demográfico similar, limitando la separabilidad."),
    ]
    return html.Div([
        html.H2("Conclusiones del Análisis", className="section-title"),
        html.P(
            "Síntesis de los principales hallazgos del análisis exploratorio y el modelado predictivo "
            "sobre mortalidad en Medellín 2012–2021.",
            className="section-subtitle",
        ),

        # ── Hallazgos ────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(html.Div([
                html.Div([
                    html.I(className=f"bi {icon}",
                           style={"color": color, "fontSize": "1.1rem", "flexShrink": "0"}),
                    html.Span(titulo, style={
                        "fontWeight": "600", "fontSize": "0.87rem", "color": "#1A2B3C",
                    }),
                ], style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "10px"}),
                html.P(texto, style={
                    "fontSize": "0.84rem", "color": "#3D5266", "lineHeight": "1.65", "marginBottom": "0",
                }),
                html.Div(style={
                    "position": "absolute", "top": "0", "left": "0",
                    "width": "4px", "height": "100%", "background": color,
                    "borderRadius": "14px 0 0 14px",
                }),
            ], className="stat-card", style={"padding": "20px 24px", "height": "100%", "position": "relative", "overflow": "hidden"}),
            md=6, className="mb-3")
            for icon, color, titulo, texto in hallazgos
        ], className="g-3 mb-4"),

        # ── Cierre académico ─────────────────────────────────────────────────
        html.Div([
            html.Span("Reflexión final", style={
                "fontSize": "0.68rem", "fontWeight": "600", "letterSpacing": "0.1em",
                "textTransform": "uppercase", "color": "#8FA3BD", "display": "block", "marginBottom": "12px",
            }),
            html.P(
                "Este dashboard demuestra que la mortalidad urbana no es un fenómeno aleatorio: "
                "refleja estructuras demográficas, socioeconómicas y temporales que pueden ser "
                "identificadas y cuantificadas a través del análisis de datos. Las visualizaciones "
                "presentadas permiten a tomadores de decisiones y equipos de salud pública identificar "
                "poblaciones en riesgo, priorizar intervenciones y monitorear cambios en los patrones "
                "de mortalidad a lo largo del tiempo.",
                style={"fontSize": "0.92rem", "color": "#CBD8E6", "lineHeight": "1.75", "marginBottom": "0"},
            ),
        ], style={
            "background": "linear-gradient(135deg, #0F1E2E 0%, #1A2B3C 100%)",
            "borderRadius": "14px", "padding": "30px 36px",
            "border": "1px solid #1E3050",
        }),
    ])
    

# ---------- Layout principal ----------
app.layout = html.Div([

    dcc.Store(id="active-section", data="intro"),

    sidebar,

    html.Div(
        id="page-content",
        children=[],
        style={
            "marginLeft": "224px",
            "padding": "40px 44px",
            "minHeight": "100vh",
            "backgroundColor": "#F0F4F8",
            "color": "#1E2A38",
            "fontFamily": "'DM Sans', sans-serif",
        }
    ),

], style={
    "backgroundColor": "#F0F4F8",
    "minHeight": "100vh",
})


# =============================================================================
# 5. CALLBACKS
# =============================================================================

SECTIONS = ["intro", "problema", "objetivos", "univariado", "bivariado", "modelo", "conclusiones"]
SECTION_FN = {
    "intro":         section_intro,
    "problema":      section_problema,
    "objetivos":     section_objetivos,
    "univariado":    section_univariado,
    "bivariado":     section_bivariado,
    "modelo":        section_modelo,
    "conclusiones":  section_conclusiones,
}

# Navegación: actualizar sección activa
@app.callback(
    Output("active-section", "data"),
    [Input(f"btn-{s}", "n_clicks") for s in SECTIONS],
    prevent_initial_call=True,
)
def update_active(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "intro"
    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return btn_id.replace("btn-", "")

# Renderizar contenido de sección
@app.callback(
    Output("page-content", "children"),
    Input("active-section", "data"),
)
def render_section(section):
    return SECTION_FN.get(section, section_intro)()

# Actualizar estilos de botones activos
@app.callback(
    [Output(f"btn-{s}", "className") for s in SECTIONS],
    Input("active-section", "data"),
)
def update_nav_styles(active):
    return [
        f"nav-btn btn btn-link {'nav-btn-active' if s == active else ''}"
        for s in SECTIONS
    ]

# Matriz de confusión dinámica
@app.callback(
    Output("graph-cm", "figure"),
    Input("dd-model-cm", "value"),
)
def update_cm(model_name):
    return fig_conf_matrix(model_name)

# Predicción interactiva
@app.callback(
    Output("pred-output", "children"),
    Input("btn-predecir", "n_clicks"),
    State("pred-modelo",   "value"),
    State("pred-sexo",     "value"),
    State("pred-estcivil", "value"),
    State("pred-segsocial","value"),
    State("pred-edu",      "value"),
    State("pred-edad",     "value"),
    State("pred-ano",      "value"),
    State("pred-mes",      "value"),
    prevent_initial_call=True,
)
def predict(n, modelo, sexo, estcivil, segsocial, edu, edad, ano, mes):
    if None in [sexo, estcivil, segsocial, edu]:
        return dbc.Alert("⚠️ Por favor completa todos los campos.", color="warning")

    # Construir fila de predicción
    row = {
        "SEXO":           sexo,
        "EDAD_SIMPLE":    float(edad),
        "EST_CIVIL":      estcivil,
        "SEG_SOCIAL":     segsocial,
        "NIVEL_EDU_GRUPO":edu,
        "ANO":            int(ano),
        "MES":            int(mes),
    }
    df_pred = pd.DataFrame([row])

    # Codificar categorías con los mismos encoders del entrenamiento
    for col in CAT_FEATURES:
        le = encoders[col]
        val = df_pred[col].astype(str).iloc[0]
        if val in le.classes_:
            df_pred[col] = le.transform([val])
        else:
            df_pred[col] = 0

    X_new = df_pred[FEATURES]

    model = rf_model if modelo == "rf" else dt_model

    pred_class = model.predict(X_new)[0]
    pred_label = le_target.inverse_transform([pred_class])[0]

    # Probabilidades
    proba = model.predict_proba(X_new)[0]
    classes = le_target.inverse_transform(np.arange(len(proba)))
    proba_df = pd.DataFrame({"Grupo OPS": classes, "Probabilidad": proba}).sort_values("Probabilidad", ascending=True)

    color = OPS_COLORS.get(pred_label, "#00D4FF")

    fig_proba = px.bar(
        proba_df, x="Probabilidad", y="Grupo OPS", orientation="h",
        color="Probabilidad", color_continuous_scale="Teal",
        title="Probabilidades por clase"
    )
    fig_proba.update_layout(**LAYOUT_BASE, height=320, showlegend=False)
    fig_proba.update_xaxes(tickformat=".0%")

    return dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Predicción del Grupo OPS", className="text-muted mb-1 small"),
            html.H3(pred_label, style={"color": color, "fontWeight":"700"}),
            html.Hr(style={"borderColor": color}),
            html.Small(f"Modelo: {'Random Forest' if modelo=='rf' else 'Árbol de Decisión'}", className="text-muted"),
            html.Br(),
            html.Small(f"Confianza: {proba.max()*100:.1f}%", style={"color": color}),
        ]), className="stat-card pred-result-card h-100"), md=5),
        dbc.Col(dbc.Card(dbc.CardBody([
            dcc.Graph(figure=fig_proba, config={"displayModeBar": False}),
        ]), className="stat-card"), md=7),
    ])
# Descripciones univariado
_UNIV_DESC = {
    "ops":  ("bi-bar-chart-line", "#1B4F72", "Las enfermedades circulatorias concentran ~28% de las defunciones. Tres grupos explican más del 65% de la mortalidad total del período."),
    "sexo": ("bi-gender-ambiguous", "#2E86C1", "Los hombres representan la mayoría de fallecidos. La brecha es especialmente pronunciada en causas externas (violencia y accidentes)."),
    "edad": ("bi-person-fill", "#2A9D8F", "La distribución se concentra en adultos mayores de 60–85 años, con un segundo pico en edades jóvenes asociado a causas externas."),
    "seg":  ("bi-shield-fill", "#546E8A", "Los regímenes contributivo y subsidiado agrupan la mayoría de registros, reflejando la estructura del sistema de salud colombiano."),
    "edu":  ("bi-mortarboard-fill", "#E9A128", "La mayoría de registros corresponden a educación básica o media, coherente con el perfil demográfico de la población fallecida."),
    "anual":("bi-graph-up", "#2E86C1", "La mortalidad se mantiene estable entre 2012–2019. Se detecta un incremento en 2020–2021 asociado al contexto de la pandemia COVID-19."),
}

@app.callback(
    Output("desc-univariado", "children"),
    Input("dd-univariado", "value"),
)
def update_desc_univariado(val):
    icon, color, texto = _UNIV_DESC.get(val, ("bi-info-circle", "#5A7290", ""))
    return html.Div([
        html.I(className=f"bi {icon} me-2", style={"color": color, "fontSize": "0.85rem"}),
        html.Span(texto, style={"fontSize": "0.83rem", "color": "#3D5266", "lineHeight": "1.6"}),
    ], style={
        "background": "#F7FAFD",
        "border": "1px solid #E4ECF4",
        "borderLeft": f"3px solid {color}",
        "borderRadius": "8px",
        "padding": "12px 16px",
        "display": "flex",
        "alignItems": "flex-start",
        "gap": "4px",
    })

# Callback univariado
@app.callback(
    Output("graph-univariado", "figure"),
    Input("dd-univariado", "value"),
)
def update_univariado(val):
    return {
        "ops":   fig_ops_dist,
        "sexo":  fig_sexo,
        "edad":  fig_edad,
        "seg":   fig_seg_social,
        "edu":   fig_edu,
        "anual": fig_anual,
    }[val]()

# Callback bivariado
@app.callback(
    Output("graph-bivariado", "figure"),
    Input("dd-bivariado", "value"),
)
def update_bivariado(val):
    return {
        "sexo":    fig_ops_sexo,
        "edad":    fig_ops_edad,
        "anual":   fig_ops_anual,
        "seg":     fig_ops_seg,
        "heatmap": fig_heatmap_edad_ops,
    }[val]()

# Descripción dinámica bivariado
_BIV_DESC = {
    "sexo":    ("bi-gender-ambiguous", "#2E86C1", "Las causas externas muestran la mayor brecha de género: predominio masculino marcado. En enfermedades circulatorias y neoplasias la distribución es más equilibrada."),
    "edad":    ("bi-person-fill",      "#2A9D8F", "Las causas perinatales y externas concentran fallecidos jóvenes; las enfermedades crónicas (circulatorias, neoplasias) afectan principalmente a mayores de 60 años."),
    "anual":   ("bi-graph-up",         "#546E8A", "Las enfermedades circulatorias lideran consistentemente. Se detecta un incremento en causas respiratorias en 2020–2021, coherente con el impacto de la pandemia."),
    "seg":     ("bi-shield-fill",      "#1B4F72", "El régimen subsidiado está sobre-representado en causas mal definidas y externas, sugiriendo menor acceso a diagnóstico oportuno y atención preventiva."),
    "heatmap": ("bi-grid-3x3-gap-fill","#E9A128", "Los grupos etarios extremos (<5 y >75 años) muestran perfiles de causa de muerte claramente diferenciados del resto de la población."),
}

@app.callback(
    Output("desc-bivariado", "children"),
    Input("dd-bivariado", "value"),
)
def update_desc_bivariado(val):
    icon, color, texto = _BIV_DESC.get(val, ("bi-info-circle", "#5A7290", ""))
    return html.Div([
        html.I(className=f"bi {icon} me-2", style={"color": color, "fontSize": "0.85rem"}),
        html.Span(texto, style={"fontSize": "0.83rem", "color": "#3D5266", "lineHeight": "1.6"}),
    ], style={
        "background": "#F7FAFD",
        "border": "1px solid #E4ECF4",
        "borderLeft": f"3px solid {color}",
        "borderRadius": "8px",
        "padding": "12px 16px",
        "display": "flex",
        "alignItems": "flex-start",
        "gap": "4px",
    })

# =============================================================================
# 6. MAIN
# =============================================================================

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)