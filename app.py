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
                 color_discrete_map={"Masculino": "#2E86C1", "Femenino": "#D4A5C9", "Indeterminado": "#AAB7B8"},
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
                 color_discrete_map={"Masculino": "#2E86C1", "Femenino": "#D4A5C9", "Indeterminado": "#AAB7B8"},
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
    font=dict(family="IBM Plex Sans, sans-serif", size=12, color="#2C3E50"),
    title_font=dict(size=14, color="#2C3E50"),
    margin=dict(l=20, r=20, t=50, b=20),
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
            .table-dark { background: transparent !important; }
            .table-dark th {
                background: #1A2B3C !important;
                color: #CBD8E6 !important;
                font-size: 0.78rem;
                font-weight: 600;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                border: none !important;
                padding: 10px 14px !important;
            }
            .table-dark td {
                color: #2C3E50 !important;
                font-size: 0.85rem;
                border-color: #EBF0F5 !important;
                padding: 9px 14px !important;
                vertical-align: middle;
            }
            .table-dark tbody tr { background: #FFFFFF !important; }
            .table-dark tbody tr:nth-child(even) { background: #F7FAFD !important; }
            .table-dark tbody tr:hover { background: #EFF6FF !important; }

            /* ── Scrollbar ────────────────────────────────── */
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: transparent; }
            ::-webkit-scrollbar-thumb { background: #B0C4D8; border-radius: 3px; }
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
        make_nav_btn("bi-info-circle",     "Introducción", "intro",      True),
        make_nav_btn("bi-question-circle", "Problema",     "problema"),
        make_nav_btn("bi-bullseye",        "Objetivos",    "objetivos"),
        make_nav_btn("bi-bar-chart-line",  "Univariado",   "univariado"),
        make_nav_btn("bi-diagram-3",       "Bivariado",    "bivariado"),
        make_nav_btn("bi-cpu",             "Modelado",     "modelo"),
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
        ]),
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
            html.Tr([html.Td("NOM_667_OPS_GRUPO"), html.Td(dbc.Badge("Objetivo",  color="danger",    pill=True)), html.Td("Grupo de causa de muerte — 7 categorías OPS")]),
            html.Tr([html.Td("ANO / MES"),          html.Td(dbc.Badge("Temporal",  color="info",      pill=True)), html.Td("Año (2012–2021) y mes de fallecimiento")]),
            html.Tr([html.Td("SEXO"),               html.Td(dbc.Badge("Demog.",    color="success",   pill=True)), html.Td("Masculino / Femenino / Indeterminado")]),
            html.Tr([html.Td("EDAD_SIMPLE"),        html.Td(dbc.Badge("Numérica",  color="warning",   pill=True)), html.Td("Edad en años al momento del fallecimiento")]),
            html.Tr([html.Td("ETAREO_QUIN"),        html.Td(dbc.Badge("Categ.",    color="secondary", pill=True)), html.Td("Grupo etario quinquenal")]),
            html.Tr([html.Td("EST_CIVIL"),          html.Td(dbc.Badge("Categ.",    color="secondary", pill=True)), html.Td("Estado civil del fallecido")]),
            html.Tr([html.Td("SEG_SOCIAL"),         html.Td(dbc.Badge("Categ.",    color="secondary", pill=True)), html.Td("Régimen de seguridad social")]),
            html.Tr([html.Td("NIVEL_EDU_GRUPO"),    html.Td(dbc.Badge("Categ.",    color="secondary", pill=True)), html.Td("Nivel educativo agrupado")]),
            html.Tr([html.Td("COMUNA_RES"),         html.Td(dbc.Badge("Geog.",     color="primary",   pill=True)), html.Td("Comuna de residencia (22 comunas)")]),
        ])
    ], striped=True, hover=True, size="sm", className="table-dark")

    return html.Div([

        # ── Header ──────────────────────────────────────────────────────────
        html.Div([
            html.H2("Análisis de Mortalidad Urbana", className="section-title"),
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
                html.Span("📋", style={"marginRight": "8px"}),
                html.Span("Descripción del Dataset",
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

        # ── Dos columnas ─────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(html.Div([
                html.Div([
                    html.Span("📌", style={"marginRight": "8px"}),
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
                    html.Span("⚠️", style={"marginRight": "8px"}),
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
                               style={"color": color, "fontSize": "1.3rem", "flexShrink": "0"}),
                        html.Span(str(i+1), style={
                            "fontSize": "0.65rem", "fontWeight": "700", "color": color,
                            "background": f"rgba(100,150,200,0.1)", "borderRadius": "4px",
                            "padding": "1px 5px", "flexShrink": "0",
                        }),
                    ], style={"display": "flex", "flexDirection": "column",
                               "alignItems": "center", "gap": "4px", "marginRight": "14px"}),
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
        ), className="stat-card"),
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
        ), className="stat-card"),
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

    # ── TAB 1: Métricas ───────────────────────────────────────────────────────
    tab_metricas = html.Div([
        # Contexto desbalance
        dbc.Alert([
            html.Strong("⚠️ Desbalance de clases: ratio 55:1 "),
            "(circulatorio 28.4% vs mal definidas 0.5%). ",
            "El Accuracy puede ser engañoso — la métrica principal es el ",
            html.Strong("F1-Score Weighted"), ". Se usó ",
            html.Code("class_weight='balanced_subsample'"),
            " en Random Forest y ",
            html.Code("class_weight='balanced'"),
            " en Árbol de Decisión.",
        ], color="warning", className="mb-3 py-2"),

        # Tabla
        dbc.Card(dbc.CardBody([
            html.P("★ F1 Weighted = métrica principal  |  verde = mejor valor por columna",
                   className="text-muted small mb-2"),
            metrics_table,
        ]), className="mb-4"),

        # Barras + Feature importance
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_bars, config={"displayModeBar": False})
            )), md=7, className="mb-4"),
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_feat_imp(), config={"displayModeBar": False})
            )), md=5, className="mb-4"),
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
        ])),

        # Conclusión del notebook
        html.Hr(className="my-4"),
        dbc.Alert([
            html.H6("📌 Conclusión — Modelo seleccionado: Random Forest", className="mb-2"),
            html.Ul([
                html.Li([html.Strong("F1 Weighted: "), "37.52% vs 34.51% (+3 pp) — métrica principal bajo desbalance."]),
                html.Li([html.Strong("F1 Macro: "),    "38.85% vs 36.50% — mejor detección en clases minoritarias."]),
                html.Li([html.Strong("Recall Macro: "),"46.30% vs 44.60% — mayor cobertura real entre todas las clases."]),
                html.Li([html.Strong("Límite del modelo: "), "las 3 clases mayoritarias (circulatorio, neoplasias, otras) comparten "
                         "perfil etario similar — la edad explica el 60.7% de importancia pero no es "
                         "suficiente para separar esas clases."]),
                html.Li([html.Strong("class_weight='balanced_subsample': "), "ajusta pesos en cada árbol del ensamble, "
                         "más robusto que 'balanced' en un árbol único con clases de soporte muy bajo."]),
            ], className="mb-0"),
        ], color="light", className="border border-primary mt-3"),
    ], className="pt-3")

    # ── TAB 2: Predicción ────────────────────────────────────────────────────
    tab_prediccion = html.Div([
        dbc.Alert([
            html.Strong("Modelo en producción: Random Forest"),
            " — mayor F1 Weighted (37.52%) y Recall Macro (46.30%). "
            "Selecciona Árbol de Decisión para comparar predicciones.",
        ], color="info", className="mb-3 py-2"),

        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Modelo:", className="fw-bold mb-1 small"),
                    dcc.RadioItems(
                        id="pred-modelo",
                        options=[
                            {"label": "  Random Forest",     "value": "rf"},
                            {"label": "  Árbol de Decisión", "value": "dt"},
                        ],
                        value="rf", inline=True, className="mb-4",
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"marginRight": "20px"},
                    ),
                ], md=12),

                dbc.Col([
                    html.Label("Sexo", className="text-muted small"),
                    mk_select("pred-sexo", ["Masculino","Femenino","Indeterminado"], "Seleccionar…"),
                    html.Label("Estado Civil", className="text-muted small"),
                    mk_select("pred-estcivil", ["Soltero/a","Casado/a","Viudo/a","Unión libre","Separado/a","Sin info"], "Seleccionar…"),
                    html.Label("Seguridad Social", className="text-muted small"),
                    mk_select("pred-segsocial", ["Contributivo","Subsidiado","Excepción","Particular","Vinculado","Sin info"], "Seleccionar…"),
                ], md=4),

                dbc.Col([
                    html.Label("Nivel Educativo", className="text-muted small"),
                    mk_select("pred-edu", ["Básica","Media","Técnico/Tecnológico","Superior","Sin info"], "Seleccionar…"),
                    html.Label("Edad (años)", className="text-muted small"),
                    dcc.Slider(id="pred-edad", min=0, max=110, step=1, value=68,
                               marks={0:"0", 20:"20", 40:"40", 60:"60", 80:"80", 110:"110"},
                               tooltip={"placement":"bottom","always_visible":True},
                               className="mb-4"),
                    html.Label("Año de defunción", className="text-muted small"),
                    dcc.Slider(id="pred-ano", min=2012, max=2021, step=1, value=2019,
                               marks={y: str(y) for y in range(2012, 2022, 2)},
                               tooltip={"placement":"bottom","always_visible":True},
                               className="mb-4"),
                ], md=4),

                dbc.Col([
                    html.Label("Mes", className="text-muted small"),
                    dcc.Slider(id="pred-mes", min=1, max=12, step=1, value=6,
                               marks={1:"Ene",3:"Mar",6:"Jun",9:"Sep",12:"Dic"},
                               tooltip={"placement":"bottom","always_visible":True},
                               className="mb-4"),
                    html.Br(),
                    dbc.Button(
                        [html.I(className="bi bi-lightning-charge-fill me-2"), "Predecir"],
                        id="btn-predecir", color="primary", size="lg",
                        className="w-100 mt-2 fw-bold",
                    ),
                ], md=4),
            ]),
            html.Div(id="pred-output", className="mt-4"),
        ])),
    ], className="pt-3")

    # ── Layout con Tabs ──────────────────────────────────────────────────────
    return html.Div([
        html.H2("Modelado Predictivo", className="section-title"),
        html.P("Comparativa entre Random Forest y Árbol de Decisión para clasificar grupos OPS.",
               className="section-subtitle"),

        dbc.Alert([
            html.Strong("Pipeline: "),
            "Features: SEXO, EDAD_SIMPLE, EST_CIVIL, SEG_SOCIAL, NIVEL_EDU_GRUPO, ANO, MES  |  "
            "LabelEncoder  |  80/20 stratified  |  ",
            html.Code("class_weight"), " balanceado en ambos modelos  |  Train: 116,301  |  Test: 29,076",
        ], color="light", className="border mb-4 py-2"),

        dbc.Tabs([
            dbc.Tab(tab_metricas,   label="📊 Métricas de modelos",
                    tab_id="tab-metricas",   className="border border-top-0 p-3"),
            dbc.Tab(tab_prediccion, label="🎯 Predicción interactiva",
                    tab_id="tab-prediccion", className="border border-top-0 p-3"),
        ], id="tabs-modelo", active_tab="tab-metricas"),
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

SECTIONS = ["intro", "problema", "objetivos", "univariado", "bivariado", "modelo"]
SECTION_FN = {
    "intro":       section_intro,
    "problema":    section_problema,
    "objetivos":   section_objetivos,
    "univariado":  section_univariado,
    "bivariado":   section_bivariado,
    "modelo":      section_modelo,
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
#@app.callback(
#    [Output(f"btn-{s}", "className") for s in SECTIONS],
#    Input("active-section", "data"),
#)
#def update_nav_styles(active):
#    return [
#        f"nav-btn btn btn-link {'nav-btn-active' if s == active else ''}"
#        for s in SECTIONS
#    ]

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

# =============================================================================
# 6. MAIN
# =============================================================================

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)