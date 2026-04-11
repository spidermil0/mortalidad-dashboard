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
    "Enfermedades del sistema circulatorio":                "#E63946",
    "Neoplasias (Tumores)":                                 "#457B9D",
    "Todas las demas enfermedades":                         "#2A9D8F",
    "Enfermedades Transmisibles":                           "#E9C46A",
    "Causas externas":                                      "#F4A261",
    "Ciertas afecciones originadas en el periodo perinatal":"#6D6875",
    "Signos sintomas y afecciones mal definidas":           "#A8DADC",
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
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# ---- Decision Tree ----
dt_model = DecisionTreeClassifier(max_depth=8, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# ---- Métricas ----
def get_metrics(y_true, y_pred, name):
    return {
        "Modelo":     name,
        "Accuracy":   round(accuracy_score(y_true, y_pred) * 100, 2),
        "Precision":  round(precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "Recall":     round(recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "F1-Score":   round(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
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
                 color_discrete_sequence=["#457B9D","#E63946","#A8DADC"],
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
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 text="Cantidad", title="Distribución por Régimen de Seguridad Social")
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_BASE, showlegend=False, height=350)
    return fig

def fig_edu():
    orden = ["Básica", "Media", "Técnico/Tecnológico", "Superior", "Sin info"]
    counts = df["NIVEL_EDU_GRUPO"].value_counts().reindex(orden).dropna().reset_index()
    counts.columns = ["Nivel", "Cantidad"]
    fig = px.bar(counts, x="Nivel", y="Cantidad", color="Nivel",
                 color_discrete_sequence=px.colors.qualitative.Pastel,
                 text="Cantidad", title="Distribución por Nivel Educativo")
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_BASE, showlegend=False, height=350)
    return fig

def fig_anual():
    anual = df.groupby("ANO").size().reset_index(name="Defunciones")
    fig = px.line(anual, x="ANO", y="Defunciones", markers=True,
                  title="Total de Defunciones por Año",
                  color_discrete_sequence=["#E63946"])
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
                 color_discrete_sequence=["#457B9D","#E63946","#A8DADC"],
                 title="Distribución de Sexo por Grupo OPS (%)")
    fig.update_xaxes(title="", tickangle=-25)
    fig.update_layout(**LAYOUT_BASE, height=420)
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
    fig.update_layout(**LAYOUT_BASE, height=430)
    return fig

def fig_ops_seg():
    df_s = df[df["SEG_SOCIAL"] != "Sin info"]
    ct = pd.crosstab(df_s["NOM_667_OPS_GRUPO"], df_s["SEG_SOCIAL"], normalize="index") * 100
    ct_melted = ct.reset_index().melt(id_vars="NOM_667_OPS_GRUPO", var_name="Régimen", value_name="Porcentaje")
    fig = px.bar(ct_melted, x="NOM_667_OPS_GRUPO", y="Porcentaje", color="Régimen",
                 barmode="stack",
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 title="Seguridad Social por Grupo OPS (%)")
    fig.update_xaxes(title="", tickangle=-25)
    fig.update_layout(**LAYOUT_BASE, height=420)
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
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, sans-serif", size=12, color="#E0E6ED"),
    title_font=dict(size=14, color="#E0E6ED"),
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)

# =============================================================================
# 4. APP LAYOUT
# =============================================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono&display=swap",
        dbc.icons.BOOTSTRAP,  # 👈 esto ya incluye los iconos
    ],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.title = "Dashboard · Mortalidad Medellín"
server = app.server
# ---------- Sidebar ----------
def make_nav_btn(icon, label, section_id, active=False):
    return dbc.Button(
        [
            html.I(className=f"bi {icon}", style={"marginRight": "8px"}),
            label
        ],
        id=f"btn-{section_id}",
        n_clicks=0,
        style={
            "width": "100%",
            "textAlign": "left",
            "marginBottom": "6px",
            "padding": "10px",
            "backgroundColor": "#1A2D45" if active else "#1A2535",
            "color": "#00D4FF" if active else "#E0E6ED",
            "border": "none",
            "borderRadius": "6px",
            "cursor": "pointer"
        }
    )

sidebar = html.Div([

    html.Div([
        html.Div("📊", style={"fontSize": "2rem"}),
        html.Div([
            html.H6("Mortalidad", style={
                "margin": "0",
                "fontWeight": "bold",
                "color": "#00D4FF"
            }),
            html.Small("Medellín 2012–2021", style={
                "color": "#8892A0",
                "fontSize": "0.7rem"
            }),
        ])
    ], style={
        "display": "flex",
        "alignItems": "center",
        "gap": "10px",
        "marginBottom": "20px"
    }),

    html.Hr(style={"borderColor": "#2A3545"}),

    html.Div([
        make_nav_btn("bi-info-circle", "Introducción", "intro", True),
        make_nav_btn("bi-question-circle", "Problema", "problema"),
        make_nav_btn("bi-bullseye", "Objetivos", "objetivos"),
        make_nav_btn("bi-bar-chart", "Univariado", "univariado"),
        make_nav_btn("bi-graph-up", "Bivariado", "bivariado"),
        make_nav_btn("bi-cpu", "Modelado", "modelo"),
    ]),

    html.Hr(style={"borderColor": "#2A3545"}),

    html.Div([
        html.Small("Dataset: 145,377 registros", style={"color": "#8892A0", "display": "block"}),
        html.Small("Variables: 11 columnas", style={"color": "#8892A0", "display": "block"}),
    ])

], style={
    "position": "fixed",
    "top": "0",
    "left": "0",
    "height": "100vh",
    "width": "220px",
    "backgroundColor": "#0D1521",
    "padding": "20px",
    "borderRight": "1px solid #1E2D40"
})

# ---------- Secciones ----------

def section_intro():
    kpis = [
        ("145,377", "Registros totales",  "bi-database",        "#00D4FF"),
        ("10 años", "Período analizado",  "bi-calendar-range",  "#2A9D8F"),
        ("7 grupos","Categorías OPS",      "bi-diagram-3",       "#E63946"),
        ("11",      "Variables",          "bi-table",           "#E9C46A"),
    ]
    kpi_cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.I(className=f"bi {icon} fs-3 mb-2", style={"color": color}),
                html.H3(val, className="fw-bold mb-0", style={"color": color}),
                html.Small(label, style={"color":"#8892A0"}),
            ], className="text-center py-3")
        ], className="stat-card"), width=6, md=3, className="mb-3")
        for val, label, icon, color in kpis
    ], className="mb-4")

    vars_table = dbc.Table([
        html.Thead(html.Tr([html.Th("Variable"), html.Th("Tipo"), html.Th("Descripción")])),
        html.Tbody([
            html.Tr([html.Td("NOM_667_OPS_GRUPO"), html.Td(dbc.Badge("Objetivo", color="danger")),  html.Td("Grupo de causa de muerte (7 categorías)")]),
            html.Tr([html.Td("ANO / MES"),          html.Td(dbc.Badge("Temporal", color="info")),   html.Td("Año (2012–2021) y mes de fallecimiento")]),
            html.Tr([html.Td("SEXO"),               html.Td(dbc.Badge("Demog.", color="success")),  html.Td("Masculino / Femenino / Indeterminado")]),
            html.Tr([html.Td("EDAD_SIMPLE"),        html.Td(dbc.Badge("Numérica", color="warning")),html.Td("Edad en años al momento del fallecimiento")]),
            html.Tr([html.Td("ETAREO_QUIN"),        html.Td(dbc.Badge("Categ.", color="secondary")),html.Td("Grupo etario quinquenal")]),
            html.Tr([html.Td("EST_CIVIL"),          html.Td(dbc.Badge("Categ.", color="secondary")),html.Td("Estado civil del fallecido")]),
            html.Tr([html.Td("SEG_SOCIAL"),         html.Td(dbc.Badge("Categ.", color="secondary")),html.Td("Régimen de seguridad social")]),
            html.Tr([html.Td("NIVEL_EDU_GRUPO"),    html.Td(dbc.Badge("Categ.", color="secondary")),html.Td("Nivel educativo agrupado")]),
            html.Tr([html.Td("COMUNA_RES"),         html.Td(dbc.Badge("Geog.", color="primary")),   html.Td("Comuna de residencia (22 comunas)")]),
        ])
    ], striped=True, hover=True, size="sm", className="table-dark")

    return html.Div([
        html.H2("Introducción", className="section-title"),
        dbc.Alert([
            html.H5("🏥 Contexto del problema", className="mb-2"),
            html.P(
                "La mortalidad urbana es un indicador clave del estado de salud pública de una ciudad. "
                "Medellín, con más de 2.5 millones de habitantes, registra anualmente decenas de miles de "
                "defunciones clasificadas según la Organización Panamericana de la Salud (OPS). Comprender "
                "qué variables demográficas, socioeconómicas y temporales se asocian con cada tipo de causa "
                "de muerte permite diseñar políticas de salud más focalizadas y eficientes.",
                className="mb-1"
            ),
            html.P(
                "Este análisis explora datos del Sistema de Estadísticas Vitales de Medellín (2012–2021), "
                "aplicando técnicas de análisis exploratorio y modelos de machine learning para predecir "
                "el grupo OPS de una defunción a partir de características del individuo.",
                className="mb-0"
            ),
        ], color="dark", className="border border-secondary mb-4"),
        kpi_cards,
        html.H5("📋 Descripción del Dataset", className="mb-3"),
        vars_table,
    ])


def section_problema():
    return html.Div([
        html.H2("Problema de Análisis", className="section-title"),
        dbc.Card(dbc.CardBody([
            html.H5("❓ Pregunta central", className="text-info mb-3"),
            html.Blockquote(
                "¿Qué factores demográficos, socioeconómicos y temporales determinan "
                "el grupo de causa de muerte (NOM_667_OPS_GRUPO) de una defunción registrada en Medellín?",
                className="blockquote fs-5 border-start border-danger ps-3"
            ),
        ]), className="mb-4 stat-card"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("📌 Sub-preguntas analíticas", className="text-warning mb-3"),
                html.Ul([
                    html.Li("¿Cómo varía la distribución de causas de muerte según sexo y edad?"),
                    html.Li("¿Existen diferencias en las causas de muerte según el régimen de seguridad social o nivel educativo?"),
                    html.Li("¿Ha cambiado la proporción de causas de muerte a lo largo del tiempo (2012–2021)?"),
                    html.Li("¿Es posible predecir el grupo OPS con variables disponibles en el registro de defunción?"),
                ], className="mb-0")
            ]), className="stat-card"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("⚠️ Relevancia del problema", className="text-danger mb-3"),
                html.Ul([
                    html.Li("Las enfermedades circulatorias y neoplasias representan más del 50% de las muertes."),
                    html.Li("La mortalidad por causas externas afecta desproporcionadamente a hombres jóvenes."),
                    html.Li("El régimen de seguridad social puede reflejar desigualdades en acceso a salud."),
                    html.Li("Modelos predictivos pueden apoyar sistemas de alerta temprana en salud pública."),
                ], className="mb-0")
            ]), className="stat-card"), md=6),
        ])
    ])


def section_objetivos():
    esp = [
        ("bi-1-circle", "#00D4FF", "Caracterizar la distribución de causas de muerte según grupos OPS en Medellín."),
        ("bi-2-circle", "#2A9D8F", "Explorar relaciones entre causas de muerte y variables demográficas (sexo, edad, estado civil)."),
        ("bi-3-circle", "#E9C46A", "Analizar el comportamiento temporal de la mortalidad entre 2012 y 2021."),
        ("bi-4-circle", "#F4A261", "Comparar el desempeño predictivo de Random Forest y Árbol de Decisión."),
        ("bi-5-circle", "#E63946", "Construir una herramienta interactiva de predicción del grupo OPS."),
    ]
    return html.Div([
        html.H2("Objetivos", className="section-title"),
        dbc.Card(dbc.CardBody([
            html.H5("🎯 Objetivo General", className="text-info mb-2"),
            html.P(
                "Desarrollar un dashboard analítico-predictivo que permita explorar los patrones de mortalidad "
                "en Medellín e implementar modelos de clasificación capaces de predecir el grupo OPS de una "
                "defunción a partir de variables demográficas y socioeconómicas disponibles en el registro.",
                className="mb-0 fs-5"
            ),
        ]), className="stat-card mb-4"),
        html.H5("📌 Objetivos Específicos", className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div([
                    html.I(className=f"bi {icon} fs-2 me-3", style={"color": color}),
                    html.P(desc, className="mb-0"),
                ], className="d-flex align-items-center")
            ]), className="stat-card mb-3"), md=6)
            for icon, color, desc in esp
        ])
    ])


def section_univariado():
    return html.Div([
        html.H2("Análisis Univariado", className="section-title"),
        html.P("Distribución individual de las variables más relevantes del dataset.", className="text-muted mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    dcc.Graph(figure=fig_ops_dist(), config={"displayModeBar": False}),
                ]), className="stat-card"),
            ], md=12, className="mb-4"),
        ]),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(figure=fig_sexo(), config={"displayModeBar": False}),
            ]), className="stat-card"), md=6, className="mb-4"),
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(figure=fig_edad(), config={"displayModeBar": False}),
            ]), className="stat-card"), md=6, className="mb-4"),
        ]),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(figure=fig_seg_social(), config={"displayModeBar": False}),
            ]), className="stat-card"), md=6, className="mb-4"),
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(figure=fig_edu(), config={"displayModeBar": False}),
            ]), className="stat-card"), md=6, className="mb-4"),
        ]),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(figure=fig_anual(), config={"displayModeBar": False}),
            ]), className="stat-card"), md=12, className="mb-4"),
        ]),
    ])


def section_bivariado():
    return html.Div([
        html.H2("Análisis Bivariado", className="section-title"),
        html.P("Relaciones entre la variable objetivo (Grupo OPS) y las demás variables del dataset.", className="text-muted mb-4"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(figure=fig_ops_sexo(), config={"displayModeBar": False}),
            ]), className="stat-card"), md=6, className="mb-4"),
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(figure=fig_ops_edad(), config={"displayModeBar": False}),
            ]), className="stat-card"), md=6, className="mb-4"),
        ]),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(figure=fig_ops_anual(), config={"displayModeBar": False}),
            ]), className="stat-card"), md=12, className="mb-4"),
        ]),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(figure=fig_ops_seg(), config={"displayModeBar": False}),
            ]), className="stat-card"), md=6, className="mb-4"),
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(figure=fig_heatmap_edad_ops(), config={"displayModeBar": False}),
            ]), className="stat-card"), md=6, className="mb-4"),
        ]),
    ])


def section_modelo():
    # Tabla de métricas
    metric_rows = []
    for _, row in metrics_df.iterrows():
        best_acc = metrics_df["Accuracy"].max()
        metric_rows.append(html.Tr([
            html.Td(row["Modelo"]),
            html.Td(f"{row['Accuracy']}%",  style={"color":"#00D4FF" if row["Accuracy"] == best_acc else "inherit", "fontWeight":"bold" if row["Accuracy"] == best_acc else "normal"}),
            html.Td(f"{row['Precision']}%"),
            html.Td(f"{row['Recall']}%"),
            html.Td(f"{row['F1-Score']}%"),
        ]))

    metrics_table = dbc.Table([
        html.Thead(html.Tr([html.Th(c) for c in ["Modelo","Accuracy","Precision","Recall","F1-Score"]])),
        html.Tbody(metric_rows)
    ], striped=True, hover=True, size="sm", className="table-dark")

    # Opciones de dropdowns para predicción
    def mk_select(id_, opts, placeholder):
        return dcc.Dropdown(
            id=id_, options=[{"label": o, "value": o} for o in opts],
            placeholder=placeholder, clearable=False, className="mb-3 dropdown-dark"
        )

    return html.Div([
        html.H2("Modelado Predictivo", className="section-title"),

        # Sub-sección: pipeline
        dbc.Alert([
            html.H6("⚙️ Pipeline de Modelado", className="mb-2 text-info"),
            html.Ul([
                html.Li("Features: SEXO, EDAD_SIMPLE, EST_CIVIL, SEG_SOCIAL, NIVEL_EDU_GRUPO, ANO, MES"),
                html.Li("Codificación: LabelEncoder para variables categóricas"),
                html.Li("División: 80% Train / 20% Test (stratify=y, random_state=42)"),
                html.Li("Modelos: RandomForestClassifier (100 árboles, max_depth=10) y DecisionTreeClassifier (max_depth=8)"),
            ], className="mb-0"),
        ], color="dark", className="border border-info mb-4"),

        # Métricas
        html.H5("📊 Comparación de Modelos", className="mb-3"),
        dbc.Card(dbc.CardBody(metrics_table), className="stat-card mb-4"),

        # Gráficos de evaluación
        dbc.Row([
            dbc.Col([
                html.Label("Seleccionar modelo:", className="text-muted mb-1"),
                dcc.Dropdown(
                    id="dd-model-cm",
                    options=[
                        {"label":"Random Forest",    "value":"Random Forest"},
                        {"label":"Árbol de Decisión","value":"Árbol de Decisión"},
                    ],
                    value="Random Forest", clearable=False, className="mb-3 dropdown-dark"
                ),
                dbc.Card(dbc.CardBody([
                    dcc.Graph(id="graph-cm", config={"displayModeBar": False}),
                ]), className="stat-card"),
            ], md=7, className="mb-4"),

            dbc.Col([
                dbc.Card(dbc.CardBody([
                    dcc.Graph(figure=fig_feat_imp(), config={"displayModeBar": False}),
                ]), className="stat-card"),
            ], md=5, className="mb-4"),
        ]),

        html.Hr(className="border-secondary my-4"),

        # ---- Predicción interactiva ----
        html.H4("🎯 Predicción Interactiva", className="mb-1 text-info"),
        html.P("Ingresa los valores del individuo y obtén la predicción del Grupo OPS.", className="text-muted mb-4"),

        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Modelo a usar:", className="fw-bold mb-1"),
                    dcc.RadioItems(
                        id="pred-modelo",
                        options=[
                            {"label":"  Random Forest",    "value":"rf"},
                            {"label":"  Árbol de Decisión","value":"dt"},
                        ],
                        value="rf", inline=True, className="mb-3",
                        inputStyle={"marginRight":"6px"},
                        labelStyle={"marginRight":"20px"},
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
                    html.Label(f"Edad: ", className="text-muted small", id="lbl-edad"),
                    dcc.Slider(id="pred-edad", min=0, max=110, step=1, value=68,
                               marks={0:"0", 20:"20", 40:"40", 60:"60", 80:"80", 110:"110"},
                               tooltip={"placement":"bottom","always_visible":True},
                               className="mb-3"),
                    html.Label("Año de defunción", className="text-muted small"),
                    dcc.Slider(id="pred-ano", min=2012, max=2021, step=1, value=2019,
                               marks={y: str(y) for y in range(2012, 2022, 2)},
                               tooltip={"placement":"bottom","always_visible":True},
                               className="mb-3"),
                ], md=4),

                dbc.Col([
                    html.Label("Mes", className="text-muted small"),
                    dcc.Slider(id="pred-mes", min=1, max=12, step=1, value=6,
                               marks={1:"Ene",3:"Mar",6:"Jun",9:"Sep",12:"Dic"},
                               tooltip={"placement":"bottom","always_visible":True},
                               className="mb-3"),
                    html.Br(),
                    dbc.Button(
                        [html.I(className="bi bi-lightning-charge-fill me-2"), "Predecir"],
                        id="btn-predecir", color="danger", size="lg",
                        className="w-100 mt-2 fw-bold"
                    ),
                ], md=4),
            ]),

            html.Div(id="pred-output", className="mt-4"),
        ]), className="stat-card mb-4"),
    ])


# ---------- Layout principal ----------
app.layout = html.Div([

    dcc.Store(id="active-section", data="intro"),

    sidebar,

    html.Div(
        id="page-content",
        children=[],
        style={
            "marginLeft": "220px",
            "padding": "32px",
            "minHeight": "100vh",
            "backgroundColor": "#0A0F1A",
            "color": "#E0E6ED",
            "fontFamily": "Arial, sans-serif"
        }
    ),

], style={
    "backgroundColor": "#0A0F1A",
    "minHeight": "100vh"
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


# =============================================================================
# 6. MAIN
# =============================================================================

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
