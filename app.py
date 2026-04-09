# =============================================================================
# DASHBOARD DE MORTALIDAD — MEDELLÍN 2012–2021
# Autores: Camilo González & Rubén Esguerra
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.impute import SimpleImputer

# =============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

df = pd.read_csv("defunciones_clean.csv")

# Paleta de colores por grupo OPS
OPS_GRUPOS = df["NOM_667_OPS_GRUPO"].unique().tolist()
OPS_COLORS = {
    "Enfermedades del sistema circulatorio": "#E63946",
    "Neoplasias (Tumores)":                  "#457B9D",
    "Todas las demas enfermedades":          "#2A9D8F",
    "Enfermedades Transmisibles":            "#E9C46A",
    "Causas externas":                       "#F4A261",
    "Signos sintomas y afecciones mal definidas": "#8338EC",
    "Ciertas afecciones originadas en el periodo perinatal": "#06D6A0",
}
COLOR_SEQ = list(OPS_COLORS.values())

MESES_NOMBRES = {
    1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
    7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"
}

# ── Orden grupo etario ──────────────────────────────────────────────────────
ETAREO_ORDER = [
    "< 1 año","1-4","5-9","10-14","15-19","20-24","25-29","30-34",
    "35-39","40-44","45-49","50-54","55-59","60-64","65-69",
    "70-74","75-79","80-84","85-89","90-94","95-99","100 y mas"
]

# =============================================================================
# 2. ENTRENAMIENTO DE MODELOS
# =============================================================================

FEATURE_COLS = ["SEXO","EDAD_SIMPLE","MES","EST_CIVIL",
                "SEG_SOCIAL","NIVEL_EDU_GRUPO","COMUNA_RES"]
TARGET_COL   = "NOM_667_OPS_GRUPO"

df_model = df[FEATURE_COLS + [TARGET_COL]].copy()
df_model["EDAD_SIMPLE"] = df_model["EDAD_SIMPLE"].fillna(df_model["EDAD_SIMPLE"].median())

# Codificar categóricas
cat_cols = ["SEXO","EST_CIVIL","SEG_SOCIAL","NIVEL_EDU_GRUPO","COMUNA_RES"]
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df_model[cat_cols] = enc.fit_transform(df_model[cat_cols])

X = df_model[FEATURE_COLS]
y = df_model[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf  = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
dt  = DecisionTreeClassifier(max_depth=5, random_state=42)

rf.fit(X_train, y_train)
dt.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_pred_dt = dt.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
acc_dt = accuracy_score(y_test, y_pred_dt)

labels = sorted(y.unique())

# =============================================================================
# 3. GRÁFICOS — UNIVARIADO
# =============================================================================

def fig_ops_bar():
    vc = df["NOM_667_OPS_GRUPO"].value_counts().reset_index()
    vc.columns = ["Grupo","Defunciones"]
    pct = (vc["Defunciones"] / vc["Defunciones"].sum() * 100).round(1)
    vc["Porcentaje"] = pct
    fig = px.bar(
        vc, x="Defunciones", y="Grupo", orientation="h",
        color="Grupo", color_discrete_map=OPS_COLORS,
        text=pct.map(lambda x: f"{x}%"),
        title="Distribución de causas de muerte (Grupo OPS)",
        template="plotly_dark",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis={"categoryorder":"total ascending"},
                      margin=dict(l=10,r=40,t=50,b=10), height=380)
    return fig

def fig_ops_pie():
    vc = df["NOM_667_OPS_GRUPO"].value_counts().reset_index()
    vc.columns = ["Grupo","Defunciones"]
    fig = px.pie(
        vc, names="Grupo", values="Defunciones",
        color="Grupo", color_discrete_map=OPS_COLORS,
        title="Proporción por grupo OPS",
        hole=0.4, template="plotly_dark",
    )
    fig.update_traces(textposition="inside", textinfo="percent")
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=380,
                      legend=dict(font=dict(size=10)))
    return fig

def fig_sexo():
    vc = df[df["SEXO"]!="Indeterminado"]["SEXO"].value_counts().reset_index()
    vc.columns = ["Sexo","Defunciones"]
    fig = px.bar(vc, x="Sexo", y="Defunciones", color="Sexo",
                 color_discrete_sequence=["#457B9D","#E63946"],
                 text="Defunciones", title="Distribución por sexo",
                 template="plotly_dark")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(l=10,r=10,t=50,b=10), height=340)
    return fig

def fig_edad_hist():
    fig = px.histogram(
        df.dropna(subset=["EDAD_SIMPLE"]), x="EDAD_SIMPLE",
        nbins=30, title="Distribución de edad al fallecimiento",
        template="plotly_dark", color_discrete_sequence=["#2A9D8F"],
    )
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=340,
                      xaxis_title="Edad (años)", yaxis_title="Frecuencia")
    return fig

def fig_etareo():
    vc = df["ETAREO_QUIN"].value_counts().reindex(ETAREO_ORDER).dropna().reset_index()
    vc.columns = ["Grupo etario","Defunciones"]
    fig = px.bar(vc, x="Grupo etario", y="Defunciones",
                 color_discrete_sequence=["#E9C46A"],
                 title="Distribución por grupo etario (quinquenal)",
                 template="plotly_dark")
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=340,
                      xaxis_tickangle=-45)
    return fig

def fig_seg_social():
    vc = df["SEG_SOCIAL"].value_counts().reset_index()
    vc.columns = ["Régimen","Defunciones"]
    fig = px.bar(vc, x="Régimen", y="Defunciones",
                 color="Régimen", text="Defunciones",
                 title="Distribución por régimen de seguridad social",
                 template="plotly_dark",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(l=10,r=10,t=50,b=10), height=340)
    return fig

def fig_edu():
    vc = df["NIVEL_EDU_GRUPO"].value_counts().reset_index()
    vc.columns = ["Nivel educativo","Defunciones"]
    fig = px.bar(vc, x="Nivel educativo", y="Defunciones",
                 color="Nivel educativo", text="Defunciones",
                 title="Distribución por nivel educativo",
                 template="plotly_dark",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(l=10,r=10,t=50,b=10), height=340)
    return fig

def fig_ano():
    vc = df.groupby("ANO").size().reset_index(name="Defunciones")
    fig = px.line(vc, x="ANO", y="Defunciones", markers=True,
                  title="Defunciones totales por año",
                  template="plotly_dark", color_discrete_sequence=["#E63946"])
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=300,
                      xaxis=dict(tickmode="linear", dtick=1))
    return fig

def fig_mes():
    vc = df.groupby("MES").size().reset_index(name="Defunciones")
    vc["MesNombre"] = vc["MES"].map(MESES_NOMBRES)
    fig = px.bar(vc, x="MesNombre", y="Defunciones",
                 color_discrete_sequence=["#457B9D"],
                 title="Defunciones por mes (promedio histórico)",
                 template="plotly_dark")
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=300,
                      xaxis_title="Mes")
    return fig

# =============================================================================
# 4. GRÁFICOS — BIVARIADO
# =============================================================================

def fig_biv_sexo(ano_filter=None):
    data = df[df["SEXO"].isin(["Masculino","Femenino"])]
    if ano_filter:
        data = data[data["ANO"]==ano_filter]
    ct = pd.crosstab(data["NOM_667_OPS_GRUPO"], data["SEXO"], normalize="index") * 100
    ct = ct.reset_index().melt(id_vars="NOM_667_OPS_GRUPO",
                                value_name="Porcentaje", var_name="Sexo")
    fig = px.bar(ct, x="NOM_667_OPS_GRUPO", y="Porcentaje", color="Sexo",
                 barmode="stack", text=ct["Porcentaje"].round(1).map(lambda x: f"{x}%"),
                 color_discrete_sequence=["#457B9D","#E63946"],
                 title="Causa de muerte × Sexo (%)",
                 template="plotly_dark")
    fig.update_traces(textposition="inside")
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=420,
                      xaxis_tickangle=-20, legend_title="Sexo",
                      xaxis_title="")
    return fig

def fig_biv_etareo(ano_filter=None):
    data = df.dropna(subset=["ETAREO_QUIN"])
    if ano_filter:
        data = data[data["ANO"]==ano_filter]
    hm = pd.crosstab(data["NOM_667_OPS_GRUPO"], data["ETAREO_QUIN"])
    hm = hm.reindex(columns=[c for c in ETAREO_ORDER if c in hm.columns])
    hm_norm = hm.div(hm.sum(axis=1), axis=0).round(3) * 100
    fig = px.imshow(
        hm_norm, aspect="auto",
        color_continuous_scale="YlOrRd",
        title="Causa de muerte × Grupo etario (% por fila)",
        template="plotly_dark",
        labels={"color":"% dentro del grupo OPS"},
    )
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=380,
                      xaxis_tickangle=-45)
    return fig

def fig_biv_temporal(ano_filter=None):
    evol = df.groupby(["ANO","NOM_667_OPS_GRUPO"]).size().reset_index(name="Defunciones")
    fig = px.line(evol, x="ANO", y="Defunciones", color="NOM_667_OPS_GRUPO",
                  markers=True, color_discrete_map=OPS_COLORS,
                  title="Evolución temporal de causas de muerte",
                  template="plotly_dark")
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=400,
                      xaxis=dict(tickmode="linear", dtick=1),
                      legend_title="Grupo OPS",
                      legend=dict(font=dict(size=10)))
    return fig

def fig_biv_seg(ano_filter=None):
    data = df[df["SEG_SOCIAL"]!="Sin info"]
    if ano_filter:
        data = data[data["ANO"]==ano_filter]
    ct = pd.crosstab(data["NOM_667_OPS_GRUPO"], data["SEG_SOCIAL"],
                     normalize="index") * 100
    fig = px.imshow(ct.round(1), aspect="auto",
                    color_continuous_scale="Blues",
                    title="Causa de muerte × Régimen de seguridad social (% por fila)",
                    template="plotly_dark",
                    labels={"color":"% dentro del grupo OPS"})
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=380)
    return fig

def fig_biv_edu(ano_filter=None):
    data = df[~df["NIVEL_EDU_GRUPO"].isin(["Sin info"])]
    if ano_filter:
        data = data[data["ANO"]==ano_filter]
    orden_edu = ["Básica","Media","Técnico/Tecnológico","Superior"]
    ct = pd.crosstab(data["NOM_667_OPS_GRUPO"], data["NIVEL_EDU_GRUPO"],
                     normalize="index") * 100
    ct = ct.reindex(columns=[c for c in orden_edu if c in ct.columns])
    ct_m = ct.reset_index().melt(id_vars="NOM_667_OPS_GRUPO",
                                  value_name="Porcentaje", var_name="Nivel")
    fig = px.bar(ct_m, x="NOM_667_OPS_GRUPO", y="Porcentaje", color="Nivel",
                 barmode="group",
                 title="Causa de muerte × Nivel educativo (%)",
                 template="plotly_dark",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=420,
                      xaxis_tickangle=-20, legend_title="Nivel educativo",
                      xaxis_title="")
    return fig

def fig_biv_comuna(ano_filter=None):
    data = df[~df["COMUNA_RES"].isin(["Sin informacion","sin informacion"])]
    if ano_filter:
        data = data[data["ANO"]==ano_filter]
    top10 = data["COMUNA_RES"].value_counts().head(10).index
    data = data[data["COMUNA_RES"].isin(top10)]
    ct = pd.crosstab(data["COMUNA_RES"], data["NOM_667_OPS_GRUPO"],
                     normalize="index") * 100
    fig = px.imshow(ct.round(1), aspect="auto",
                    color_continuous_scale="Teal",
                    title="Top 10 comunas × Causa de muerte (% por fila)",
                    template="plotly_dark",
                    labels={"color":"% dentro de la comuna"})
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=400)
    return fig

# =============================================================================
# 5. GRÁFICOS — MODELADO
# =============================================================================

def fig_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = (cm / cm.sum(axis=1, keepdims=True) * 100).round(1)
    short_labels = [l[:25]+"…" if len(l)>25 else l for l in labels]
    fig = px.imshow(
        cm_pct, x=short_labels, y=short_labels,
        color_continuous_scale="Blues", aspect="auto",
        title=title, template="plotly_dark",
        labels={"color":"% real"},
        text_auto=True,
    )
    fig.update_layout(margin=dict(l=10,r=10,t=60,b=10),
                      height=450,
                      xaxis_title="Predicho",
                      yaxis_title="Real",
                      xaxis_tickangle=-30)
    return fig

def fig_importancias():
    imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
    fig = px.bar(imp, orientation="h",
                 title="Importancia de variables — Random Forest",
                 template="plotly_dark",
                 color=imp.values,
                 color_continuous_scale="Reds")
    fig.update_layout(showlegend=False, coloraxis_showscale=False,
                      margin=dict(l=10,r=10,t=50,b=10), height=340,
                      xaxis_title="Importancia (Gini)", yaxis_title="")
    return fig

def fig_comparacion_modelos():
    rep_rf = classification_report(y_test, y_pred_rf, output_dict=True)
    rep_dt = classification_report(y_test, y_pred_dt, output_dict=True)
    rows = []
    for lbl in labels:
        short = lbl[:30]+"…" if len(lbl)>30 else lbl
        rows.append({
            "Grupo OPS": short,
            "RF Precision": round(rep_rf[lbl]["precision"],3),
            "RF Recall":    round(rep_rf[lbl]["recall"],3),
            "RF F1":        round(rep_rf[lbl]["f1-score"],3),
            "DT Precision": round(rep_dt[lbl]["precision"],3),
            "DT Recall":    round(rep_dt[lbl]["recall"],3),
            "DT F1":        round(rep_dt[lbl]["f1-score"],3),
        })
    return pd.DataFrame(rows)

# =============================================================================
# 6. LAYOUT — COMPONENTES REUTILIZABLES
# =============================================================================

NAVBAR = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col(html.Img(src="https://img.icons8.com/fluency/48/heart-with-pulse.png",
                             height="40px"), width="auto"),
            dbc.Col(dbc.NavbarBrand(
                "Dashboard de Mortalidad — Medellín 2012–2021",
                style={"fontSize":"1.2rem","fontWeight":"700","color":"#fff"}
            )),
        ], align="center"),
    ], fluid=True),
    color="#1a1a2e", dark=True, sticky="top",
    style={"borderBottom":"2px solid #E63946"}
)

def kpi_card(titulo, valor, icono, color):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icono, style={"fontSize":"2rem"}),
                html.Div([
                    html.H3(valor, style={"margin":"0","color":color,"fontWeight":"800"}),
                    html.P(titulo, style={"margin":"0","fontSize":"0.85rem","color":"#aaa"}),
                ], style={"marginLeft":"12px"}),
            ], style={"display":"flex","alignItems":"center"}),
        ])
    ], style={"background":"#16213e","border":f"1px solid {color}","borderRadius":"12px"})

def section_header(num, titulo, subtitulo=""):
    return html.Div([
        html.Div([
            html.Span(str(num), style={
                "background":"#E63946","color":"#fff","borderRadius":"50%",
                "width":"36px","height":"36px","display":"flex",
                "alignItems":"center","justifyContent":"center",
                "fontWeight":"800","fontSize":"1rem","flexShrink":"0"
            }),
            html.Div([
                html.H4(titulo, style={"margin":"0","color":"#fff","fontWeight":"700"}),
                html.Small(subtitulo, style={"color":"#aaa"}) if subtitulo else None,
            ], style={"marginLeft":"12px"}),
        ], style={"display":"flex","alignItems":"center","marginBottom":"20px"}),
        html.Hr(style={"borderColor":"#333","marginTop":"0"}),
    ])

def ano_dropdown(id_suffix):
    anos = sorted(df["ANO"].unique())
    return dbc.Row([
        dbc.Col([
            html.Label("Filtrar por año:", style={"color":"#aaa","fontSize":"0.85rem"}),
            dcc.Dropdown(
                id=f"ano-filter-{id_suffix}",
                options=[{"label":"Todos los años","value":"todos"}] +
                        [{"label":str(a),"value":a} for a in anos],
                value="todos",
                clearable=False,
                style={"background":"#16213e","color":"#000"},
            )
        ], md=4),
    ], className="mb-3")

# =============================================================================
# 7. LAYOUT COMPLETO
# =============================================================================

df_cmp = fig_comparacion_modelos()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    meta_tags=[{"name":"viewport","content":"width=device-width, initial-scale=1"}],
)
server = app.server
app.title = "Mortalidad Medellín"

app.layout = dbc.Container([
    NAVBAR,
    html.Div(style={"height":"24px"}),

    # ── SECCIÓN 1: INTRODUCCIÓN ─────────────────────────────────────────────
    html.Section([
        section_header(1, "Introducción",
                       "Contexto del problema y descripción del dataset"),
        dbc.Row([
            dbc.Col(kpi_card("Registros totales","145,377","💀","#E63946"), md=3),
            dbc.Col(kpi_card("Período analizado","2012 – 2021","📅","#457B9D"), md=3),
            dbc.Col(kpi_card("Grupos OPS","7 causas","🏥","#2A9D8F"),       md=3),
            dbc.Col(kpi_card("Comunas","22 comunas","🗺️","#E9C46A"),         md=3),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("📌 Contexto del problema", className="card-title",
                        style={"color":"#E63946"}),
                html.P([
                    "El estudio de la mortalidad es un pilar fundamental en la epidemiología y la ",
                    "salud pública. En Medellín, comprender los patrones de defunción permite a ",
                    "las autoridades sanitarias diseñar políticas de prevención focalizadas, ",
                    "asignar recursos hospitalarios eficientemente y anticipar tendencias ",
                    "demográficas críticas."
                ], style={"color":"#ccc","fontSize":"0.92rem"}),
                html.P([
                    "Este análisis examina registros de defunciones del período 2012–2021, ",
                    "explorando la relación entre la causa de muerte (clasificada según el ",
                    "sistema OPS/OMS en 7 grupos) y variables demográficas, socioeconómicas ",
                    "y geográficas de cada fallecido."
                ], style={"color":"#ccc","fontSize":"0.92rem"}),
            ])], style={"background":"#16213e","border":"1px solid #333","borderRadius":"10px"}),
            md=6),
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("📊 Descripción del dataset", className="card-title",
                        style={"color":"#457B9D"}),
                html.Ul([
                    html.Li("Fuente: Secretaría de Salud de Medellín",
                            style={"color":"#ccc","fontSize":"0.9rem"}),
                    html.Li("145,377 registros de defunciones",
                            style={"color":"#ccc","fontSize":"0.9rem"}),
                    html.Li("11 variables: demográficas, socioeconómicas y geográficas",
                            style={"color":"#ccc","fontSize":"0.9rem"}),
                    html.Li("Variable objetivo: NOM_667_OPS_GRUPO (7 categorías)",
                            style={"color":"#ccc","fontSize":"0.9rem"}),
                    html.Li("Dataset limpio: sin valores nulos críticos (39 nulos en edad)",
                            style={"color":"#ccc","fontSize":"0.9rem"}),
                    html.Li("Cobertura geográfica: 22 comunas de Medellín",
                            style={"color":"#ccc","fontSize":"0.9rem"}),
                ], style={"paddingLeft":"18px"}),
            ])], style={"background":"#16213e","border":"1px solid #333","borderRadius":"10px"}),
            md=6),
        ], className="mb-5 g-3"),
    ]),

    # ── SECCIÓN 2: PROBLEMA ─────────────────────────────────────────────────
    html.Section([
        section_header(2, "Problema de Análisis",
                       "¿Qué queremos responder con estos datos?"),
        dbc.Card([dbc.CardBody([
            html.H6("🔍 Pregunta central de investigación",
                    style={"color":"#E9C46A","fontWeight":"700"}),
            html.Blockquote(
                "¿Qué factores demográficos, socioeconómicos y geográficos "
                "determinan el grupo de causa de muerte de una defunción en "
                "Medellín durante el período 2012–2021?",
                style={"borderLeft":"4px solid #E63946","paddingLeft":"16px",
                       "color":"#fff","fontStyle":"italic","fontSize":"1.05rem"}
            ),
            html.Hr(style={"borderColor":"#333"}),
            dbc.Row([
                dbc.Col([
                    html.H6("📋 Sub-preguntas derivadas",
                            style={"color":"#2A9D8F"}),
                    html.Ul([
                        html.Li("¿Cómo varía la distribución de causas de muerte según sexo y edad?",
                                style={"color":"#ccc","fontSize":"0.9rem","marginBottom":"6px"}),
                        html.Li("¿Existen diferencias por régimen de seguridad social o nivel educativo?",
                                style={"color":"#ccc","fontSize":"0.9rem","marginBottom":"6px"}),
                        html.Li("¿Se observan cambios temporales en los patrones de mortalidad?",
                                style={"color":"#ccc","fontSize":"0.9rem","marginBottom":"6px"}),
                        html.Li("¿Qué comunas presentan mayor concentración de causas específicas?",
                                style={"color":"#ccc","fontSize":"0.9rem","marginBottom":"6px"}),
                    ]),
                ], md=6),
                dbc.Col([
                    html.H6("⚠️ Relevancia del problema",
                            style={"color":"#F4A261"}),
                    html.P([
                        "La identificación de patrones de mortalidad permite a los sistemas de salud ",
                        "anticipar demandas hospitalarias, priorizar programas de prevención en ",
                        "poblaciones vulnerables y optimizar la distribución territorial de recursos médicos."
                    ], style={"color":"#ccc","fontSize":"0.9rem"}),
                ], md=6),
            ]),
        ])], style={"background":"#16213e","border":"1px solid #333","borderRadius":"10px",
                    "marginBottom":"40px"}),
    ]),

    # ── SECCIÓN 3: OBJETIVOS ────────────────────────────────────────────────
    html.Section([
        section_header(3, "Objetivos",
                       "General y específicos del análisis"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("🎯 Objetivo General", style={"color":"#E63946","fontWeight":"700"}),
                html.P([
                    "Desarrollar un dashboard analítico e interactivo que permita explorar ",
                    "los patrones de mortalidad en Medellín (2012–2021), identificar factores ",
                    "asociados a las diferentes causas de muerte y predecir el grupo OPS de ",
                    "una defunción mediante modelos de machine learning."
                ], style={"color":"#ccc","fontSize":"0.92rem"}),
            ])], style={"background":"#1a1a2e","border":"2px solid #E63946",
                        "borderRadius":"10px","height":"100%"}), md=5),
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("📌 Objetivos Específicos",
                        style={"color":"#457B9D","fontWeight":"700"}),
                html.Ol([
                    html.Li("Describir la distribución univariada de las principales variables del dataset.",
                            style={"color":"#ccc","fontSize":"0.88rem","marginBottom":"8px"}),
                    html.Li("Identificar relaciones entre causa de muerte y variables demográficas/socioeconómicas mediante análisis bivariado.",
                            style={"color":"#ccc","fontSize":"0.88rem","marginBottom":"8px"}),
                    html.Li("Entrenar y evaluar dos modelos predictivos (Random Forest y Árbol de Decisión) para clasificar el grupo OPS.",
                            style={"color":"#ccc","fontSize":"0.88rem","marginBottom":"8px"}),
                    html.Li("Habilitar una interfaz de predicción interactiva que permita al usuario ingresar datos y obtener una estimación del grupo de causa de muerte.",
                            style={"color":"#ccc","fontSize":"0.88rem","marginBottom":"8px"}),
                ], style={"paddingLeft":"18px"}),
            ])], style={"background":"#1a1a2e","border":"1px solid #457B9D",
                        "borderRadius":"10px","height":"100%"}), md=7),
        ], className="mb-5 g-3"),
    ]),

    # ── SECCIÓN 4.1: UNIVARIADO ─────────────────────────────────────────────
    html.Section([
        section_header(4, "Análisis Exploratorio — Sección 4.1: Univariado",
                       "Distribución individual de las variables más relevantes"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_ops_bar(), config={"displayModeBar":False}), md=8),
            dbc.Col(dcc.Graph(figure=fig_ops_pie(), config={"displayModeBar":False}), md=4),
        ], className="mb-3 g-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_sexo(),    config={"displayModeBar":False}), md=4),
            dbc.Col(dcc.Graph(figure=fig_edad_hist(),config={"displayModeBar":False}),md=4),
            dbc.Col(dcc.Graph(figure=fig_etareo(),  config={"displayModeBar":False}), md=4),
        ], className="mb-3 g-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_seg_social(),config={"displayModeBar":False}),md=4),
            dbc.Col(dcc.Graph(figure=fig_edu(),     config={"displayModeBar":False}), md=4),
            dbc.Col([
                dcc.Graph(figure=fig_ano(), config={"displayModeBar":False}),
                dcc.Graph(figure=fig_mes(), config={"displayModeBar":False}),
            ], md=4),
        ], className="mb-5 g-2"),
    ]),

    # ── SECCIÓN 4.2: BIVARIADO ──────────────────────────────────────────────
    html.Section([
        section_header(4, "Análisis Exploratorio — Sección 4.2: Bivariado",
                       "Relaciones entre la causa de muerte y variables demográficas/socioeconómicas"),
        ano_dropdown("biv"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="biv-sexo",    config={"displayModeBar":False}), md=6),
            dbc.Col(dcc.Graph(id="biv-etareo",  config={"displayModeBar":False}), md=6),
        ], className="mb-3 g-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="biv-temporal",config={"displayModeBar":False}), md=12),
        ], className="mb-3 g-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="biv-seg",     config={"displayModeBar":False}), md=6),
            dbc.Col(dcc.Graph(id="biv-edu",     config={"displayModeBar":False}), md=6),
        ], className="mb-3 g-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="biv-comuna",  config={"displayModeBar":False}), md=12),
        ], className="mb-5 g-2"),
    ]),

    # ── SECCIÓN 5: MODELADO ─────────────────────────────────────────────────
    html.Section([
        section_header(5, "Modelado Predictivo",
                       "Random Forest y Árbol de Decisión — Clasificación del grupo OPS"),

        # KPI Accuracy
        dbc.Row([
            dbc.Col(kpi_card(
                "Accuracy — Random Forest",
                f"{acc_rf*100:.1f}%","🌲","#2A9D8F"), md=3),
            dbc.Col(kpi_card(
                "Accuracy — Árbol de Decisión",
                f"{acc_dt*100:.1f}%","🌿","#E9C46A"), md=3),
            dbc.Col(kpi_card(
                "Registros train",
                f"{len(X_train):,}","📚","#457B9D"), md=3),
            dbc.Col(kpi_card(
                "Registros test",
                f"{len(X_test):,}","🧪","#F4A261"), md=3),
        ], className="mb-4 g-3"),

        # Info de preparación
        dbc.Card([dbc.CardBody([
            html.H6("⚙️ Preparación de datos para modelado",
                    style={"color":"#E9C46A"}),
            dbc.Row([
                dbc.Col([
                    html.P("Features utilizadas:", style={"color":"#aaa","fontSize":"0.85rem","margin":"0"}),
                    html.P(", ".join(FEATURE_COLS),
                           style={"color":"#fff","fontSize":"0.88rem","fontFamily":"monospace"}),
                ], md=6),
                dbc.Col([
                    html.P("Preprocesamiento:", style={"color":"#aaa","fontSize":"0.85rem","margin":"0"}),
                    html.Ul([
                        html.Li("EDAD_SIMPLE: imputación con mediana",
                                style={"color":"#ccc","fontSize":"0.85rem"}),
                        html.Li("Variables categóricas: OrdinalEncoder",
                                style={"color":"#ccc","fontSize":"0.85rem"}),
                        html.Li("División: 80% train / 20% test (stratified, seed=42)",
                                style={"color":"#ccc","fontSize":"0.85rem"}),
                    ]),
                ], md=6),
            ]),
        ])], style={"background":"#16213e","border":"1px solid #333","borderRadius":"10px",
                    "marginBottom":"20px"}),

        # Matrices de confusión
        dbc.Row([
            dbc.Col(dcc.Graph(
                figure=fig_confusion(y_test, y_pred_rf,
                                     "Matriz de Confusión — Random Forest"),
                config={"displayModeBar":False}), md=6),
            dbc.Col(dcc.Graph(
                figure=fig_confusion(y_test, y_pred_dt,
                                     "Matriz de Confusión — Árbol de Decisión"),
                config={"displayModeBar":False}), md=6),
        ], className="mb-3 g-2"),

        # Importancias + tabla comparativa
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_importancias(),
                              config={"displayModeBar":False}), md=5),
            dbc.Col([
                html.H6("📋 Reporte de clasificación comparativo",
                        style={"color":"#fff","marginBottom":"10px"}),
                dash_table.DataTable(
                    data=df_cmp.to_dict("records"),
                    columns=[{"name":c,"id":c} for c in df_cmp.columns],
                    style_table={"overflowX":"auto"},
                    style_cell={"backgroundColor":"#16213e","color":"#ccc",
                                "border":"1px solid #333","fontSize":"12px",
                                "textAlign":"center","padding":"6px"},
                    style_header={"backgroundColor":"#1a1a2e","color":"#E63946",
                                  "fontWeight":"bold","border":"1px solid #444"},
                    style_data_conditional=[
                        {"if":{"filter_query":"{RF F1} > 0.6"},
                         "color":"#2A9D8F","fontWeight":"bold"},
                    ],
                    page_size=10,
                )
            ], md=7),
        ], className="mb-5 g-2"),
    ]),

    # ── SECCIÓN 6: PREDICCIÓN ───────────────────────────────────────────────
    html.Section([
        section_header(6, "Predicción Interactiva",
                       "Ingresa los datos de un individuo para predecir su grupo de causa de muerte"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("📝 Datos del individuo", style={"color":"#E9C46A","fontWeight":"700"}),
                html.Hr(style={"borderColor":"#333"}),

                dbc.Row([
                    dbc.Col([
                        html.Label("Sexo", style={"color":"#aaa","fontSize":"0.85rem"}),
                        dcc.Dropdown(id="pred-sexo",
                            options=[{"label":v,"value":v} for v in ["Masculino","Femenino"]],
                            value="Masculino", clearable=False,
                            style={"marginBottom":"12px"}),
                    ], md=6),
                    dbc.Col([
                        html.Label("Estado civil", style={"color":"#aaa","fontSize":"0.85rem"}),
                        dcc.Dropdown(id="pred-civil",
                            options=[{"label":v,"value":v} for v in
                                     ["Casado/a","Soltero/a","Viudo/a","Unión libre","Separado/a","Sin info"]],
                            value="Casado/a", clearable=False,
                            style={"marginBottom":"12px"}),
                    ], md=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label(f"Edad: ", id="label-edad",
                                   style={"color":"#aaa","fontSize":"0.85rem"}),
                        dcc.Slider(id="pred-edad", min=0, max=110, step=1, value=65,
                                   marks={0:"0",20:"20",40:"40",60:"60",80:"80",100:"100",110:"110"},
                                   tooltip={"placement":"bottom","always_visible":True}),
                    ], md=12),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Mes de fallecimiento", style={"color":"#aaa","fontSize":"0.85rem"}),
                        dcc.Dropdown(id="pred-mes",
                            options=[{"label":MESES_NOMBRES[m],"value":m} for m in range(1,13)],
                            value=6, clearable=False,
                            style={"marginBottom":"12px"}),
                    ], md=6),
                    dbc.Col([
                        html.Label("Régimen de seguridad social",
                                   style={"color":"#aaa","fontSize":"0.85rem"}),
                        dcc.Dropdown(id="pred-seg",
                            options=[{"label":v,"value":v} for v in
                                     ["Contributivo","Subsidiado","Excepción","Vinculado","Particular","Sin info"]],
                            value="Contributivo", clearable=False,
                            style={"marginBottom":"12px"}),
                    ], md=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("Nivel educativo", style={"color":"#aaa","fontSize":"0.85rem"}),
                        dcc.Dropdown(id="pred-edu",
                            options=[{"label":v,"value":v} for v in
                                     ["Básica","Media","Técnico/Tecnológico","Superior","Sin info"]],
                            value="Básica", clearable=False,
                            style={"marginBottom":"12px"}),
                    ], md=6),
                    dbc.Col([
                        html.Label("Comuna de residencia",
                                   style={"color":"#aaa","fontSize":"0.85rem"}),
                        dcc.Dropdown(id="pred-comuna",
                            options=[{"label":v,"value":v}
                                     for v in sorted(df["COMUNA_RES"].unique())],
                            value=sorted(df["COMUNA_RES"].unique())[0],
                            clearable=False,
                            style={"marginBottom":"12px"}),
                    ], md=6),
                ]),
                html.Hr(style={"borderColor":"#333"}),
                html.Label("Modelo a utilizar", style={"color":"#aaa","fontSize":"0.85rem"}),
                dbc.RadioItems(
                    id="pred-modelo",
                    options=[
                        {"label":" Random Forest","value":"rf"},
                        {"label":" Árbol de Decisión","value":"dt"},
                    ],
                    value="rf", inline=True,
                    style={"color":"#fff","marginBottom":"16px"},
                ),
                dbc.Button("🔮 Predecir causa de muerte", id="btn-predecir",
                           color="danger", size="lg", className="w-100"),
            ])],
            style={"background":"#16213e","border":"1px solid #E9C46A",
                   "borderRadius":"12px","height":"100%"}),
            md=5),

            dbc.Col([
                html.Div(id="pred-resultado"),
                html.Div(id="pred-probabilidades"),
            ], md=7),
        ], className="mb-5 g-3"),
    ]),

    html.Footer([
        html.Hr(style={"borderColor":"#333"}),
        html.P("Dashboard de Mortalidad — Medellín 2012–2021 | Camilo González & Rubén Esguerra",
               style={"textAlign":"center","color":"#555","fontSize":"0.8rem","padding":"10px"}),
    ]),

], fluid=True, style={"background":"#0f0f1a","minHeight":"100vh","padding":"0 20px"})

# =============================================================================
# 8. CALLBACKS
# =============================================================================

@app.callback(
    Output("biv-sexo",    "figure"),
    Output("biv-etareo",  "figure"),
    Output("biv-temporal","figure"),
    Output("biv-seg",     "figure"),
    Output("biv-edu",     "figure"),
    Output("biv-comuna",  "figure"),
    Input("ano-filter-biv","value"),
)
def update_bivariados(ano):
    ano_val = None if ano == "todos" else int(ano)
    return (
        fig_biv_sexo(ano_val),
        fig_biv_etareo(ano_val),
        fig_biv_temporal(ano_val),
        fig_biv_seg(ano_val),
        fig_biv_edu(ano_val),
        fig_biv_comuna(ano_val),
    )


@app.callback(
    Output("pred-resultado",      "children"),
    Output("pred-probabilidades", "children"),
    Input("btn-predecir","n_clicks"),
    State("pred-sexo",   "value"),
    State("pred-edad",   "value"),
    State("pred-mes",    "value"),
    State("pred-civil",  "value"),
    State("pred-seg",    "value"),
    State("pred-edu",    "value"),
    State("pred-comuna", "value"),
    State("pred-modelo", "value"),
    prevent_initial_call=True,
)
def predecir(n, sexo, edad, mes, civil, seg, edu, comuna, modelo):
    # Construir fila con encoder
    row = pd.DataFrame([{
        "SEXO": sexo,
        "EDAD_SIMPLE": float(edad),
        "MES": int(mes),
        "EST_CIVIL": civil,
        "SEG_SOCIAL": seg,
        "NIVEL_EDU_GRUPO": edu,
        "COMUNA_RES": comuna,
    }])
    row[cat_cols] = enc.transform(row[cat_cols])

    mdl = rf if modelo == "rf" else dt
    pred_label = mdl.predict(row)[0]
    probs = mdl.predict_proba(row)[0]
    mdl_name = "Random Forest" if modelo == "rf" else "Árbol de Decisión"
    color = OPS_COLORS.get(pred_label, "#E63946")

    result_card = dbc.Card([dbc.CardBody([
        html.H6(f"Resultado — {mdl_name}",
                style={"color":"#aaa","fontWeight":"400","fontSize":"0.85rem"}),
        html.H4("Grupo OPS predicho:", style={"color":"#fff","marginTop":"10px"}),
        html.H3(pred_label,
                style={"color":color,"fontWeight":"800",
                       "borderLeft":f"5px solid {color}","paddingLeft":"12px",
                       "margin":"10px 0 20px 0"}),
        html.Small("⚠️ Esta predicción es orientativa. No reemplaza criterio médico.",
                   style={"color":"#666"}),
    ])], style={"background":"#16213e","border":f"2px solid {color}",
                "borderRadius":"12px","marginBottom":"16px"})

    # Gráfico de probabilidades
    prob_df = pd.DataFrame({
        "Grupo": mdl.classes_,
        "Probabilidad": probs * 100
    }).sort_values("Probabilidad", ascending=True)
    prob_df["Color"] = prob_df["Grupo"].map(OPS_COLORS)

    fig_prob = px.bar(
        prob_df, x="Probabilidad", y="Grupo", orientation="h",
        color="Grupo", color_discrete_map=OPS_COLORS,
        text=prob_df["Probabilidad"].map(lambda x: f"{x:.1f}%"),
        title="Probabilidades por grupo OPS",
        template="plotly_dark",
    )
    fig_prob.update_traces(textposition="outside")
    fig_prob.update_layout(
        showlegend=False,
        margin=dict(l=10, r=60, t=50, b=10),
        height=380,
        xaxis_title="Probabilidad (%)",
        yaxis_title="",
    )
    fig_pred_card = dcc.Graph(figure=fig_prob, config={"displayModeBar": False})

    return result_card, fig_pred_card


# =============================================================================
# 9. RUN
# =============================================================================

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
