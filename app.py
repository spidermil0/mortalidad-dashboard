
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("defunciones_clean.csv")

# -----------------------------
# Basic preprocessing
# -----------------------------
df = df.dropna(subset=["NOM_667_OPS_GRUPO", "EDAD_SIMPLE"])

# Encode target
target_encoder = LabelEncoder()
df["target_encoded"] = target_encoder.fit_transform(df["NOM_667_OPS_GRUPO"])

# Simple feature set for modeling
X = df[["EDAD_SIMPLE"]]
y = df["target_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train models
# -----------------------------
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
tree_acc = accuracy_score(y_test, tree_pred)

# -----------------------------
# Figures for EDA
# -----------------------------

fig_age = px.histogram(df, x="EDAD_SIMPLE", nbins=30, title="Distribución de Edad")

# Crear DataFrame desde value_counts y resetear índice
df_ops_counts = df['NOM_667_OPS_GRUPO'].value_counts().reset_index()
df_ops_counts.columns = ['NOM_667_OPS_GRUPO', 'count']

fig_ops = px.bar(
    df_ops_counts,
    x='NOM_667_OPS_GRUPO',
    y='count',
    title="Distribución de NOM_667_OPS_GRUPO"
)

fig_sex = px.histogram(
    df,
    x="NOM_667_OPS_GRUPO",
    color="SEXO",
    barmode="group",
    title="Grupo OPS vs Sexo"
)

agegrp_counts = pd.crosstab(df["ETAREO_QUIN"], df["NOM_667_OPS_GRUPO"])
fig_agegrp = px.imshow(
    agegrp_counts,
    text_auto=True,
    labels=dict(x="Grupo OPS", y="Grupo Etario", color="Count"),
    title="Grupo OPS vs Grupo Etario"
)

# -----------------------------
# Dash App
# -----------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

tabs = dbc.Tabs(
    [
        dbc.Tab(label="Introducción", tab_id="intro"),
        dbc.Tab(label="Problema", tab_id="problema"),
        dbc.Tab(label="Objetivos", tab_id="objetivos"),
        dbc.Tab(label="Análisis Univariado", tab_id="uni"),
        dbc.Tab(label="Análisis Bivariado", tab_id="bi"),
        dbc.Tab(label="Modelos", tab_id="ml"),
        dbc.Tab(label="Predicción", tab_id="pred"),
    ],
    id="tabs",
    active_tab="intro",
)

app.layout = dbc.Container(
    [
        html.H1("Dashboard de Mortalidad Medellín", className="mt-4 mb-4"),
        tabs,
        html.Div(id="tab-content", className="p-4"),
    ],
    fluid=True,
)

# -----------------------------
# Tab content
# -----------------------------
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
)
def render_tab(tab):

    if tab == "intro":
        return html.Div([
            html.H3("Introducción"),
            html.P(
                "Este dashboard explora patrones de mortalidad en Medellín "
                "utilizando variables demográficas, socioeconómicas y geográficas."
            ),
        ])

    if tab == "problema":
        return html.Div([
            html.H3("Problema"),
            html.P(
                "Identificar patrones en las causas de mortalidad y analizar "
                "cómo se relacionan con variables demográficas."
            ),
        ])

    if tab == "objetivos":
        return html.Div([
            html.H3("Objetivos"),
            html.P("Objetivo general: Analizar patrones de mortalidad."),
            html.P("Objetivos específicos:"),
            html.Ul([
                html.Li("Explorar distribuciones de variables."),
                html.Li("Analizar relaciones entre variables."),
                html.Li("Aplicar modelos predictivos."),
            ])
        ])

    if tab == "uni":
        return html.Div([
            html.H3("Análisis Univariado"),
            dcc.Graph(figure=fig_age),
            dcc.Graph(figure=fig_ops),
        ])

    if tab == "bi":
        return html.Div([
            html.H3("Análisis Bivariado"),
            dcc.Graph(figure=fig_sex),
            dcc.Graph(figure=fig_agegrp),
        ])

    if tab == "ml":
        return html.Div([
            html.H3("Modelos Predictivos"),
            html.P(f"Random Forest Accuracy: {rf_acc:.3f}"),
            html.P(f"Decision Tree Accuracy: {tree_acc:.3f}"),
        ])

    if tab == "pred":
        return html.Div([
            html.H3("Predicción interactiva"),
            html.P("Ingresa una edad para predecir el grupo OPS probable."),
            dcc.Input(id="edad_input", type="number", placeholder="Edad"),
            html.Br(),
            html.Br(),
            html.Div(id="pred_output")
        ])


# -----------------------------
# Prediction callback
# -----------------------------
@app.callback(
    Output("pred_output", "children"),
    Input("edad_input", "value")
)
def predict_group(edad):
    # Validar la entrada
    if edad is None:
        return "Ingresa una edad"
    
    try:
        edad = float(edad)
    except ValueError:
        return "Edad inválida"

    if edad < 0 or edad > 120:
        return "Ingresa una edad válida (0-120)"

    # Predicción
    pred = rf.predict([[edad]])[0]
    label = target_encoder.inverse_transform([pred])[0]

    return f"Grupo OPS predicho: {label}"


if __name__ == "__main__":
    app.run(debug=True)
