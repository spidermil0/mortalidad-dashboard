# 📊 Dashboard de Mortalidad en Medellín (2012–2021)

> Análisis exploratorio y modelado predictivo de causas de muerte en Medellín, Colombia.
> Proyecto académico — Visualización de Datos & Machine Learning aplicado a Salud Pública.

---

## 🔍 Descripción del Proyecto

Este dashboard interactivo permite explorar los patrones de mortalidad en Medellín entre 2012 y 2021, utilizando datos del Sistema de Estadísticas Vitales. A través de análisis univariado, bivariado y modelos de clasificación supervisada, se busca comprender qué factores demográficos y socioeconómicos determinan el grupo de causa de muerte de un individuo.

**Variable objetivo:** `NOM_667_OPS_GRUPO` — Clasificación de causa de muerte según grupos OPS (7 categorías)

---

## 🧱 Estructura del Dashboard

| Sección | Contenido |
|---------|-----------|
| **Introducción** | Contexto del problema, KPIs del dataset, descripción de variables |
| **Problema** | Pregunta central de investigación y sub-preguntas analíticas |
| **Objetivos** | Objetivo general y 5 objetivos específicos |
| **Análisis Univariado** | Distribución de Grupo OPS, Sexo, Edad, Seguridad Social, Educación, Tendencia anual |
| **Análisis Bivariado** | Cruce de Grupo OPS con Sexo, Edad (boxplot), evolución anual, Seguridad Social, heatmap etario |
| **Modelado Predictivo** | Random Forest + Árbol de Decisión, métricas, matriz de confusión, importancia de variables, **predicción interactiva** |

---

## 🤖 Modelos Implementados

| Modelo | Parámetros clave |
|--------|-----------------|
| **Random Forest** | `n_estimators=100`, `max_depth=10`, `random_state=42` |
| **Árbol de Decisión** | `max_depth=8`, `random_state=42` |

- División train/test: **80% / 20%** con `stratify=y`
- Features: `SEXO`, `EDAD_SIMPLE`, `EST_CIVIL`, `SEG_SOCIAL`, `NIVEL_EDU_GRUPO`, `ANO`, `MES`
- Codificación: `LabelEncoder` por variable categórica

---

## 🛠️ Tecnologías Utilizadas

| Herramienta | Rol |
|-------------|-----|
| [Dash](https://dash.plotly.com/) | Framework principal del dashboard |
| [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/) | Layout y componentes UI |
| [Plotly](https://plotly.com/) | Visualizaciones interactivas |
| [scikit-learn](https://scikit-learn.org/) | Modelos de machine learning |
| [pandas](https://pandas.pydata.org/) | Manipulación de datos |
| [NumPy](https://numpy.org/) | Operaciones numéricas |
| [Gunicorn](https://gunicorn.org/) | Servidor WSGI para producción |

---

## 📁 Estructura del Proyecto

```
mortalidad-dashboard/
│
├── app.py                   # Aplicación Dash principal
├── requirements.txt         # Dependencias del proyecto
├── README.md                # Documentación
│
└── defunciones_clean.csv    # Dataset limpio (debe estar en esta carpeta)
```

> ⚠️ El archivo `defunciones_clean.csv` debe ubicarse en la **misma carpeta** que `app.py`.

---

## 🚀 Instrucciones para Ejecutar en Local

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/mortalidad-medellin-dashboard.git
cd mortalidad-medellin-dashboard
```

### 2. Crear entorno virtual

```bash
# Con venv (Python estándar)
python -m venv venv

# Activar en macOS / Linux
source venv/bin/activate

# Activar en Windows
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Colocar el dataset

Asegúrate de que `defunciones_clean.csv` esté en la raíz del proyecto (misma carpeta que `app.py`).

### 5. Ejecutar la aplicación

```bash
python app.py
```

Abre tu navegador en: **http://localhost:8050**

---

## ☁️ Despliegue en Render (Opcional)

El proyecto está preparado para ser desplegado en Render como Web Service.

### Configuración sugerida:

| Campo             | Valor                           |
| ----------------- | ------------------------------- |
| **Environment**   | Python 3                        |
| **Build Command** | pip install -r requirements.txt |
| **Start Command** | gunicorn app:server             |

> ⚠️ Nota: El despliegue en Render no es obligatorio para la ejecución del proyecto.
> La aplicación funciona correctamente en entorno local siguiendo los pasos indicados anteriormente.

> 💡 El dataset (`defunciones_clean.csv`) debe estar incluido en el repositorio.

---

## 📊 Dataset

| Campo | Detalle |
|-------|---------|
| Nombre | `defunciones_clean.csv` |
| Registros | 145,377 |
| Variables | 11 |
| Período | 2012 – 2021 |
| Fuente | Sistema de Estadísticas Vitales – Medellín |

---

## 👥 Autores

**Camilo González & Rubén Esguerra**
Proyecto de Análisis de Datos – Universidad

---

## 📝 Licencia

Este proyecto es de uso académico. Los datos utilizados son de acceso público.
