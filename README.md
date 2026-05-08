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
| **Objetivos** | Objetivo general y objetivos específicos |
| **Análisis Univariado** | Distribución de Grupo OPS, Sexo, Edad, Seguridad Social, Educación, Tendencia anual |
| **Análisis Bivariado** | Cruce de Grupo OPS con Sexo, Edad (boxplot), evolución anual, Seguridad Social, heatmap etario |
| **Modelado Predictivo** | Random Forest + Árbol de Decisión, métricas, matrices de confusión, importancia de variables |
| **Predicción Interactiva** | Formulario para estimar el grupo OPS usando el modelo Random Forest |

---

## 🤖 Modelos Implementados

| Modelo | Parámetros clave |
|--------|-----------------|
| **Random Forest** | `n_estimators=100`, `max_depth=10`, `class_weight="balanced_subsample"`, `random_state=42` |
| **Árbol de Decisión** | `max_depth=8`, `class_weight="balanced"`, `random_state=42` |

### Configuración del Pipeline

- División train/test: **80% / 20%** usando `stratify=y`
- Features utilizadas:
  - `SEXO`
  - `EDAD_SIMPLE`
  - `EST_CIVIL`
  - `SEG_SOCIAL`
  - `NIVEL_EDU_GRUPO`
  - `ANO`
  - `MES`
- Codificación de variables categóricas mediante `LabelEncoder`
- Manejo de desbalance:
  - Random Forest → `class_weight="balanced_subsample"`
  - Árbol de Decisión → `class_weight="balanced"`

---

## ⚠️ Desbalance de Clases

La variable objetivo `NOM_667_OPS_GRUPO` presenta un desbalance importante:

- Clase mayoritaria: **Enfermedades del sistema circulatorio** (~28.4%)
- Clase minoritaria: **Signos, síntomas y afecciones mal definidas** (~0.5%)
- Ratio mayoría/minoría aproximado: **55:1**

Debido a este comportamiento:

- El **Accuracy** puede resultar engañoso.
- La métrica principal utilizada para seleccionar el modelo es el **F1-Score Weighted**.
- También se analiza el **F1 Macro** y el **Recall Macro** para evaluar desempeño en clases minoritarias.

---

## 📈 Resultados del Modelo

| Modelo | Accuracy | F1 Weighted ★ | F1 Macro | Recall Macro | Precision Weighted |
|---|---|---|---|---|---|
| Random Forest | 38.79% | 37.52% | 38.85% | 46.30% | 38.77% |
| Árbol de Decisión | 35.80% | 34.51% | 36.50% | 44.60% | 38.30% |

### ✅ Modelo Seleccionado: Random Forest

Razones de selección:

1. Mayor F1-Score Weighted bajo desbalance de clases.
2. Mejor desempeño en F1 Macro y Recall Macro.
3. Mejor capacidad para detectar clases minoritarias.
4. Uso de `class_weight="balanced_subsample"` para ajustar pesos dinámicamente en cada árbol del ensamble.
5. Mayor robustez general frente al Árbol de Decisión individual.

---

## 📌 Hallazgos del Modelado

- `EDAD_SIMPLE` fue la variable más importante del modelo (~60.7% de importancia).
- Las clases:
  - *Enfermedades del sistema circulatorio*
  - *Neoplasias*
  - *Todas las demás enfermedades*

  presentan perfiles demográficos muy similares, dificultando la separación entre categorías.

- Variables socioeconómicas como:
  - `SEG_SOCIAL`
  - `SEXO`

  aportaron menor capacidad predictiva.

- El desempeño del modelo está limitado por la naturaleza de las variables disponibles, que describen el perfil del individuo pero no la causa médica directa de fallecimiento.
## 🛠️ Tecnologías Utilizadas

---

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

## 💻 Instrucciones para Ejecutar en Local

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

## 🐳 Instrucciones para Ejecutar en Local (Docker)

### 1. Verificar instalación de Docker

Antes de iniciar, asegúrate de tener instalado y ejecutando Docker Desktop en el sistema.

Descargar desde:
https://www.docker.com/products/docker-desktop/

### 2. Construir la imagen del proyecto
docker build -t dashboard .

Este comando crea la imagen del proyecto con todas sus dependencias definidas en el Dockerfile.

### 3. Ejecutar el contenedor
docker run -e PORT=8050 -p 8050:8050 dashboard

Este comando inicia la aplicación dentro de un contenedor y expone el puerto 10000.

### 4. Abrir la aplicación

Abrir en el navegador:

http://localhost:8050

### 5. Detener la ejecución

Para detener la aplicación en ejecución:

Ctrl + C

---

## ☁️ Despliegue en Render

El proyecto está desplegado como servicio web en la plataforma **Render**, lo que permite acceder al dashboard de forma pública sin necesidad de ejecución local.

### ⚙️ Configuración del servicio

El despliegue se realiza directamente desde el repositorio de GitHub, utilizando **Docker** como entorno de ejecución.

### Configuración utilizada:

- **Branch:** main  
- **Environment:** Docker  
- **Build Method:** Dockerfile  
- **Start Command:** definido automáticamente por el Dockerfile  

---

### 🔄 Actualización del despliegue

Cada cambio realizado en la rama **main** del repositorio activa automáticamente un nuevo despliegue en Render.

En caso de que los cambios recientes no se reflejen, se puede ejecutar un **redeploy manual** desde el panel del servicio.

### 🌐 Acceso a la aplicación

🔗 **Link del dashboard:** https://mortalidad-dashboard-pr0q.onrender.com/

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
Proyecto de Visualización de Datos – Universidad

---

## 📝 Licencia

Este proyecto es de uso académico. Los datos utilizados son de acceso público.
