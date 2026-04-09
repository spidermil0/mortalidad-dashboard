# 💀 Dashboard de Mortalidad — Medellín 2012–2021

Dashboard interactivo construido con **Python + Dash + Plotly** para explorar los patrones de mortalidad en la ciudad de Medellín durante el período 2012–2021.  
Desarrollado como proyecto de portafolio de Ciencia de Datos.

**Autores:** Camilo González & Rubén Esguerra

---

## 📸 Vista del Dashboard

El dashboard incluye cuatro secciones principales:

| Sección | Contenido |
|---|---|
| 📊 Vista General | KPIs, distribución OPS, histograma de edad, seguridad social |
| 📈 Análisis Temporal | Evolución anual y mensual de defunciones |
| 👥 Análisis Demográfico | OPS × Sexo, heatmap etario, nivel educativo |
| 🗺️ Análisis Geográfico | Top 10 comunas por causa de muerte |

---

## 🗂️ Estructura del Proyecto

```
project_dashboard/
├── app.py                  # Aplicación principal Dash
├── defunciones_clean.csv   # Dataset limpio (145,377 registros)
├── requirements.txt        # Dependencias Python
├── Procfile                # Configuración para Render / Railway
└── README.md               # Este archivo
```

---

## ⚙️ Instalación Local

### 1. Clonar el repositorio

```bash
git clone https://github.com/spidermil0/mortalidad-dashboard.git
cd mortalidad-dashboard
```

### 2. Crear entorno virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar localmente

```bash
python app.py
```

Abre tu navegador en **http://localhost:8050**

---

## 🚀 Despliegue Gratuito en Render

### Paso 1 — Subir a GitHub

```bash
# Desde la carpeta del proyecto
git init
git add .
git commit -m "feat: dashboard mortalidad medellín"

# Crear repositorio en github.com y luego:
git remote add origin https://github.com/spidermil0/mortalidad-dashboard.git
git branch -M main
git push -u origin main
```

> ⚠️ Asegúrate de incluir `defunciones_clean.csv` en el commit (no lo agregues al `.gitignore`).

---

### Paso 2 — Crear cuenta en Render

1. Ve a [https://render.com](https://render.com) y regístrate gratis.
2. Selecciona **"New +"** → **"Web Service"**.

---

### Paso 3 — Conectar GitHub

1. Conecta tu cuenta de GitHub cuando Render lo solicite.
2. Busca y selecciona el repositorio `mortalidad-dashboard`.
3. Haz clic en **"Connect"**.

---

### Paso 4 — Configurar el servicio

| Campo | Valor |
|---|---|
| **Name** | `mortalidad-dashboard` |
| **Region** | `Oregon (US West)` |
| **Branch** | `main` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:server` |
| **Instance Type** | `Free` |

---

### Paso 5 — Deploy

1. Haz clic en **"Create Web Service"**.
2. Render instalará las dependencias automáticamente (~3 min).
3. Cuando el status cambie a **"Live"**, tu dashboard estará disponible en:

```
https://mortalidad-dashboard.onrender.com
```

> ⏱️ **Nota:** El plan gratuito de Render entra en "sleep" tras 15 min de inactividad.  
> El primer acceso puede tardar ~30 segundos en despertar.

---

## 🔄 Alternativa: Railway

```bash
# Instalar Railway CLI
npm install -g @railway/cli

railway login
railway init
railway up
```

O usa la interfaz web en [https://railway.app](https://railway.app):  
`New Project → Deploy from GitHub Repo`.

---

## 📦 Variables y Dataset

| Variable | Tipo | Descripción |
|---|---|---|
| `NOM_667_OPS_GRUPO` | Categórica | Causa de muerte (variable objetivo) |
| `ANO` | Numérica | Año de defunción (2012–2021) |
| `MES` | Numérica | Mes de defunción (1–12) |
| `SEXO` | Categórica | Sexo del fallecido |
| `EDAD_SIMPLE` | Numérica | Edad en años |
| `ETAREO_QUIN` | Categórica | Grupo etario quinquenal |
| `EST_CIVIL` | Categórica | Estado civil |
| `SEG_SOCIAL` | Categórica | Régimen de seguridad social |
| `NIVEL_EDU_GRUPO` | Categórica | Nivel educativo agrupado |
| `COMUNA_RES` | Categórica | Comuna de residencia |
| `BARRIO_RES` | Categórica | Barrio de residencia |

---

## 🛠️ Stack Tecnológico

- **Python 3.11+**
- **Dash 2.17** — Framework web reactivo
- **Dash Bootstrap Components** — Tema FLATLY (Bootswatch)
- **Plotly 5** — Visualizaciones interactivas
- **Pandas 2.2** — Procesamiento de datos
- **Gunicorn** — Servidor WSGI para producción

---

## 📄 Licencia

MIT — libre para uso académico y personal.
