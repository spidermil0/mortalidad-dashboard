# 💀 Dashboard de Mortalidad — Medellín (2012–2021)

Aplicación interactiva desarrollada con **Dash y Plotly** para el análisis exploratorio de datos (EDA) y el modelado predictivo de la mortalidad en la ciudad de Medellín.

El proyecto integra técnicas de análisis descriptivo y aprendizaje automático con el objetivo de identificar patrones en las causas de defunción y analizar su comportamiento según variables demográficas, sociales y temporales.

**Autores:** Camilo González & Rubén Esguerra

---

## 📊 Estructura del Dashboard

El dashboard está organizado en las siguientes secciones:

1. **Introducción**
   Contexto general, indicadores clave (KPIs) y descripción del dataset.

2. **Problema de Análisis**
   Planteamiento de la pregunta de investigación y su relevancia.

3. **Objetivos**
   Definición del objetivo general y objetivos específicos.

4. **Análisis Univariado**
   Distribución de variables como edad, sexo y causa de muerte.

5. **Análisis Bivariado**
   Relación entre variables explicativas y la causa de muerte.

6. **Modelado Predictivo**
   Implementación de modelos de clasificación.

7. **Predicción Interactiva**
   Ingreso de datos por el usuario y predicción en tiempo real.

---

## 🗂️ Estructura del Proyecto

```
dashboard_mortalidad/
├── app.py                  # Aplicación principal
├── defunciones_clean.csv   # Dataset limpio (145,377 registros)
├── requirements.txt        # Dependencias
└── README.md               # Documentación
```

---

## ⚙️ Ejecución Local

### 1. Clonar el repositorio

```bash
git clone https://github.com/spidermil0/mortalidad-dashboard.git
cd mortalidad-dashboard
```

### 2. Crear entorno virtual

```bash
python -m venv venv
```

### 3. Activar entorno

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 5. Ejecutar la aplicación

```bash
python app.py
```

La aplicación estará disponible en:
http://localhost:8050

---

## 📦 Dataset

* **Fuente:** Registros de defunciones de Medellín
* **Período:** 2012–2021
* **Registros:** 145,377
* **Variables:** 11
* **Variable objetivo:** `NOM_667_OPS_GRUPO`

---

## 🤖 Modelos Implementados

| Modelo            | Descripción                           |
| ----------------- | ------------------------------------- |
| Random Forest     | Modelo de ensamble para clasificación |
| Árbol de Decisión | Modelo interpretable basado en reglas |

**División de datos:**
80% entrenamiento / 20% prueba (estratificado)

---
## Autores
Camilo Gonzalez & Rubén Esguerra

## 📄 Licencia

Uso académico y educativo.
