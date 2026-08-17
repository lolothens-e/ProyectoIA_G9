# Sistema de Clasificación y Evaluación de Riesgo de Deserción Escolar

Este proyecto tiene como objetivo clasificar y evaluar el nivel de riesgo de deserción escolar en unidades educativas de Ecuador. Utiliza datos abiertos del Ministerio de Educación (MINEDUC), implementa un modelo de Perceptrón Multicapa (MLP) en Keras/TensorFlow y cinco modelos de línea base comparativos (Random Forest, Gradient Boosting, SVM, Regresión Logística y KNN), además de proporcionar explicabilidad e interpretabilidad a través de valores SHAP (Shapley Additive exPlanations).

---

## Estructura del Repositorio

La estructura del proyecto está organizada de la siguiente manera:

* **`app.py`**: Interfaz web interactiva desarrollada en Streamlit. Permite consultar planteles educativos por código AMIE, visualizar semáforos de riesgo, realizar inferencias para el siguiente periodo y analizar la explicabilidad de las decisiones del modelo.
* **`exemplify_model.py`**: Script ejecutable independiente que demuestra la inferencia offline y la explicabilidad de SHAP sobre registros de ejemplo sin necesidad de levantar el servidor web.
* **`generar_notebook.py`**: Automatización para generar el archivo del notebook de Jupyter con los experimentos de entrenamiento.
* **`requirements.txt`**: Listado de dependencias de Python necesarias para la ejecución (Streamlit, TensorFlow, Scikit-Learn, Pandas, SHAP, etc.).
* **`test_fase1.py`**: Script de validación para la Fase 1 (ETL, preprocesamiento de datos, cálculo de abandono y discretización de riesgo).
* **`test_fase2.py`**: Script de validación para la Fase 2 (entrenamiento del clasificador MLP propio, entrenamiento de modelos base y generación de la tabla comparativa de métricas).
* **`test_shap.py`**: Script de validación para probar el módulo de explicabilidad SHAP local.
* **`src/`**: Módulos con la lógica central del sistema:
  * [`data_preprocessing.py`](file:///Users/lolothens/Code/ProyectoIA_G9%20v2/src/data_preprocessing.py): Carga, fusión, limpieza, ingeniería de características (Inicio/Fin de año) y división temporal de datos.
  * [`mlp_classifier.py`](file:///Users/lolothens/Code/ProyectoIA_G9%20v2/src/mlp_classifier.py): Definición de arquitectura del Perceptrón Multicapa, entrenamiento y funciones de predicción en TensorFlow/Keras.
  * [`baseline_evaluator.py`](file:///Users/lolothens/Code/ProyectoIA_G9%20v2/src/baseline_evaluator.py): Entrenamiento y evaluación de algoritmos base en Scikit-Learn.
  * [`shap_explainer.py`](file:///Users/lolothens/Code/ProyectoIA_G9%20v2/src/shap_explainer.py): Gestión de explicabilidad local y global usando KernelExplainer y generación de gráficos de importancia.
  * [`download_utils.py`](file:///Users/lolothens/Code/ProyectoIA_G9%20v2/src/download_utils.py): Descarga automática de modelos y datos persistidos desde almacenamiento en la nube.
  * [`export_pretrained_models.py`](file:///Users/lolothens/Code/ProyectoIA_G9%20v2/src/export_pretrained_models.py): Exportación manual y persistencia de transformaciones de datos y modelos entrenados.
* **`data/`**: Directorio para almacenar los conjuntos de datos crudos (en formato `.xlsx` y `.csv` provenientes del MINEDUC) y procesados.
* **`models/`**: Contiene los artefactos de modelos serializados (`.pkl`, `.h5`), variables preprocesadas (`.npy`) y la tabla de resultados de las evaluaciones.
* **`notebooks/`**: Contiene [`Entrenatorio_y_Evaluacion.ipynb`](file:///Users/lolothens/Code/ProyectoIA_G9%20v2/notebooks/Entrenatorio_y_Evaluacion.ipynb), utilizado para ejecutar interactivamente la exploración y entrenamiento de los modelos.

---

## Requisitos de Sistema

* **Python 3.11** (recomendado para asegurar total compatibilidad con las librerías científicas y TensorFlow 2.x).
* **Git** (para control de versiones).

---

## Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO
```

### 2. Configurar el Entorno Virtual

* **En macOS y Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
* **En Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

### 3. Instalar las Dependencias
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt ipykernel
```

---

## Instrucciones de Ejecución

### Aplicación Web Interactiva (Streamlit)
Para iniciar la interfaz de visualización y semáforo de riesgo por AMIE:
```bash
streamlit run app.py
```
El servidor local se levantará por defecto en `http://localhost:8501`.

### Inferencia Offline de Prueba
Para ejecutar un análisis rápido de inferencia y explicabilidad SHAP en consola sin requerir la interfaz web:
```bash
python exemplify_model.py
```

### Scripts de Validación por Fases

* **Prueba de Fase 1 (ETL y preprocesamiento):**
  ```bash
  python test_fase1.py
  ```
* **Prueba de Fase 2 (Entrenamiento comparativo de modelos):**
  ```bash
  python test_fase2.py
  ```
* **Validación de SHAP:**
  ```bash
  python test_shap.py
  ```
