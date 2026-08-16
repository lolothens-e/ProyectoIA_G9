import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pickle
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from src.baseline_evaluator import BaselineEvaluator
from src.data_preprocessing import DataPreprocessor
from src.mlp_classifier import MLPClassifier
from src.shap_explainer import SHAPExplainer
import streamlit as st

st.set_page_config(
    page_title="Sistema de Alerta de Deserción Escolar - MINEDUC Ecuador",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de CSS para Tema Institucional MINEDUC Ecuador (High Contrast & WCAG AA - Sin Logos)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* 1. Fondo Principal y Tipografía Global */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        background-color: #F4F6F9 !important;
        color: #1E293B !important;
    }

    /* Ajuste de contenedor principal */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }
    
    /* 2. Header Institucional Mineduc */
    .mineduc-header {
        background-color: #0B2545;
        color: #FFFFFF;
        padding: 24px 32px 18px 32px;
        border-radius: 8px 8px 0 0;
        margin-bottom: 0px;
        box-shadow: 0 4px 12px rgba(11, 37, 69, 0.15);
    }
    .mineduc-header-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    .mineduc-title {
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #FFFFFF !important;
        margin: 0;
        text-transform: uppercase;
    }
    .mineduc-subtitle {
        font-size: 0.95rem;
        color: #93C5FD !important;
        font-weight: 500;
        margin-top: 4px;
    }
    .mineduc-tag {
        background-color: #007791 !important;
        color: #FFFFFF !important;
        padding: 6px 16px !important;
        border-radius: 20px !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Franja Tricolor Bandera de Ecuador */
    .ecuador-stripe {
        height: 5px;
        background: linear-gradient(90deg, #FFCC00 0%, #FFCC00 50%, #003399 50%, #003399 75%, #FF0000 75%, #FF0000 100%);
        margin-bottom: 20px;
        border-radius: 0 0 6px 6px;
    }

    /* 3. Contenedor de Pestañas (Barra de Navegación Funcional MINEDUC) */
    div[data-testid="stTabs"] {
        background-color: transparent !important;
    }
    
    div[data-baseweb="tab-list"], [data-testid="stTabsHeader"] {
        background-color: #0B2545 !important;
        padding: 6px 12px !important;
        border-radius: 8px !important;
        gap: 8px !important;
        margin-bottom: 24px !important;
        border-bottom: none !important;
        box-shadow: 0 4px 10px rgba(11, 37, 69, 0.15) !important;
    }
    
    /* Pestañas Inactivas */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 22px !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    button[data-baseweb="tab"] *, 
    button[data-baseweb="tab"] p, 
    button[data-baseweb="tab"] span, 
    button[data-baseweb="tab"] div {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }
    
    button[data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.15) !important;
    }
    button[data-baseweb="tab"]:hover * {
        color: #FFFFFF !important;
    }
    
    /* Pestaña Seleccionada / Activa */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #007791 !important;
        border-radius: 6px !important;
        box-shadow: 0 2px 8px rgba(0, 119, 145, 0.4) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] *,
    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] span,
    button[data-baseweb="tab"][aria-selected="true"] div {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    div[data-baseweb="tab-highlight"], div[data-testid="stTabHeaderHighlight"] {
        display: none !important;
    }

    /* 4. Botones Principales (st.button) */
    div.stButton > button, 
    button[kind="secondary"], 
    button[kind="primary"], 
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-primary"] {
        background-color: #0B2545 !important;
        color: #FFFFFF !important;
        border: 1px solid #0B2545 !important;
        border-radius: 6px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 2px 6px rgba(11, 37, 69, 0.2) !important;
        transition: all 0.2s ease-in-out !important;
    }

    div.stButton > button *, 
    button[kind="secondary"] *, 
    button[kind="primary"] * {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    div.stButton > button:hover, 
    button[kind="secondary"]:hover, 
    button[kind="primary"]:hover {
        background-color: #007791 !important;
        border-color: #007791 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(0, 119, 145, 0.35) !important;
    }

    /* 5. Barra Lateral (Sidebar) Estilo "Temas Importantes" */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #1E293B !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #0B2545 !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stSidebar"] .stAlert {
        background-color: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
        border-left: 4px solid #1D4ED8 !important;
        color: #1E40AF !important;
        border-radius: 8px !important;
    }

    /* 6. Encabezados y Subtítulos */
    h1, h2, h3, h4, h5, h6 {
        color: #0B2545 !important;
        font-weight: 700 !important;
    }
    p, span {
        color: #334155;
    }
    .stCaption {
        color: #64748B !important;
    }

    /* 7. Tarjetas de Métricas (KPIs Institucionales) */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 16px 20px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        border-top: 4px solid #0B2545 !important;
    }
    div[data-testid="stMetric"] label {
        color: #475569 !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #0B2545 !important;
        font-size: 1.45rem !important;
        font-weight: 800 !important;
        white-space: normal !important;
        word-wrap: break-word !important;
    }

    /* Grid de Servicios Institucionales Mineduc */
    .services-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 14px;
        margin-bottom: 24px;
    }
    .service-card {
        background-color: #1B365D;
        color: #FFFFFF;
        padding: 14px 16px;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        border-left: 4px solid #007791;
    }
    .service-card h4 {
        color: #FFFFFF !important;
        margin: 4px 0 2px 0;
        font-size: 0.95rem;
    }
    .service-card p {
        color: #CBD5E1 !important;
        font-size: 0.78rem;
        margin: 0;
    }

    /* 8. Campos de Selección y Controles */
    label[data-testid="stWidgetLabel"], label {
        color: #0B2545 !important;
        font-weight: 700 !important;
    }
    div[data-baseweb="input"] > div, input {
        background-color: #FFFFFF !important;
        color: #0F2240 !important;
        border-color: #CBD5E1 !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        color: #0F2240 !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="select"] * {
        color: #0F2240 !important;
    }

    /* 9. Callouts / Banners de Alertas de Riesgo Personalizadas */
    .callout-success {
        background-color: #ECFDF5 !important;
        border-left: 6px solid #10B981 !important;
        border-top: 1px solid #A7F3D0;
        border-right: 1px solid #A7F3D0;
        border-bottom: 1px solid #A7F3D0;
        border-radius: 8px;
        padding: 16px 20px;
        color: #065F46 !important;
        font-weight: 600;
        margin-top: 14px;
        margin-bottom: 14px;
    }
    .callout-warning {
        background-color: #FFFBEB !important;
        border-left: 6px solid #F59E0B !important;
        border-top: 1px solid #FDE68A;
        border-right: 1px solid #FDE68A;
        border-bottom: 1px solid #FDE68A;
        border-radius: 8px;
        padding: 16px 20px;
        color: #92400E !important;
        font-weight: 600;
        margin-top: 14px;
        margin-bottom: 14px;
    }
    .callout-error {
        background-color: #FEF2F2 !important;
        border-left: 6px solid #EF4444 !important;
        border-top: 1px solid #FCA5A5;
        border-right: 1px solid #FCA5A5;
        border-bottom: 1px solid #FCA5A5;
        border-radius: 8px;
        padding: 16px 20px;
        color: #991B1B !important;
        font-weight: 600;
        margin-top: 14px;
        margin-bottom: 14px;
    }
    .callout-info {
        background-color: #EFF6FF !important;
        border-left: 6px solid #2563EB !important;
        border-top: 1px solid #BFDBFE;
        border-right: 1px solid #BFDBFE;
        border-bottom: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 16px 20px;
        color: #1E40AF !important;
        font-weight: 600;
        margin-top: 14px;
        margin-bottom: 14px;
    }

    /* Contenedor Limpio para Gráficos */
    .plot-card {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 16px;
    }

    /* Pie de página institucional (Footer) */
    .mineduc-footer {
        background-color: #0B2545;
        color: #FFFFFF;
        padding: 24px 32px;
        border-radius: 8px;
        margin-top: 40px;
        font-size: 0.85rem;
    }
    .mineduc-footer-links {
        display: flex;
        justify-content: space-around;
        border-bottom: 1px solid rgba(255, 255, 255, 0.15);
        padding-bottom: 14px;
        margin-bottom: 14px;
        flex-wrap: wrap;
        gap: 12px;
    }
    .mineduc-footer-links a {
        color: #93C5FD;
        text-decoration: none;
        font-weight: 600;
    }
    .mineduc-footer-info {
        text-align: center;
        color: #CBD5E1;
        line-height: 1.6;
    }
</style>

<!-- Banner Superior Institucional (Sin Logos Oficiales) -->
<div class="mineduc-header">
    <div class="mineduc-header-top">
        <div>
            <h1 class="mineduc-title">SISTEMA DE ALERTA DE DESERCIÓN ESCOLAR</h1>
            <div class="mineduc-subtitle">Plataforma Tecnológica de Analítica Predictiva y Evaluación de Riesgo Educativo</div>
        </div>
        <div class="mineduc-tag">Módulo Analítico IA</div>
    </div>
</div>
<div class="ecuador-stripe"></div>
""", unsafe_allow_html=True)

st.sidebar.title("MINEDUC - Analítica")
stats_placeholder = st.sidebar.container()

# Rutas de artefactos pre-entrenados
pretrained_mlp_path = "models/mlp_model.h5"
pretrained_baselines_path = "models/baseline_models.pkl"
pretrained_scaler_path = "models/scaler_encoder.pkl"
pretrained_df_path = "models/df_final_preprocessed.csv"
pretrained_results_path = "models/evaluation_results.csv"

# Verificación específica para baseline_models.pkl (excluido de git por tamaño >100MB)
if not os.path.exists(pretrained_baselines_path):
    st.warning("⚠️ No se encontró localmente el archivo de modelos base (`models/baseline_models.pkl`).")
    st.info("📦 Debido a las políticas de tamaño de Git (>100MB), este archivo se encuentra alojado externamente en Google Drive.")
    
    col_dl1, col_dl2 = st.columns([1, 1])
    with col_dl1:
        if st.button("⬇️ Descargar automáticamente desde Google Drive", use_container_width=True, type="primary"):
            from src.download_utils import download_file_from_google_drive
            bar = st.progress(0, text="Iniciando descarga...")
            def _update_progress(pct):
                pct_int = int(pct * 100)
                bar.progress(pct_int, text=f"Descargando baseline_models.pkl ({pct_int}%)...")
            try:
                download_file_from_google_drive("1MpKowch9JG9Qx4AU1JD8D2FUfORM1PXF", pretrained_baselines_path, progress_callback=_update_progress)
                st.success("✅ ¡Descarga completada con éxito! Recargando aplicación...")
                st.rerun()
            except Exception as e:
                st.error(f"Error al descargar automáticamente: {e}")
    with col_dl2:
        st.markdown("""
        **Descarga Manual:**  
        👉 [Descargar desde Google Drive](https://drive.google.com/file/d/1MpKowch9JG9Qx4AU1JD8D2FUfORM1PXF/view?usp=drive_link)  
        *(Guarda el archivo en la carpeta `models/` con el nombre exacto `baseline_models.pkl`)*
        """)
    st.stop()

has_pretrained = all(
    os.path.exists(p) for p in [
        pretrained_mlp_path, pretrained_baselines_path, pretrained_scaler_path, 
        pretrained_df_path, pretrained_results_path, "models/X_train.npy", "models/X_test.npy"
    ]
)

# Carga de datos y artefactos
if has_pretrained:
  df_final = pd.read_csv(pretrained_df_path, low_memory=False)
  # Asegurar estandarización de columnas
  col_anio = next((c for c in df_final.columns if "lectivo" in c.lower()), "Año_lectivo")
  if col_anio != "Año_lectivo":
    df_final.rename(columns={col_anio: "Año_lectivo"}, inplace=True)
  df_final["Año_lectivo"] = df_final["Año_lectivo"].astype(str).str.replace(" Inicio", "", case=False).str.replace(" Fin", "", case=False).str.strip()

  col_nombre = next((c for c in df_final.columns if "nombre" in c.lower() and "institucion" in c.lower()), None)
  if col_nombre and col_nombre != "Nombre_Institucion":
    df_final["Nombre_Institucion"] = df_final[col_nombre]
  elif "Nombre_Institucion" not in df_final.columns:
    df_final["Nombre_Institucion"] = "Institución Educativa"
  
  with open(pretrained_scaler_path, "rb") as f:
    scaler_config = pickle.load(f)
  
  preprocesador = DataPreprocessor("data/raw/1Registro-Administrativo-Historico_2009-202X-Inicio.csv", "data/raw/2Registro-Administrativo-Historico_2009-2024-Fin.csv")
  preprocesador.scaler = scaler_config["scaler"]
  preprocesador.encoder = scaler_config["encoder"]
  preprocesador.feature_names = scaler_config["feature_names"]
  
  X_train = np.load("models/X_train.npy")
  X_test = np.load("models/X_test.npy")
  
  # Cargar MLP
  mlp = MLPClassifier(input_dim=X_train.shape[1], num_classes=3)
  mlp.model = tf.keras.models.load_model(pretrained_mlp_path)
  
  # Cargar Baselines
  evaluador = BaselineEvaluator()
  with open(pretrained_baselines_path, "rb") as f:
    evaluador.modelos_entrenados = pickle.load(f)
    
  df_baseline = pd.read_csv(pretrained_results_path)
  
  st.sidebar.success("Base de datos y modelos institucionales listos.")
  st.sidebar.info("Criterio de Validación: Hold-Out Temporal (Train: < 2024-2025 | Test: >= 2024-2025)")
else:
  st.error("No se encontraron los modelos pre-entrenados en `models/`. Ejecute `python src/export_pretrained_models.py`.")
  st.stop()

# Rellenar estadísticas globales en sidebar
with stats_placeholder:
  st.subheader("Estadísticas Globales")
  st.metric("Total Registros Históricos", f"{len(df_final):,}")
  st.metric("Planteles Únicos", f"{df_final['AMIE'].nunique():,}")
  tasa_prom = df_final['Tasa_Abandono'].dropna().mean() * 100
  st.metric("Tasa Promedio Abandono", f"{tasa_prom:.2f}%")
  st.divider()

# Diccionario unificado de modelos para inferencia dinámica
diccionario_modelos = {
    "MLP (Perceptrón Multicapa - Keras)": mlp,
    "Random Forest": evaluador.modelos_entrenados.get("Random Forest"),
    "Gradient Boosting": evaluador.modelos_entrenados.get("Gradient Boosting"),
    "SVM (RBF)": evaluador.modelos_entrenados.get("SVM (RBF)"),
    "Regresión Logística": evaluador.modelos_entrenados.get("Regresión Logística"),
    "KNN (K-Nearest Neighbors)": evaluador.modelos_entrenados.get("KNN"),
}

# Funciones auxiliares para calcular predicción de inferencia
def predecir_con_modelo(modelo_obj, X_vec):
  if isinstance(modelo_obj, MLPClassifier):
    return modelo_obj.predecir(X_vec)[0]
  elif hasattr(modelo_obj, "predict"):
    return int(modelo_obj.predict(X_vec)[0])
  else:
    raise ValueError("Modelo no reconocido para inferencia.")

def obtener_siguiente_periodo(anio_actual: str, lista_anios: list) -> str:
  """Calcula el año lectivo inmediatamente posterior."""
  if anio_actual in lista_anios:
    idx = lista_anios.index(anio_actual)
    if idx + 1 < len(lista_anios):
      return lista_anios[idx + 1]
  
  # Si es el último año de la lista o no está, proyectar año numérico
  partes = anio_actual.split("-")
  if len(partes) == 2 and partes[0].isdigit() and partes[1].isdigit():
    inicio_sig = int(partes[0]) + 1
    fin_sig = int(partes[1]) + 1
    return f"{inicio_sig}-{fin_sig}"
  return "2025-2026"

# Pestañas principales
tab1, tab2, tab3 = st.tabs(
    ["Semáforo de Riesgo y Predicción", "Comparativa de Modelos", "Explicabilidad (SHAP)"]
)

# -------------------------------------------------------------
# TAB 1: Semáforo de Riesgo, Selección de Año y Predicción T+1
# -------------------------------------------------------------
with tab1:
  st.markdown("""
  <div class="services-grid">
      <div class="service-card">
          <h4>🔍 1. Búsqueda Institucional</h4>
          <p>Localice el plantel por código AMIE o nombre</p>
      </div>
      <div class="service-card">
          <h4>📅 2. Selección de Año Base</h4>
          <p>Consulte registros históricos consolidados (≤ T)</p>
      </div>
      <div class="service-card">
          <h4>🤖 3. Selección de Modelo IA</h4>
          <p>Compare predicciones entre MLP y 5 baselines</p>
      </div>
      <div class="service-card">
          <h4>🔮 4. Predicción Preventiva</h4>
          <p>Proyecte el riesgo de deserción del año siguiente (T+1)</p>
      </div>
  </div>
  """, unsafe_allow_html=True)

  st.subheader("Búsqueda y Selección de Unidad Educativa")

  col_busq1, col_busq2 = st.columns([2, 1])
  with col_busq1:
    busqueda = st.text_input("Buscar plantel (por código AMIE o nombre):", value="", placeholder="Ej: 09H00018 o Guayaquil")
  
  with col_busq2:
    # Selector rápido de AMIEs destacados si no hay búsqueda
    amies_disponibles = sorted(df_final["AMIE"].unique())
    amie_default = amies_disponibles[0] if amies_disponibles else ""

  if busqueda.strip():
    busqueda_clean = busqueda.strip().lower()
    df_filtrado = df_final[
        df_final["AMIE"].astype(str).str.lower().str.contains(busqueda_clean) | 
        df_final["Nombre_Institucion"].astype(str).str.lower().str.contains(busqueda_clean)
    ]
    
    if len(df_filtrado) == 0:
      st.warning("No se encontraron planteles con ese término de búsqueda. Intente con otro código o nombre.")
      st.stop()
    
    amies_coincidentes = df_filtrado["AMIE"].unique()
    if len(amies_coincidentes) > 1:
      mapa_nombres = {}
      for am in amies_coincidentes:
        sub = df_filtrado[df_filtrado["AMIE"] == am]
        nom = sub["Nombre_Institucion"].iloc[0] if "Nombre_Institucion" in sub.columns else "Institución"
        mapa_nombres[f"{am} - {nom}"] = am
      seleccion_amie = st.selectbox(f"Se encontraron {len(amies_coincidentes)} instituciones. Seleccione:", list(mapa_nombres.keys()))
      amie_activo = mapa_nombres[seleccion_amie]
    else:
      amie_activo = amies_coincidentes[0]
  else:
    amie_activo = amie_default

  # Filtrar todos los registros históricos del AMIE seleccionado
  df_inst_todos = df_final[df_final["AMIE"] == amie_activo].sort_values("Año_lectivo").copy()
  nombre_inst = df_inst_todos["Nombre_Institucion"].iloc[0] if "Nombre_Institucion" in df_inst_todos.columns else "Institución Educativa"

  st.success(f"🏫 Institución Seleccionada: **{nombre_inst}** (Código AMIE: **{amie_activo}**)")

  st.divider()

  # -------------------------------------------------------------
  # 1. SELECTOR DE AÑO Y TABLA DE DATOS REALES (≤ Año Seleccionado)
  # -------------------------------------------------------------
  st.subheader("1. Lógica de Registros Históricos y Selección de Año Base")

  anios_inst = sorted(df_inst_todos["Año_lectivo"].unique().tolist())
  anios_todos_dataset = sorted(df_final["Año_lectivo"].unique().tolist())

  # Layout con selector de año y selector de modelo
  col_s1, col_s2 = st.columns([1, 1])

  with col_s1:
    idx_default_anio = len(anios_inst) - 2 if len(anios_inst) >= 2 else 0
    anio_base_seleccionado = st.selectbox(
        "📅 Seleccione el Año Base de Análisis (T):",
        options=anios_inst,
        index=idx_default_anio,
        help="La tabla inferior solo mostrará los datos reales registrados hasta este año seleccionado para evitar cualquier fuga de información."
    )

  with col_s2:
    modelo_seleccionado_nombre = st.selectbox(
        "🤖 Seleccione el Modelo de Inteligencia Artificial para la Predicción:",
        options=list(diccionario_modelos.keys()),
        index=0,
        help="Permite generar y comparar la predicción utilizando cualquiera de los modelos evaluados."
    )

  # Filtrar datos históricos reales HASTA el año base seleccionado (<= anio_base_seleccionado)
  df_historico_visible = df_inst_todos[df_inst_todos["Año_lectivo"] <= anio_base_seleccionado].copy()
  registro_actual = df_historico_visible.iloc[-1]

  # -------------------------------------------------------------
  # Ficha Técnica Institucional del Año Base Seleccionado
  # -------------------------------------------------------------
  st.markdown(f"#### 🏛️ Ficha Técnica de la Institución — Periodo {anio_base_seleccionado}")

  # Métricas numéricas del año base
  docentes = int(registro_actual["Total_Docentes"]) if pd.notnull(registro_actual.get("Total_Docentes")) else 0
  admin = int(registro_actual["Total_Administrativos"]) if pd.notnull(registro_actual.get("Total_Administrativos")) else 0
  estud = int(registro_actual["Total_Estudiantes_inicio"]) if pd.notnull(registro_actual.get("Total_Estudiantes_inicio")) else 0
  discap = int(registro_actual["Estudiantes_con_discapacidad"]) if pd.notnull(registro_actual.get("Estudiantes_con_discapacidad")) else 0

  col_m1, col_m2, col_m3, col_m4 = st.columns(4)
  col_m1.metric("Docentes Registrados", docentes)
  col_m2.metric("Personal Administrativo", admin)
  col_m3.metric("Estudiantes al Inicio", estud)
  col_m4.metric("Estudiantes con Discapacidad", discap)

  # Ubicación y características administrativas
  prov = registro_actual.get("Provincia", registro_actual.get("Provincia_inicio", "Desconocida"))
  cant = registro_actual.get("Canton", registro_actual.get("Canton_inicio", "Desconocido"))
  parr = registro_actual.get("Parroquia", registro_actual.get("Parroquia_inicio", "Desconocida"))
  sost = registro_actual.get("Sostenimiento", registro_actual.get("Sostenimiento_inicio", "Desconocido"))
  area = registro_actual.get("Área", registro_actual.get("Area", "Desconocida"))
  jorn = registro_actual.get("Jornada", registro_actual.get("Jornada_inicio", "Desconocida"))
  regi = registro_actual.get("Regimen_Escolar", registro_actual.get("Regimen_Escolar_inicio", "Desconocido"))
  moda = registro_actual.get("Modalidad", registro_actual.get("Modallidad", "Desconocida"))

  col_d1, col_d2, col_d3 = st.columns(3)
  col_d1.markdown(f"📍 **Ubicación:** {prov} / {cant} / {parr}")
  col_d2.markdown(f"🏢 **Sostenimiento:** {sost}")
  col_d3.markdown(f"🌲 **Área Geográfica:** {area}")

  col_d4, col_d5, col_d6 = st.columns(3)
  col_d4.markdown(f"⏰ **Jornada:** {jorn}")
  col_d5.markdown(f"📚 **Régimen Escolar:** {regi}")
  col_d6.markdown(f"🎓 **Modalidad:** {moda}")

  st.markdown("<br>", unsafe_allow_html=True)

  # -------------------------------------------------------------
  # Tabla de Registros Históricos Reales
  # -------------------------------------------------------------
  st.markdown(f"#### 📊 Historial Operativo Consolidado (Periodo 2009-2010 hasta {anio_base_seleccionado})")

  # Formatear columnas para visualización clara
  col_anio_presente = next((c for c in df_historico_visible.columns if "lectivo" in c.lower()), "Año_lectivo")
  columnas_mostrar = {
      col_anio_presente: "Año Lectivo",
      "Total_Docentes": "Docentes",
      "Total_Administrativos": "Administrativos",
      "Total_Estudiantes_inicio": "Estudiantes Inicio",
      "Estudiantes_con_discapacidad": "Con Discapacidad",
      "Total_Estudiantes_Fin": "Estudiantes Fin",
      "Promovidos": "Promovidos",
      "No promovidos": "No Promovidos",
      "Abandono": "Casos Abandono",
      "Tasa_Abandono": "Tasa Abandono Real",
      "NivelRiesgoDesercion": "Nivel de Riesgo Real",
  }

  cols_presentes = [c for c in columnas_mostrar.keys() if c in df_historico_visible.columns]
  df_tabla_display = df_historico_visible[cols_presentes].copy()

  # Mapear etiquetas legibles sobre las columnas existentes
  mapa_nivel_texto = {0: "🟢 0 - Bajo", 1: "🟡 1 - Medio", 2: "🔴 2 - Alto"}
  if "NivelRiesgoDesercion" in df_tabla_display.columns:
    df_tabla_display["NivelRiesgoDesercion"] = df_tabla_display["NivelRiesgoDesercion"].map(mapa_nivel_texto).fillna("Sin Registro Fin")
  if "Tasa_Abandono" in df_tabla_display.columns:
    df_tabla_display["Tasa_Abandono"] = df_tabla_display["Tasa_Abandono"].apply(lambda v: f"{v*100:.2f}%" if pd.notnull(v) else "Pendiente")

  # Renombrar columnas al formato legible final
  df_tabla_display.rename(columns={k: v for k, v in columnas_mostrar.items() if k in df_tabla_display.columns}, inplace=True)

  st.dataframe(df_tabla_display, use_container_width=True, hide_index=True)
  st.caption(f"ℹ️ **Control Temporal:** Mostrando estrictamente {len(df_tabla_display)} registro(s) histórico(s) hasta el año {anio_base_seleccionado}. Para ver más años en el historial, cambie el 'Año Base de Análisis' arriba.")

  st.divider()

  # -------------------------------------------------------------
  # 2. PREDICCIÓN PARA EL AÑO SIGUIENTE (T + 1)
  # -------------------------------------------------------------
  anio_siguiente = obtener_siguiente_periodo(anio_base_seleccionado, anios_todos_dataset)

  st.subheader(f"2. Predicción de Deserción Escolar para el Año Siguiente ({anio_siguiente})")

  st.markdown(f"""
  Se utilizará el modelo **{modelo_seleccionado_nombre}** alimentado con las características de inicio de ciclo del periodo **{anio_siguiente}** (o proyección operativa) para clasificar el nivel de riesgo preventivo.
  """)

  # Botón de Predicción
  boton_predecir = st.button(f"🔮 Predecir Deserción para el Año {anio_siguiente} con {modelo_seleccionado_nombre}")

  # Estado de sesión para persistir la predicción al cambiar entre pestañas o modelos
  if boton_predecir:
    st.session_state["prediccion_activa"] = {
        "amie": amie_activo,
        "anio_base": anio_base_seleccionado,
        "anio_sig": anio_siguiente,
        "modelo_nombre": modelo_seleccionado_nombre
    }

  pred_data = st.session_state.get("prediccion_activa", None)

  # Verificar si la predicción guardada coincide con los parámetros actuales
  if pred_data and pred_data.get("amie") == amie_activo and pred_data.get("anio_base") == anio_base_seleccionado:
    modelo_a_usar_nombre = modelo_seleccionado_nombre
    modelo_obj = diccionario_modelos[modelo_a_usar_nombre]

    # Obtener el registro de features para el año siguiente (T+1)
    df_registro_siguiente = df_inst_todos[df_inst_todos["Año_lectivo"] == anio_siguiente]

    if len(df_registro_siguiente) > 0:
      fila_siguiente = df_registro_siguiente.iloc[0].copy()
    else:
      # Si el año siguiente está más allá del dataset, se toma la última configuración conocida de inicio
      fila_siguiente = df_inst_todos.iloc[-1].copy()
      fila_siguiente["Año_lectivo"] = anio_siguiente
      fila_siguiente["Abandono"] = np.nan
      fila_siguiente["Total_Estudiantes_Fin"] = np.nan
      fila_siguiente["Tasa_Abandono"] = np.nan
      fila_siguiente["NivelRiesgoDesercion"] = np.nan

    # Transformar a vector X de inferencia (Únicamente variables de INICIO)
    X_inferencia = preprocesador.transformar_fila_inferencia(fila_siguiente)

    # Inferencia con el modelo seleccionado
    clase_predicha = predecir_con_modelo(modelo_obj, X_inferencia)

    riesgo_etiquetas = {
        0: ("BAJO RIESGO DE DESERCIÓN", "callout-success", "🟢"),
        1: ("RIESGO MEDIO DE DESERCIÓN", "callout-warning", "🟡"),
        2: ("ALTO RIESGO DE DESERCIÓN", "callout-error", "🔴"),
    }

    etiqueta_pred, callout_pred, icono_pred = riesgo_etiquetas[clase_predicha]

    # Comprobar si existen datos reales de fin de año para anio_siguiente
    tiene_datos_reales_fin = (
        pd.notnull(fila_siguiente.get("Abandono")) and 
        pd.notnull(fila_siguiente.get("Total_Estudiantes_Fin")) and 
        pd.notnull(fila_siguiente.get("NivelRiesgoDesercion")) and
        not np.isnan(fila_siguiente.get("NivelRiesgoDesercion", np.nan))
    )

    st.markdown(f"### Resultado de la Inferencia — Periodo {anio_siguiente}")

    # Columnas de resultados
    if tiene_datos_reales_fin:
      clase_real = int(fila_siguiente["NivelRiesgoDesercion"])
      etiqueta_real, _, icono_real = riesgo_etiquetas[clase_real]
      tasa_real_val = float(fila_siguiente["Tasa_Abandono"]) * 100
      estud_fin_val = int(fila_siguiente["Total_Estudiantes_Fin"])
      abandono_val = int(fila_siguiente["Abandono"])

      col_res1, col_res2, col_res3, col_res4 = st.columns(4)
      col_res1.metric("Modelo Utilizado", modelo_a_usar_nombre)
      col_res2.metric("Nivel Predicho por IA", f"{icono_pred} {etiqueta_pred}")
      col_res3.metric("Nivel Real Histórico", f"{icono_real} {etiqueta_real}")
      col_res4.metric("Tasa Real al Cierre", f"{tasa_real_val:.2f}% ({abandono_val} de {estud_fin_val})")

      # Comparativa de concordancia
      if clase_predicha == clase_real:
        st.markdown(f"""
        <div class="callout-success">
            <b>🎯 Acierto Exacto del Modelo ({modelo_a_usar_nombre}):</b><br>
            El modelo anticipó exitosamente la categoría de <b>{etiqueta_pred}</b> para el año lectivo {anio_siguiente}, coincidiendo con el registro oficial de fin de ciclo del MINEDUC.
        </div>
        """, unsafe_allow_html=True)
      else:
        st.markdown(f"""
        <div class="callout-warning">
            <b>📊 Comparativa de Estimación ({modelo_a_usar_nombre}):</b><br>
            El modelo clasificó preventivamente la institución como <b>{etiqueta_pred}</b> frente a un registro real consolidado de <b>{etiqueta_real}</b> (Tasa real: {tasa_real_val:.2f}%).
        </div>
        """, unsafe_allow_html=True)

    else:
      # -------------------------------------------------------------
      # 3. RESTRICCIÓN DE DATOS FUTUROS (Sin Data Leakage)
      # -------------------------------------------------------------
      col_f1, col_f2 = st.columns([1, 1])
      col_f1.metric("Modelo Utilizado", modelo_a_usar_nombre)
      col_f2.metric("Nivel Predicho por IA", f"{icono_pred} {etiqueta_pred}")

      st.markdown(f"""
      <div class="callout-info">
          <b>🔒 Restricción de Datos Futuros — Predicción Preventiva 100% Ciega:</b><br>
          Para el periodo escolar <b>{anio_siguiente}</b>, el Ministerio de Educación aún no cuenta con registros de cierre de año (Promovidos / No promovidos / Abandono).<br>
          La comparativa con datos reales se encuentra <b>automáticamente deshabilitada</b>, cumpliendo con la restricción estricta de no utilizar información futura y evitando el <i>data leakage</i>.
      </div>
      """, unsafe_allow_html=True)

    # Botones de regeneración rápida con otros modelos
    st.markdown("**🔄 Probar y regenerar predicción con otro modelo:**")
    cols_btn = st.columns(len(diccionario_modelos))
    for idx, (m_nombre, m_inst) in enumerate(diccionario_modelos.items()):
      with cols_btn[idx]:
        if st.button(m_nombre.split(" (")[0], key=f"btn_quick_{idx}"):
          st.session_state["prediccion_activa"] = {
              "amie": amie_activo,
              "anio_base": anio_base_seleccionado,
              "anio_sig": anio_siguiente,
              "modelo_nombre": m_nombre
          }
          st.rerun()

# -------------------------------------------------------------
# TAB 2: Comparativa de Modelos (Layout Pulido y Mejorado)
# -------------------------------------------------------------
with tab2:
  st.subheader("Evaluación Comparativa de Rendimiento (MLP Propio vs. 5 Modelos de Línea Base)")
  st.caption("Validación rigurosa sobre el conjunto de prueba cronológico (Hold-out temporal con datos posteriores a 2024).")

  # Métricas de resumen en tarjetas superiores
  mejor_modelo_f1 = df_baseline.sort_values("F1-Score (Macro)", ascending=False).iloc[0]
  mejor_modelo_acc = df_baseline.sort_values("Accuracy", ascending=False).iloc[0]

  col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
  col_kpi1.metric("Total Modelos Evaluados", len(df_baseline))
  col_kpi2.metric("Mejor F1-Score (Macro)", f"{mejor_modelo_f1['F1-Score (Macro)']*100:.2f}%", f"{mejor_modelo_f1['Modelo']}")
  col_kpi3.metric("Mejor Exactitud (Accuracy)", f"{mejor_modelo_acc['Accuracy']*100:.2f}%", f"{mejor_modelo_acc['Modelo']}")

  st.divider()

  # Layout en 2 columnas: Tabla a la izquierda y Gráfico de Barras a la derecha
  col_grid1, col_grid2 = st.columns([1, 1])

  with col_grid1:
    st.markdown("#### 📋 Tabla Comparativa de Métricas")
    st.dataframe(
        df_baseline.style.highlight_max(subset=["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"], color="#BFDBFE"),
        use_container_width=True,
        hide_index=True
    )
    st.info("💡 **Nota Metodológica:** Todas las métricas fueron calculadas aplicando partición temporal y ponderación de clases (`class_weight='balanced'`) para garantizar equidad ante el desbalance natural de la deserción.")

  with col_grid2:
    st.markdown("#### 📊 Comparativa de F1-Score (Macro)")
    
    # Gráfico con matplotlib para mayor control estético y consistencia
    fig_comp, ax_comp = plt.subplots(figsize=(7, 4.2))
    colores = ["#007791" if "MLP" in m else "#0B2545" for m in df_baseline["Modelo"]]
    
    barras = ax_comp.barh(df_baseline["Modelo"], df_baseline["F1-Score (Macro)"], color=colores, edgecolor="none", height=0.6)
    ax_comp.set_xlabel("F1-Score (Macro)", fontsize=10, fontweight="bold", color="#0B2545")
    ax_comp.set_xlim(0, 1.0)
    ax_comp.grid(axis="x", linestyle="--", alpha=0.5)
    ax_comp.spines["top"].set_visible(False)
    ax_comp.spines["right"].set_visible(False)
    ax_comp.spines["left"].set_color("#CBD5E1")
    ax_comp.spines["bottom"].set_color("#CBD5E1")
    ax_comp.tick_params(colors="#1E293B", labelsize=9)

    for barra in barras:
      ancho = barra.get_width()
      ax_comp.annotate(
          f"{ancho:.4f}",
          xy=(ancho, barra.get_y() + barra.get_height() / 2),
          xytext=(5, 0),
          textcoords="offset points",
          ha="left",
          va="center",
          fontsize=9,
          fontweight="bold",
          color="#0B2545"
      )

    plt.tight_layout()
    st.pyplot(fig_comp)

# -------------------------------------------------------------
# TAB 3: Explicabilidad SHAP (Exactamente igual, sin modificaciones)
# -------------------------------------------------------------
with tab3:
  st.subheader("Análisis de Explicabilidad con Valores SHAP")
  st.write(
      "Visualización del impacto y peso de las características sociodemográficas e"
      " institucionales de Inicio de Año sobre las predicciones del Perceptrón"
      " Multicapa."
  )

  if st.button("Generar Gráfico SHAP"):
    with st.spinner("Calculando valores de Shapley sobre el modelo MLP..."):
      explainer = SHAPExplainer(
          mlp.model.predict, X_train, preprocesador.feature_names
      )
      fig = explainer.generar_grafico_resumen(X_test, n_samples=15)
      st.pyplot(fig)

# Pie de página institucional MINEDUC Ecuador
st.markdown("""
<div class="mineduc-footer">
    <div class="mineduc-footer-links">
        <span><b>Módulos del Sistema:</b></span>
        <span>•</span>
        <span>Semáforo de Riesgo y Predicción T+1</span>
        <span>•</span>
        <span>Comparativa de 6 Modelos IA</span>
        <span>•</span>
        <span>Explicabilidad SHAP</span>
        <span>•</span>
        <span>Registros Abiertos MINEDUC (2009 - 2026)</span>
    </div>
    <div class="mineduc-footer-info">
        <b>Sistema de Alerta Temprana de Deserción Escolar</b> • Proyecto IA - Grupo 9<br>
        Desarrollado con Perceptrón Multicapa (MLP en Keras), 5 Modelos de Línea Base y SHAP Explainer<br>
        Fuente de Datos: Registros Administrativos Históricos Abiertos del Ministerio de Educación de Ecuador (2009 - 2026)
    </div>
</div>
""", unsafe_allow_html=True)