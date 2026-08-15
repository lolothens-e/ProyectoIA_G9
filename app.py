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
)

# Inyección de CSS para Tema Oscuro Profesional (Professional Dark Theme - High Contrast & WCAG AA)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* 1. Fondo Principal y Tipografía Global */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        background-color: #0F172A !important; /* Azul Slate Oscuro Profundo */
        color: #F8FAFC !important; /* Blanco Alto Contraste */
    }
    
    /* 2. Barra Lateral (Sidebar) */
    section[data-testid="stSidebar"] {
        background-color: #1E293B !important; /* Azul Slate Medio */
        border-right: 1px solid #334155 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }
    
    /* Tarjeta de Criterio de Partición en Sidebar */
    div[data-testid="stSidebar"] .stAlert {
        background-color: #0F172A !important;
        border: 1px solid #334155 !important;
        border-left: 4px solid #3B82F6 !important;
        color: #F8FAFC !important;
        border-radius: 10px !important;
    }

    /* 3. Encabezados y Subtítulos */
    h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }
    .stCaption, p, span {
        color: #94A3B8;
    }

    /* 4. Pestañas de Navegación (Pill Buttons) */
    button[data-baseweb="tab"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 20px !important;
        color: #94A3B8 !important;
        padding: 8px 20px !important;
        font-weight: 600 !important;
        margin-right: 8px !important;
        transition: all 0.2s ease-in-out;
    }
    button[data-baseweb="tab"]:hover {
        border-color: #3B82F6 !important;
        color: #F8FAFC !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-color: #3B82F6 !important;
        box-shadow: 0px 4px 14px rgba(37, 99, 235, 0.4) !important;
    }

    /* 5. Tarjetas Neumórficas de Métricas (KPIs) */
    div[data-testid="stMetric"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 18px 22px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25) !important;
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #38BDF8 !important; /* Azul destacado */
        font-size: 1.4rem !important;
        font-weight: 800 !important;
        white-space: normal !important;
        word-wrap: break-word !important;
    }

    /* 6. Campos de Selección y Controles */
    div[data-baseweb="select"] > div {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #F8FAFC !important;
        border-radius: 8px !important;
    }

    /* 7. Callouts / Banners de Alertas de Riesgo Personalizadas */
    .callout-success {
        background-color: rgba(16, 185, 129, 0.12) !important;
        border-left: 5px solid #10B981 !important;
        border-top: 1px solid rgba(16, 185, 129, 0.2);
        border-right: 1px solid rgba(16, 185, 129, 0.2);
        border-bottom: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 10px;
        padding: 16px;
        color: #6EE7B7 !important;
        font-weight: 600;
        margin-top: 14px;
        margin-bottom: 14px;
    }
    .callout-warning {
        background-color: rgba(245, 158, 11, 0.12) !important;
        border-left: 5px solid #F59E0B !important;
        border-top: 1px solid rgba(245, 158, 11, 0.2);
        border-right: 1px solid rgba(245, 158, 11, 0.2);
        border-bottom: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 10px;
        padding: 16px;
        color: #FDE68A !important;
        font-weight: 600;
        margin-top: 14px;
        margin-bottom: 14px;
    }
    .callout-error {
        background-color: rgba(239, 68, 68, 0.12) !important;
        border-left: 5px solid #EF4444 !important;
        border-top: 1px solid rgba(239, 68, 68, 0.2);
        border-right: 1px solid rgba(239, 68, 68, 0.2);
        border-bottom: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 10px;
        padding: 16px;
        color: #FCA5A5 !important;
        font-weight: 600;
        margin-top: 14px;
        margin-bottom: 14px;
    }

    /* Contenedor Limpio para Gráficos */
    .plot-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("ProyectoIA")

# Placeholder para las estadísticas globales (se rellenarán luego de cargar los datos)
stats_placeholder = st.sidebar.container()

# Panel lateral de configuración
st.sidebar.header("Configuración del Pipeline")
sample_size = st.sidebar.slider(
    "Tamaño de muestra de datos",
    min_value=1000,
    max_value=10000,
    value=3500,
    step=500,
)


# Cargar y procesar datos y modelos (Prioriza pre-entrenados para carga instantánea)
pretrained_mlp_path = "models/mlp_model.h5"
pretrained_baselines_path = "models/baseline_models.pkl"
pretrained_scaler_path = "models/scaler_encoder.pkl"
pretrained_df_path = "models/df_final_preprocessed.csv"
pretrained_results_path = "models/evaluation_results.csv"

has_pretrained = all(
    os.path.exists(p) for p in [
        pretrained_mlp_path, pretrained_baselines_path, pretrained_scaler_path, 
        pretrained_df_path, pretrained_results_path, "models/X_train.npy", "models/X_test.npy"
    ]
)

# Opción para forzar re-entrenamiento en la barra lateral
forzar_entrenamiento = st.sidebar.checkbox("Forzar entrenamiento en tiempo real", value=False)

if has_pretrained and not forzar_entrenamiento:
  st.sidebar.success("Datos y modelos pre-entrenados cargados con éxito (Carga instantánea)")
  
  # Cargar datos preprocesados
  df_final = pd.read_csv(pretrained_df_path)
  
  # Cargar configuraciones de preprocessing
  with open(pretrained_scaler_path, "rb") as f:
    scaler_config = pickle.load(f)
  
  # Crear preprocesador shell
  preprocesador = DataPreprocessor("data/raw/1Registro-Administrativo-Historico_2009-202X-Inicio.csv", "data/raw/2Registro-Administrativo-Historico_2009-2024-Fin.csv")
  preprocesador.scaler = scaler_config["scaler"]
  preprocesador.encoder = scaler_config["encoder"]
  preprocesador.feature_names = scaler_config["feature_names"]
  
  # Cargar matrices X_train y X_test para SHAP
  X_train = np.load("models/X_train.npy")
  X_test = np.load("models/X_test.npy")
  y_train = df_final["NivelRiesgoDesercion"].values[:len(X_train)] # proxy
  y_test = df_final["NivelRiesgoDesercion"].values[-len(X_test):] # proxy
  
  # Cargar MLP
  mlp = MLPClassifier(input_dim=X_train.shape[1], num_classes=3)
  mlp.model = tf.keras.models.load_model(pretrained_mlp_path)
  
  # Cargar baselines
  evaluador = BaselineEvaluator()
  with open(pretrained_baselines_path, "rb") as f:
    evaluador.modelos_entrenados = pickle.load(f)
    
  # Cargar tabla de resultados
  df_baseline = pd.read_csv(pretrained_results_path)
  info_split = "Partición temporal pre-entrenada (Test: >= 2023-2024)"
  st.sidebar.info(f"Criterio de Partición: {info_split}")
  
else:
  # Cargar y procesar datos en tiempo real
  if "preprocesador" not in st.session_state or st.session_state.get("sample_size_previo_data") != sample_size:
    with st.spinner("Cargando y procesando dataset histórico del MINEDUC..."):
      ruta_inicio = (
          "data/raw/1Registro-Administrativo-Historico_2009-202X-Inicio.xlsx"
      )
      ruta_fin = "data/raw/2Registro-Administrativo-Historico_2009-2024-Fin.xlsx"

      preprocesador = DataPreprocessor(ruta_inicio, ruta_fin)
      df_raw = preprocesador.cargar_y_fusionar_datasets(sample_size=sample_size)
      df_clean = preprocesador.limpiar_y_calcular_abandono(df_raw)
      df_final = preprocesador.discretizar_riesgo(df_clean)

      X, y = preprocesador.transformar_caracteristicas(df_final, is_training=True)
      
      st.session_state["preprocesador"] = preprocesador
      st.session_state["df_final"] = df_final
      st.session_state["X"] = X
      st.session_state["y"] = y
      st.session_state["sample_size_previo_data"] = sample_size

  preprocesador = st.session_state["preprocesador"]
  df_final = st.session_state["df_final"]
  X = st.session_state["X"]
  y = st.session_state["y"]

  # Partición temporal
  X_train, X_test, y_train, y_test, info_split = preprocesador.dividir_por_tiempo(
      df_final, X, y
  )
  st.sidebar.info(f"Criterio de Particion: {info_split}")

  # Entrenar en tiempo real
  if "modelos_entrenados" not in st.session_state or st.session_state.get("sample_size_previo") != sample_size or forzar_entrenamiento:
    with st.spinner("Entrenando modelos en el sistema..."):
      progress_status = st.empty()
      
      progress_status.info("**[1/6] Entrenando Perceptron Multicapa (MLP en Keras/CPU)...**")
      mlp = MLPClassifier(input_dim=X_train.shape[1], num_classes=3)
      mlp.entrenar(X_train, y_train, epochs=30, batch_size=32)
      
      evaluador = BaselineEvaluator()
      
      progress_status.info("**[2/6] Entrenando Random Forest...**")
      evaluador.modelos_entrenados["Random Forest"] = evaluador.modelos["Random Forest"].fit(X_train, y_train)
      
      progress_status.info("**[3/6] Entrenando Gradient Boosting...**")
      evaluador.modelos_entrenados["Gradient Boosting"] = evaluador.modelos["Gradient Boosting"].fit(X_train, y_train)
      
      progress_status.info("**[4/6] Entrenando SVM (RBF)...**")
      evaluador.modelos_entrenados["SVM (RBF)"] = evaluador.modelos["SVM (RBF)"].fit(X_train, y_train)
      
      progress_status.info("**[5/6] Entrenando Regresion Logistica...**")
      evaluador.modelos_entrenados["Regresión Logística"] = evaluador.modelos["Regresión Logística"].fit(X_train, y_train)
      
      progress_status.info("**[6/6] Entrenando KNN...**")
      evaluador.modelos_entrenados["KNN"] = evaluador.modelos["KNN"].fit(X_train, y_train)
      
      progress_status.info("**Calculando metricas de evaluacion...**")
      
      resultados = []
      for nombre, modelo in evaluador.modelos_entrenados.items():
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        
        resultados.append({
            "Modelo": nombre,
            "Accuracy": round(acc, 4),
            "Precision (Macro)": round(prec, 4),
            "Recall (Macro)": round(rec, 4),
            "F1-Score (Macro)": round(f1, 4),
        })
      
      df_baseline = pd.DataFrame(resultados)
      progress_status.empty()
      
      st.session_state["mlp"] = mlp
      st.session_state["evaluador"] = evaluador
      st.session_state["df_baseline"] = df_baseline
      st.session_state["modelos_entrenados"] = True
      st.session_state["sample_size_previo"] = sample_size

  mlp = st.session_state["mlp"]
  evaluador = st.session_state["evaluador"]
  df_baseline = st.session_state["df_baseline"]

# Rellenar el placeholder con las estadísticas globales al principio del sidebar
with stats_placeholder:
  st.subheader("Estadísticas Globales")
  st.metric("Planteles Analizados", len(df_final))
  st.metric(
      "Tasa Promedio Abandono", f"{df_final['Tasa_Abandono'].mean()*100:.2f}%"
  )
  st.metric("Niveles de Riesgo", "0 - Bajo, 1 - Medio, 2 - Alto")
  st.divider()

# Pestañas principales
tab1, tab2, tab3 = st.tabs(
    ["Semaforo de Riesgo", "Comparativa de Modelos", "Explicabilidad (SHAP)"]
)

with tab1:
  st.subheader("Búsqueda y Evaluación por Unidad Educativa (AMIE)")

  busqueda = st.text_input("Buscar plantel (por codigo AMIE o nombre de la institucion):", "")
  
  if busqueda:
    busqueda_clean = busqueda.strip().lower()
    df_filtrado = df_final[
        df_final["AMIE"].astype(str).str.lower().str.contains(busqueda_clean) | 
        df_final["Nombre_Institucion_inicio"].astype(str).str.lower().str.contains(busqueda_clean)
    ]
    
    if len(df_filtrado) == 0:
      st.warning("No se encontraron planteles con ese termino de busqueda. Por favor intente de nuevo.")
      st.stop()
    elif len(df_filtrado) > 1:
      opciones = {f"{row['AMIE']} - {row['Nombre_Institucion_inicio']}": row['AMIE'] for idx, row in df_filtrado.iterrows()}
      seleccion = st.selectbox(f"Se encontraron {len(df_filtrado)} coincidencias. Seleccione la institución exacta:", list(opciones.keys()))
      amie_seleccionado = opciones[seleccion]
      registro = df_final[df_final["AMIE"] == amie_seleccionado].iloc[0]
    else:
      registro = df_filtrado.iloc[0]
      amie_seleccionado = registro["AMIE"]
      st.success(f"Institucion seleccionada: **{registro['Nombre_Institucion_inicio']}** (AMIE: {amie_seleccionado})")
  else:
    registro = df_final.iloc[0]
    amie_seleccionado = registro["AMIE"]

  st.divider()
  st.subheader("Datos de la Institución (Pertinentes al Modelo)")
  
  # Extraer y formatear de manera segura las variables numéricas
  docentes = int(registro["Total_Docentes"]) if pd.notnull(registro.get("Total_Docentes")) else 0
  admin = int(registro["Total_Administrativos"]) if pd.notnull(registro.get("Total_Administrativos")) else 0
  estud = int(registro["Total_Estudiantes_inicio"]) if pd.notnull(registro.get("Total_Estudiantes_inicio")) else 0
  discap = int(registro["Estudiantes_con_discapacidad"]) if pd.notnull(registro.get("Estudiantes_con_discapacidad")) else 0

  # Columnas para métricas numéricas
  col_n1, col_n2, col_n3, col_n4 = st.columns(4)
  col_n1.metric("Docentes", docentes)
  col_n2.metric("Administrativos", admin)
  col_n3.metric("Estudiantes Inicio", estud)
  col_n4.metric("Estudiantes con Discapacidad", discap)
  
  # Columnas para variables categóricas
  col_c1, col_c2, col_c3 = st.columns(3)
  col_c1.markdown(f"**Sostenimiento:** {registro.get('Sostenimiento', 'Desconocido')}")
  col_c2.markdown(f"**Área:** {registro.get('Área', 'Desconocido')}")
  col_c3.markdown(f"**Jornada:** {registro.get('Jornada', 'Desconocido')}")
  
  col_c4, col_c5, col_c6 = st.columns(3)
  col_c4.markdown(f"**Régimen Escolar:** {registro.get('Regimen_Escolar', 'Desconocido')}")
  col_c5.markdown(f"**Jurisdicción:** {registro.get('Jurisdiccion', 'Desconocido')}")
  col_c6.markdown(f"**Modalidad:** {registro.get('Modalidad', 'Desconocido')}")
  
  st.divider()
  st.subheader("Resultado de la Evaluación de Riesgo")

  riesgo_map = {
      0: ("BAJO RIESGO DE DESERCION", "callout-success"),
      1: ("RIESGO MEDIO DE DESERCION", "callout-warning"),
      2: ("ALTO RIESGO DE DESERCION", "callout-error"),
  }

  etiqueta, clase_callout = riesgo_map[registro["NivelRiesgoDesercion"]]

  st.markdown(f"### Nivel de Alerta: **{etiqueta}**")
  if registro["NivelRiesgoDesercion"] == 0:
    st.markdown(
        f'<div class="callout-success"><b>Estado Estable:</b> La institucion {amie_seleccionado} mantiene niveles estables de retencion estudiantil.</div>',
        unsafe_allow_html=True,
    )
  elif registro["NivelRiesgoDesercion"] == 1:
    st.markdown(
        f'<div class="callout-warning"><b>Monitoreo Preventivo:</b> La institucion {amie_seleccionado} presenta fluctuaciones moderadas que requieren seguimiento.</div>',
        unsafe_allow_html=True,
    )
  else:
    st.markdown(
        f'<div class="callout-error"><b>Alerta Preventiva URGENTE:</b> La institucion {amie_seleccionado} requiere intervencion pedagogica inmediata.</div>',
        unsafe_allow_html=True,
    )


with tab2:
  st.subheader(
      "Evaluación Comparativa de Desempeño (MLP Propio vs. 5 Modelos de Línea"
      " Base)"
  )

  y_pred_mlp = mlp.predecir(X_test)
  acc_mlp = accuracy_score(y_test, y_pred_mlp)
  f1_mlp = f1_score(y_test, y_pred_mlp, average="macro", zero_division=0)

  fila_mlp = pd.DataFrame([{
      "Modelo": "MLP (Propio - Keras)",
      "Accuracy": round(acc_mlp, 4),
      "Precision (Macro)": round(
          precision_score(
              y_test, y_pred_mlp, average="macro", zero_division=0
          ),
          4,
      ),
      "Recall (Macro)": round(
          recall_score(y_test, y_pred_mlp, average="macro", zero_division=0), 4
      ),
      "F1-Score (Macro)": round(f1_mlp, 4),
  }])

  tabla_completa = pd.concat([fila_mlp, df_baseline], ignore_index=True)

  st.dataframe(tabla_completa, use_container_width=True)

  st.bar_chart(
      data=tabla_completa,
      x="Modelo",
      y="F1-Score (Macro)",
      color="#38BDF8",
  )

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