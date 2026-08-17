import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:

  def __init__(self, ruta_inicio: str, ruta_fin: str):
    self.ruta_inicio = ruta_inicio
    self.ruta_fin = ruta_fin
    self.scaler = StandardScaler()
    self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    self.feature_names = []
    self.cols_categoricas = [
        "Sostenimiento",
        "Área",
        "Jornada",
        "Regimen_Escolar",
        "Jurisdiccion",
        "Modalidad",
    ]
    self.cols_numericas = [
        "Total_Docentes",
        "Total_Administrativos",
        "Total_Estudiantes_inicio",
        "Estudiantes_con_discapacidad",
    ]

  def cargar_y_fusionar_datasets(
      self, sample_size: int = 2500, persistent_sample: bool = True
  ) -> pd.DataFrame:
    """Carga y fusiona datasets asegurando identificadores limpios, sufijos claros y soporte multianual."""
    csv_inicio = self.ruta_inicio.replace(".xlsx", ".csv")
    csv_fin = self.ruta_fin.replace(".xlsx", ".csv")

    encoding = "latin1"
    if os.path.exists(csv_inicio) and os.path.exists(csv_fin):
      df_inicio = pd.read_csv(csv_inicio, encoding=encoding, low_memory=False)
      df_fin = pd.read_csv(csv_fin, encoding=encoding, low_memory=False)
    else:
      df_inicio = pd.read_excel(self.ruta_inicio)
      df_fin = pd.read_excel(self.ruta_fin)

    # Limpieza y estandarización de nombres de columnas
    df_inicio.columns = df_inicio.columns.str.strip()
    df_fin.columns = df_fin.columns.str.strip()

    # Estandarización de la clave AMIE
    for col in [
        "Codigo_Institucion",
        "CODIGO_INSTITUCION",
        "Codigo_institucion",
        "amie",
    ]:
      if col in df_inicio.columns:
        df_inicio.rename(columns={col: "AMIE"}, inplace=True)
      if col in df_fin.columns:
        df_fin.rename(columns={col: "AMIE"}, inplace=True)

    df_inicio["AMIE"] = df_inicio["AMIE"].astype(str).str.strip()
    df_fin["AMIE"] = df_fin["AMIE"].astype(str).str.strip()

    # Estandarizar nombre de columna de año lectivo
    col_anio_ini = next(
        (c for c in df_inicio.columns if "lectivo" in c.lower() or "ao" in c.lower() or "año" in c.lower()),
        "Año_lectivo",
    )
    col_anio_fin = next(
        (c for c in df_fin.columns if "lectivo" in c.lower() or "ao" in c.lower() or "año" in c.lower()),
        "Año_lectivo",
    )

    anios_ini_clean = (
        df_inicio[col_anio_ini].astype(str).str.replace(" Inicio", "", case=False).str.strip()
    )
    if col_anio_ini != "Año_lectivo" and col_anio_ini in df_inicio.columns:
      df_inicio.drop(columns=[col_anio_ini], inplace=True)
    df_inicio["Año_lectivo"] = anios_ini_clean

    anios_fin_clean = (
        df_fin[col_anio_fin].astype(str).str.replace(" Fin", "", case=False).str.strip()
    )
    if col_anio_fin != "Año_lectivo" and col_anio_fin in df_fin.columns:
      df_fin.drop(columns=[col_anio_fin], inplace=True)
    df_fin["Año_lectivo"] = anios_fin_clean

    # Si se pide muestra representativa persistente (instituciones con historial de al menos 6 años y variedad de año inicial)
    if persistent_sample and sample_size is not None and sample_size < len(df_inicio):
      g_amies = df_inicio.groupby("AMIE").agg(
          primer_anio=("Año_lectivo", "min"),
          total_anios=("Año_lectivo", "nunique"),
      ).reset_index()
      g_valid = g_amies[g_amies["total_anios"] >= 6]

      selected_amies = []
      for anio, grp in g_valid.groupby("primer_anio"):
        if anio == "2009-2010":
          n_sample = min(len(grp), int(sample_size * 0.35))
        else:
          n_sample = len(grp)
        selected_amies.extend(grp.sample(n=n_sample, random_state=42)["AMIE"].tolist())

      if len(selected_amies) > sample_size:
        top_amies = pd.Series(selected_amies).sample(n=sample_size, random_state=42).values
      else:
        top_amies = np.array(selected_amies)

      df_inicio = df_inicio[df_inicio["AMIE"].isin(top_amies)].copy()
      df_fin = df_fin[df_fin["AMIE"].isin(top_amies)].copy()

    # Columnas a traer de Fin de año lectivo
    cols_fin_keep = ["AMIE", "Año_lectivo", "Total_Estudiantes", "Promovidos", "No promovidos", "Abandono"]
    cols_fin_exist = [c for c in cols_fin_keep if c in df_fin.columns]

    # MERGE con sufijos explícitos (_inicio vs _fin)
    df_merged = pd.merge(
        df_inicio,
        df_fin[cols_fin_exist],
        on=["AMIE", "Año_lectivo"],
        how="left",
        suffixes=("_inicio", "_fin"),
    )

    # Renombrar columnas clave si es necesario
    if "Total_Estudiantes_inicio" not in df_merged.columns and "Total_Estudiantes" in df_merged.columns:
      df_merged.rename(columns={"Total_Estudiantes": "Total_Estudiantes_inicio"}, inplace=True)
    if "Área" not in df_merged.columns and "Area" in df_merged.columns:
      df_merged["Área"] = df_merged["Area"]
    if "Modalidad" not in df_merged.columns and "Modallidad" in df_merged.columns:
      df_merged["Modalidad"] = df_merged["Modallidad"]

    return df_merged

  def limpiar_y_calcular_abandono(self, df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la Tasa de Abandono aislando la variable objetivo y permitiendo registros futuros sin etiqueta."""
    df = df.copy()

    col_estudiantes_fin = next(
        (
            c
            for c in df.columns
            if ("total_estudiantes" in c.lower() or "estudiantes" in c.lower())
            and ("fin" in c.lower() or c.endswith("_fin"))
        ),
        None,
    )
    if not col_estudiantes_fin:
      col_estudiantes_fin = next(
          (
              c
              for c in df.columns
              if "total_estudiantes_fin" in c.lower() or "total_estudiantes" in c.lower()
          ),
          None,
      )

    col_abandono = next(
        (c for c in df.columns if "abandono" in c.lower()), None
    )

    if col_estudiantes_fin and col_abandono:
      df["Total_Estudiantes_Fin"] = pd.to_numeric(df[col_estudiantes_fin], errors="coerce")
      df["Abandono"] = pd.to_numeric(df[col_abandono], errors="coerce")

      # Cálculo de tasa de abandono solo donde hay datos de fin
      mask_valido = (df["Total_Estudiantes_Fin"] > 0) & (df["Abandono"].notnull())
      df.loc[mask_valido, "Tasa_Abandono"] = (
          df.loc[mask_valido, "Abandono"] / df.loc[mask_valido, "Total_Estudiantes_Fin"]
      ).clip(0.0, 1.0)
    else:
      df["Total_Estudiantes_Fin"] = np.nan
      df["Abandono"] = np.nan
      df["Tasa_Abandono"] = np.nan

    return df

  def discretizar_riesgo(self, df: pd.DataFrame) -> pd.DataFrame:
    """Categoriza la Tasa de Abandono en Bajo (0), Medio (1) y Alto (2)."""
    df = df.copy()
    valid_rates = df["Tasa_Abandono"].dropna()

    if len(valid_rates) > 0:
      quantiles = valid_rates.quantile([0.33, 0.66]).values
      q_low, q_high = quantiles[0], quantiles[1]
      if q_low == q_high:
        q_low, q_high = 0.02, 0.08
    else:
      q_low, q_high = 0.02, 0.08

    def asignar_clase(tasa):
      if pd.isna(tasa):
        return np.nan
      if tasa <= q_low:
        return 0
      elif tasa <= q_high:
        return 1
      else:
        return 2

    df["NivelRiesgoDesercion"] = df["Tasa_Abandono"].apply(asignar_clase)
    return df

  def transformar_caracteristicas(
      self, df: pd.DataFrame, is_training: bool = True
  ):
    """Transforma ÚNICAMENTE variables de INICIO de año lectivo para construir la matriz X y evitar Data Leakage."""
    cols_categoricas = [
        "Sostenimiento",
        "Área",
        "Jornada",
        "Regimen_Escolar",
        "Jurisdiccion",
        "Modalidad",
    ]

    cols_numericas_candidatas = [
        "Total_Docentes",
        "Total_Administrativos",
        "Total_Estudiantes_inicio",
        "Estudiantes_con_discapacidad",
    ]

    # Normalizar nombres de columnas
    df_clean = df.copy()
    if "Área" not in df_clean.columns and "Area" in df_clean.columns:
      df_clean["Área"] = df_clean["Area"]
    if "Modalidad" not in df_clean.columns and "Modallidad" in df_clean.columns:
      df_clean["Modalidad"] = df_clean["Modallidad"]

    cols_cat_existentes = [c for c in cols_categoricas if c in df_clean.columns]
    cols_num_existentes = [c for c in cols_numericas_candidatas if c in df_clean.columns]

    df_clean[cols_num_existentes] = df_clean[cols_num_existentes].fillna(0)
    df_clean[cols_cat_existentes] = df_clean[cols_cat_existentes].fillna("Desconocido")

    if is_training:
      cat_encoded = self.encoder.fit_transform(df_clean[cols_cat_existentes])
      num_scaled = self.scaler.fit_transform(df_clean[cols_num_existentes])
    else:
      cat_encoded = self.encoder.transform(df_clean[cols_cat_existentes])
      num_scaled = self.scaler.transform(df_clean[cols_num_existentes])

    X_processed = np.hstack([num_scaled, cat_encoded])

    if "NivelRiesgoDesercion" in df_clean.columns:
      y = df_clean["NivelRiesgoDesercion"].values
    else:
      y = np.full(len(df_clean), np.nan)

    encoded_cat_names = list(
        self.encoder.get_feature_names_out(cols_cat_existentes)
    )
    self.feature_names = cols_num_existentes + encoded_cat_names

    return X_processed, y

  def transformar_fila_inferencia(self, fila_o_dict) -> np.ndarray:
    """Convierte una sola institución (registro de Inicio de año) en vector escalado para predicción."""
    if isinstance(fila_o_dict, pd.Series):
      df_single = pd.DataFrame([fila_o_dict])
    elif isinstance(fila_o_dict, dict):
      df_single = pd.DataFrame([fila_o_dict])
    else:
      df_single = fila_o_dict.copy()

    X_single, _ = self.transformar_caracteristicas(df_single, is_training=False)
    return X_single

  def dividir_por_tiempo(self, df: pd.DataFrame, X: np.ndarray, y: np.ndarray, anio_corte: str = None):
    """Aplica la partición cronológica (Time-based Split) para evitar sesgo temporal."""
    # Filtrar solo instancias con etiqueta válida para entrenamiento/evaluación
    mask_valid_y = ~np.isnan(y)
    df_valid = df[mask_valid_y].copy()
    X_valid = X[mask_valid_y]
    y_valid = y[mask_valid_y].astype(int)

    if "Año_lectivo" not in df_valid.columns:
      corte = int(len(df_valid) * 0.8)
      return (
          X_valid[:corte],
          X_valid[corte:],
          y_valid[:corte],
          y_valid[corte:],
          "Partición secuencial (80% anterior / 20% reciente)",
      )

    anios_unicos = sorted(df_valid["Año_lectivo"].unique())
    if len(anios_unicos) <= 1:
      corte = int(len(df_valid) * 0.8)
      return (
          X_valid[:corte],
          X_valid[corte:],
          y_valid[:corte],
          y_valid[corte:],
          f"Partición por ordenamiento de lote ({anios_unicos[0]})",
      )

    if anio_corte is None or anio_corte not in anios_unicos:
      anio_test = anios_unicos[-1]
    else:
      anio_test = anio_corte

    mask_train = (df_valid["Año_lectivo"] < anio_test).values
    mask_test = (df_valid["Año_lectivo"] >= anio_test).values

    if mask_test.sum() < 20 and len(anios_unicos) > 2:
      anio_test = anios_unicos[-2]
      mask_train = (df_valid["Año_lectivo"] < anio_test).values
      mask_test = (df_valid["Año_lectivo"] >= anio_test).values

    X_train, y_train = X_valid[mask_train], y_valid[mask_train]
    X_test, y_test = X_valid[mask_test], y_valid[mask_test]

    info_split = f"Train: < {anio_test} | Test: >= {anio_test}"
    return X_train, X_test, y_train, y_test, info_split