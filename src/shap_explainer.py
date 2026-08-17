import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shap


class SHAPExplainer:

  def __init__(
      self, model_predict_fn, X_train_sample: np.ndarray, feature_names: list
  ):
    """Inicializa el explicador SHAP usando KernelExplainer.

    :param model_predict_fn: Función de predicción del modelo (ej:
    mlp.model.predict)
    :param X_train_sample: Muestra del conjunto de entrenamiento para la línea
    base
    :param feature_names: Nombres de las variables/características procesadas
    """
    self.predict_fn = model_predict_fn
    self.feature_names = feature_names

    # Usamos una muestra de fondo de máximo 30 instancias para acelerar el cálculo
    num_samples = min(30, len(X_train_sample))
    self.background = shap.sample(X_train_sample, num_samples)
    self.explainer = shap.KernelExplainer(self.predict_fn, self.background)

  def calcular_explicabilidad(
      self, X_sample: np.ndarray, n_samples: int = 10
  ):
    """Calcula Shapley values sobre una muestra de prueba."""
    muestra = X_sample[: min(n_samples, len(X_sample))]
    shap_values = self.explainer.shap_values(muestra, nsamples=100)
    return shap_values, muestra

  def generar_grafico_resumen(self, X_sample: np.ndarray, n_samples: int = 10):
    """Genera una figura de matplotlib con el gráfico Summary de SHAP."""
    shap_values, muestra = self.calcular_explicabilidad(X_sample, n_samples)

    # 1. Identificar índices a mantener (solo las 6 variables solicitadas)
    indices_mantener = []
    for idx, name in enumerate(self.feature_names):
      name_lower = name.lower()
      keep = False
      if "total_estudiantes_inicio" in name_lower:
        keep = True
      elif "total_docentes" in name_lower:
        keep = True
      elif "area" in name_lower or "área" in name_lower:
        keep = True
      elif "estudiantes_con_discapacidad" in name_lower:
        keep = True
      elif "sostenimiento" in name_lower:
        keep = True
      elif "regimen_escolar" in name_lower or "regimen" in name_lower:
        keep = True
      
      if keep:
        indices_mantener.append(idx)

    # 2. Filtrar nombres de características
    feature_names_filtrados = [self.feature_names[i] for i in indices_mantener]

    # 3. Filtrar matriz de muestra
    muestra_filtrada = muestra[:, indices_mantener]

    # 4. Filtrar valores SHAP (que pueden ser una lista de arrays o un solo array)
    if isinstance(shap_values, list):
      shap_values_filtrados = [arr[:, indices_mantener] for arr in shap_values]
    elif isinstance(shap_values, np.ndarray):
      if len(shap_values.shape) == 3:
        if shap_values.shape[2] == len(self.feature_names):
          shap_values_filtrados = shap_values[:, :, indices_mantener]
        elif shap_values.shape[1] == len(self.feature_names):
          shap_values_filtrados = shap_values[:, indices_mantener, :]
        else:
          shap_values_filtrados = shap_values
      else:
        shap_values_filtrados = shap_values[:, indices_mantener]
    else:
      shap_values_filtrados = shap_values

    fig, ax = plt.subplots(figsize=(11, 7))

    nombres_clases = ["Deserción baja", "Deserción media", "Deserción alta"]

    shap.summary_plot(
        shap_values_filtrados,
        muestra_filtrada,
        feature_names=feature_names_filtrados,
        class_names=nombres_clases,
        show=False,
        plot_type="bar",
    )

    # Position the legend outside the bar diagram on the right to prevent overlap
    plt.legend(
        title="Nivel de Riesgo",
        labels=nombres_clases,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )


    plt.tight_layout()
    return fig