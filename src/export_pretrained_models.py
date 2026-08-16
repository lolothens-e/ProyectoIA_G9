import os
import sys
sys.path.append(os.path.abspath('.'))
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.data_preprocessing import DataPreprocessor
from src.mlp_classifier import MLPClassifier
from src.baseline_evaluator import BaselineEvaluator

print("🚀 Generando modelos y binarios pre-entrenados para Streamlit...")

ruta_inicio = "data/raw/1Registro-Administrativo-Historico_2009-202X-Inicio.csv"
ruta_fin = "data/raw/2Registro-Administrativo-Historico_2009-2024-Fin.csv"

preprocesador = DataPreprocessor(ruta_inicio, ruta_fin)
df_raw = preprocesador.cargar_y_fusionar_datasets(sample_size=2000, persistent_sample=True)
df_clean = preprocesador.limpiar_y_calcular_abandono(df_raw)
df_final = preprocesador.discretizar_riesgo(df_clean)

X, y = preprocesador.transformar_caracteristicas(df_final, is_training=True)

X_train, X_test, y_train, y_test, info_split = preprocesador.dividir_por_tiempo(df_final, X, y)
print(f"📌 División temporal: {info_split} (Train: {len(X_train)} | Test: {len(X_test)})")

os.makedirs("models", exist_ok=True)

# 1. Entrenar MLP y guardar .h5
print("🧠 Entrenando MLP...")
mlp = MLPClassifier(input_dim=X_train.shape[1], num_classes=3)
mlp.entrenar(X_train, y_train, epochs=40, batch_size=32)
mlp.model.save("models/mlp_model.h5")

# 2. Entrenar modelos de línea base y guardar .pkl
print("📊 Entrenando modelos de línea base...")
evaluador = BaselineEvaluator()
df_results_baseline = evaluador.entrenar_y_evaluar_todos(X_train, y_train, X_test, y_test)

with open("models/baseline_models.pkl", "wb") as f:
    pickle.dump(evaluador.modelos_entrenados, f)

# 3. Guardar scaler, encoder y feature_names
scaler_config = {
    "scaler": preprocesador.scaler,
    "encoder": preprocesador.encoder,
    "feature_names": preprocesador.feature_names
}
with open("models/scaler_encoder.pkl", "wb") as f:
    pickle.dump(scaler_config, f)

# 4. Calcular métricas completas e incluir MLP
y_pred_mlp = mlp.predecir(X_test)
acc_mlp = accuracy_score(y_test, y_pred_mlp)
prec_mlp = precision_score(y_test, y_pred_mlp, average="macro", zero_division=0)
rec_mlp = recall_score(y_test, y_pred_mlp, average="macro", zero_division=0)
f1_mlp = f1_score(y_test, y_pred_mlp, average="macro", zero_division=0)

fila_mlp = pd.DataFrame([{
    "Modelo": "MLP (Propio - Keras)",
    "Accuracy": round(acc_mlp, 4),
    "Precision (Macro)": round(prec_mlp, 4),
    "Recall (Macro)": round(rec_mlp, 4),
    "F1-Score (Macro)": round(f1_mlp, 4),
}])

df_final_results = pd.concat([fila_mlp, df_results_baseline], ignore_index=True)
df_final_results.to_csv("models/evaluation_results.csv", index=False)

# 5. Guardar dataset procesado y matrices X_train / X_test
df_final.to_csv("models/df_final_preprocessed.csv", index=False)
np.save("models/X_train.npy", X_train)
np.save("models/X_test.npy", X_test)

print("✅ ¡Modelos pre-entrenados y artefactos binarios guardados exitosamente!")
