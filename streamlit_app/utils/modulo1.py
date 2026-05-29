"""
Funciones de carga y inferencia para el Módulo 1 (Predicción de demanda).

Centraliza:
  - La definición de la clase LSTMConEmbedding (idéntica al notebook de entrenamiento)
  - La carga del modelo entrenado (.pth) y de sus metadatos (.json)
  - La preparación de las features temporales (codificación cíclica + flag festivos)
  - La normalización Min-Max por ruta (parámetros del JSON)
  - La inferencia en dos modos:
      * Modo evaluación: predice nov 2024 (datos reales disponibles → comparación)
      * Modo predicción: predice el futuro a partir del último día del dataset
  - El cálculo PCA de los embeddings de ruta
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


# =============================================================================
# RUTAS DE LOS ARTEFACTOS
# =============================================================================

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data"

PATH_MODEL    = MODELS_DIR / "lstm_modelo_global.pth"
PATH_METADATA = MODELS_DIR / "metadata_modelo_global.json"
PATH_DEMANDA  = DATA_DIR / "rutaviva_demanda.csv"


# =============================================================================
# FESTIVOS COLOMBIANOS 2023-2025
# =============================================================================
# Replicamos la lista del notebook de generación + extensión a 2025
# para poder predecir el "futuro" a partir del último día del dataset.

FESTIVOS_RAW = [
    # 2023
    '2023-01-01', '2023-01-09', '2023-03-20', '2023-04-06', '2023-04-07',
    '2023-05-01', '2023-05-22', '2023-06-12', '2023-06-19', '2023-07-03',
    '2023-07-20', '2023-08-07', '2023-08-21', '2023-10-16', '2023-11-06',
    '2023-11-13', '2023-12-08', '2023-12-25',
    # 2024
    '2024-01-01', '2024-01-08', '2024-03-25', '2024-03-28', '2024-03-29',
    '2024-05-01', '2024-05-13', '2024-06-03', '2024-06-10', '2024-07-01',
    '2024-07-20', '2024-08-07', '2024-08-19', '2024-10-14', '2024-11-04',
    '2024-11-11', '2024-12-08', '2024-12-25',
    # 2025 (para modo predicción)
    '2025-01-01', '2025-01-06', '2025-03-24', '2025-04-17', '2025-04-18',
    '2025-05-01', '2025-06-02', '2025-06-23', '2025-06-30', '2025-08-07',
    '2025-08-18', '2025-10-13', '2025-11-03', '2025-11-17', '2025-12-08',
    '2025-12-25',
]
FESTIVOS_SET = set(pd.to_datetime(FESTIVOS_RAW).date)


# =============================================================================
# DEFINICIÓN DEL MODELO
# =============================================================================
# La estructura debe coincidir con la usada al entrenar el modelo.

class LSTMConEmbedding(nn.Module):
    """LSTM global con embedding de ruta para predicción multi-step directa.

    Entradas:
      - x_seq : (batch, ventana, n_features_temporales)
      - x_id  : (batch,) entero de ruta

    Salida:
      - (batch, horizonte)
    """

    def __init__(self, n_features_temporales, num_rutas, embedding_dim,
                 hidden_size=64, horizonte=30, num_layers=1, dropout=0.0):
        super().__init__()

        self.embedding_ruta = nn.Embedding(
            num_embeddings=num_rutas, embedding_dim=embedding_dim
        )

        self.lstm = nn.LSTM(
            input_size=n_features_temporales + embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, horizonte)

    def forward(self, x_seq, x_id):
        batch_size, ventana, _ = x_seq.shape
        emb = self.embedding_ruta(x_id)
        emb_repetido = emb.unsqueeze(1).expand(-1, ventana, -1)
        x_completo = torch.cat([x_seq, emb_repetido], dim=2)
        salida_lstm, (h_n, c_n) = self.lstm(x_completo)
        ultimo_estado = h_n[-1]
        prediccion = self.fc(ultimo_estado)
        return prediccion


# =============================================================================
# CARGA DE METADATOS Y MODELO (cacheada)
# =============================================================================

@st.cache_resource
def cargar_metadata() -> dict:
    """Carga el JSON con hiperparámetros y parámetros de normalización."""
    with open(PATH_METADATA, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def cargar_modelo() -> LSTMConEmbedding:
    """Carga el modelo LSTM entrenado en modo evaluación.

    Lee los hiperparámetros del JSON y reconstruye el modelo con las mismas
    dimensiones, luego carga los pesos. Las dimensiones deben coincidir
    exactamente con las del .pth o load_state_dict() fallará.

    Nota sobre n_features_temporales:
      El JSON lista 5 features ('dow_sin, dow_cos, mes_sin, mes_cos, es_festivo')
      pero el modelo se entrenó con 6 (las 5 anteriores + el pasajero normalizado
      del día, como feature auto-regresiva). Esta convención sigue el notebook
      de entrenamiento y se respeta aquí.
    """
    metadata = cargar_metadata()

    n_features_temporales = len(metadata["features_temporales"]) + 1  # +1 = pasajeros normalizado

    modelo = LSTMConEmbedding(
        n_features_temporales=n_features_temporales,
        num_rutas=metadata["num_rutas"],
        embedding_dim=metadata["embedding_dim"],
        hidden_size=metadata["hidden_size"],
        horizonte=metadata["horizonte_salida"],
    )

    state_dict = torch.load(PATH_MODEL, map_location="cpu", weights_only=True)
    modelo.load_state_dict(state_dict)
    modelo.eval()
    return modelo


# =============================================================================
# CARGA DE DATOS
# =============================================================================

@st.cache_data
def cargar_demanda() -> pd.DataFrame:
    """Carga el dataset de demanda con la fecha parseada."""
    df = pd.read_csv(PATH_DEMANDA, parse_dates=["fecha"])
    return df.sort_values(["ruta", "fecha"]).reset_index(drop=True)


# =============================================================================
# PREPARACIÓN DE FEATURES
# =============================================================================

def construir_features_temporales(fechas: pd.DatetimeIndex) -> np.ndarray:
    """Construye la matriz de features temporales para un conjunto de fechas.

    Parameters
    ----------
    fechas : pd.DatetimeIndex
        Las fechas para las cuales calcular las features.

    Returns
    -------
    np.ndarray
        Array de shape (n_fechas, 5) con las columnas:
        [dow_sin, dow_cos, mes_sin, mes_cos, es_festivo]
        en el orden definido por el JSON.
    """
    dow = fechas.dayofweek.values  # 0=lunes, 6=domingo
    mes = fechas.month.values

    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    mes_sin = np.sin(2 * np.pi * mes / 12)
    mes_cos = np.cos(2 * np.pi * mes / 12)

    es_festivo = np.array(
        [1.0 if f.date() in FESTIVOS_SET else 0.0 for f in fechas]
    )

    return np.column_stack([dow_sin, dow_cos, mes_sin, mes_cos, es_festivo])


def normalizar(valores: np.ndarray, ruta: str) -> np.ndarray:
    """Aplica la normalización Min-Max por ruta usando los parámetros del JSON."""
    metadata = cargar_metadata()
    y_min, y_max = metadata["parametros_norm"][ruta]
    return (valores - y_min) / (y_max - y_min)


def desnormalizar(valores: np.ndarray, ruta: str) -> np.ndarray:
    """Inversa de la normalización Min-Max para una ruta."""
    metadata = cargar_metadata()
    y_min, y_max = metadata["parametros_norm"][ruta]
    return valores * (y_max - y_min) + y_min


# =============================================================================
# INFERENCIA
# =============================================================================

def preparar_secuencia_entrada(df_ruta: pd.DataFrame, ruta: str,
                                ventana: int) -> tuple[np.ndarray, np.ndarray]:
    """Prepara la secuencia de entrada para el LSTM a partir de los últimos
    `ventana` días de la ruta.

    Returns
    -------
    x_seq : np.ndarray de shape (ventana, 6)
        Las 5 features temporales + el pasajero normalizado.
    fechas : pd.DatetimeIndex
        Las fechas correspondientes (útiles para el plot).
    """
    df_ventana = df_ruta.tail(ventana)
    fechas = pd.DatetimeIndex(df_ventana["fecha"].values)

    features_temp = construir_features_temporales(fechas)         # (ventana, 5)
    pasajeros_norm = normalizar(df_ventana["pasajeros"].values, ruta)  # (ventana,)

    # El orden de las features debe coincidir con el entrenamiento.
    # En el notebook: valores_entrada = df[[objetivo] + features]
    # El pasajero normalizado va primero, luego las 5 features temporales.
    x_seq = np.column_stack([pasajeros_norm, features_temp])       # (ventana, 6)
    return x_seq, fechas


def predecir(ruta: str, modo: str = "evaluacion") -> dict:
    """Genera una predicción a 30 días para la ruta dada.

    Parameters
    ----------
    ruta : str
        Nombre de la ruta (ej. 'Medellin-Bogota').
    modo : str
        - 'evaluacion'  : predice noviembre 2024 a partir de la ventana [sep-oct 2024].
                          Permite comparar con los valores reales del dataset.
        - 'prediccion'  : predice los 30 días que siguen al final del dataset (futuro).
                          Sin comparación con valores reales.

    Returns
    -------
    dict con las claves:
        - 'fechas_historial' : DatetimeIndex de las fechas de entrada
        - 'valores_historial': array de pasajeros del historial
        - 'fechas_prediccion': DatetimeIndex de las 30 fechas predichas
        - 'valores_prediccion': array de pasajeros predichos (desnormalizado)
        - 'valores_reales'   : array de pasajeros reales (solo en modo evaluación, sino None)
    """
    metadata = cargar_metadata()
    modelo = cargar_modelo()
    df = cargar_demanda()

    ventana = metadata["ventana_entrada"]
    horizonte = metadata["horizonte_salida"]
    ruta_id = metadata["ruta_a_id"][ruta]
    fecha_corte = pd.to_datetime(metadata["fecha_corte"])

    df_ruta = df[df["ruta"] == ruta].sort_values("fecha").reset_index(drop=True)

    if modo == "evaluacion":
        # Entrada = los `ventana` días antes de fecha_corte.
        # Según la convención del entrenamiento, la salida empieza el día
        # SIGUIENTE al último día de entrada. Derivamos las fechas de
        # predicción a partir de ese último día (NO de fecha_corte) para
        # garantizar el alineamiento exacto entre predicción y valores reales.
        df_train = df_ruta[df_ruta["fecha"] < fecha_corte]
        df_input = df_train.tail(ventana)

        fecha_ultima_input = pd.to_datetime(df_input["fecha"].iloc[-1])
        fechas_pred = pd.date_range(
            start=fecha_ultima_input + pd.Timedelta(days=1),
            periods=horizonte,
            freq="D",
        )

        # Valores reales: recuperados sobre EXACTAMENTE las mismas fechas,
        # en el mismo orden, para evitar cualquier desfase.
        df_real = df_ruta.set_index("fecha").reindex(fechas_pred)
        valores_reales = (
            df_real["pasajeros"].values if df_real["pasajeros"].notna().any() else None
        )

    elif modo == "prediccion":
        # Predecir los 30 días que siguen al último día del dataset
        df_input = df_ruta.tail(ventana)
        fecha_ultima = pd.to_datetime(df_input["fecha"].iloc[-1])
        fechas_pred = pd.date_range(
            start=fecha_ultima + pd.Timedelta(days=1),
            periods=horizonte,
            freq="D",
        )
        valores_reales = None

    else:
        raise ValueError(f"Modo desconocido: {modo}")

    # Preparar la secuencia de entrada
    fechas_input = pd.DatetimeIndex(df_input["fecha"].values)
    features_temp = construir_features_temporales(fechas_input)
    pasajeros_norm = normalizar(df_input["pasajeros"].values, ruta)
    # Mismo orden del entrenamiento: pasajero primero, luego features temporales.
    x_seq = np.column_stack([pasajeros_norm, features_temp]).astype(np.float32)

    # Inferencia
    x_seq_t = torch.tensor(x_seq).unsqueeze(0)                 # (1, ventana, 6)
    x_id_t  = torch.tensor([ruta_id], dtype=torch.long)        # (1,)

    with torch.no_grad():
        pred_norm = modelo(x_seq_t, x_id_t).numpy().flatten()  # (horizonte,)

    # Desnormalización + clip a ≥0 (no puede haber pasajeros negativos)
    pred = np.clip(desnormalizar(pred_norm, ruta), 0, None)

    return {
        "fechas_historial":   fechas_input,
        "valores_historial":  df_input["pasajeros"].values,
        "fechas_prediccion":  fechas_pred,
        "valores_prediccion": pred,
        "valores_reales":     valores_reales,
    }


# =============================================================================
# PCA DE LOS EMBEDDINGS DE RUTA
# =============================================================================

@st.cache_data
def calcular_pca_embeddings_rutas() -> pd.DataFrame:
    """Calcula la proyección PCA 2D de los embeddings de ruta aprendidos.

    Aquí se proyectan 6 rutas de 4D a 2D para facilitar la interpretación
    visual de los embeddings aprendidos.

    Returns
    -------
    pd.DataFrame con columnas: Ruta, PC1, PC2.
    Atributo `varianza_explicada` con el ratio de varianza de PC1 y PC2.
    """
    modelo = cargar_modelo()
    metadata = cargar_metadata()

    emb = modelo.embedding_ruta.weight.detach().numpy()  # (6, 4)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(emb)

    df_pca = pd.DataFrame({
        "Ruta": metadata["rutas"],
        "PC1":  coords[:, 0],
        "PC2":  coords[:, 1],
    })
    df_pca.attrs["varianza_explicada"] = pca.explained_variance_ratio_

    return df_pca
