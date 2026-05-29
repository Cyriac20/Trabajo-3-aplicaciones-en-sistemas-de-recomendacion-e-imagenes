"""
Funciones de carga y inferencia para el Módulo 3 (Sistema de Recomendación).

Centraliza:
  - La definición de la clase MatrixFactorization (idéntica al notebook de entrenamiento)
  - La carga del modelo entrenado (.pth) y de sus metadatos (.json)
  - La carga del dataset (destinos, usuarios, visitas, reviews, ground truth)
  - La generación de recomendaciones top-K para un usuario
  - El cálculo de la proyección PCA de los embeddings de usuario

Todas las funciones costosas usan @st.cache_resource o @st.cache_data
para no recargar a cada interacción del usuario.
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
# Rutas relativas a la raíz del repo (donde está app.py)

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data"

PATH_MODEL    = MODELS_DIR / "modulo3_mf.pth"
PATH_METADATA = MODELS_DIR / "metadata_modulo3.json"

PATH_DESTINATIONS = DATA_DIR / "destinations.csv"
PATH_USERS        = DATA_DIR / "users.csv"
PATH_VISITS       = DATA_DIR / "visits.csv"
PATH_REVIEWS      = DATA_DIR / "reviews.csv"
PATH_GROUND_TRUTH = DATA_DIR / "ground_truth_arquetipos.csv"


# =============================================================================
# DEFINICIÓN DEL MODELO
# =============================================================================
# La estructura debe coincidir con la usada durante el entrenamiento,
# de lo contrario load_state_dict() fallará o producirá pesos incorrectos.

class MatrixFactorization(nn.Module):
    """Filtrado colaborativo por factorización matricial con biases.

    Referencia: Koren, Y., Bell, R., & Volinsky, C. (2009).
    Matrix factorization techniques for recommender systems. IEEE Computer, 42(8).
    """
    def __init__(self, n_users, n_dests, embedding_dim=8, rating_medio=0.0):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.dest_emb = nn.Embedding(n_dests, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.dest_bias = nn.Embedding(n_dests, 1)
        self.register_buffer('global_bias', torch.tensor(rating_medio, dtype=torch.float32))

        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.dest_emb.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.dest_bias.weight)

    def forward(self, u_idx, d_idx):
        u = self.user_emb(u_idx)
        d = self.dest_emb(d_idx)
        bu = self.user_bias(u_idx).squeeze(1)
        bd = self.dest_bias(d_idx).squeeze(1)
        dot = (u * d).sum(dim=1)
        return self.global_bias + bu + bd + dot


# =============================================================================
# CARGA DE METADATOS Y MODELO (cacheada)
# =============================================================================

@st.cache_resource
def cargar_metadata() -> dict:
    """Carga el JSON con hiperparámetros, métricas y mappings."""
    with open(PATH_METADATA, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def cargar_modelo() -> MatrixFactorization:
    """Carga el modelo MF entrenado en modo evaluación.

    El modelo se construye con los hiperparámetros del JSON de metadatos
    y se le cargan los pesos del archivo .pth. Se devuelve en modo eval()
    para desactivar dropout/batchnorm (no relevantes aquí pero buena práctica).
    """
    metadata = cargar_metadata()
    hp = metadata["hiperparametros"]
    ds = metadata["dataset"]

    modelo = MatrixFactorization(
        n_users=ds["n_users"],
        n_dests=ds["n_dests"],
        embedding_dim=hp["embedding_dim"],
        rating_medio=ds["rating_medio_train"],
    )
    # map_location='cpu' garantiza que funcione en Streamlit Cloud (sin GPU)
    state_dict = torch.load(PATH_MODEL, map_location="cpu", weights_only=True)
    modelo.load_state_dict(state_dict)
    modelo.eval()
    return modelo


# =============================================================================
# CARGA DE DATOS (cacheada)
# =============================================================================

@st.cache_data
def cargar_destinations() -> pd.DataFrame:
    """30 destinos turísticos con sus categorías."""
    df = pd.read_csv(PATH_DESTINATIONS)
    # Convertir la columna 'Categorias' en lista para facilitar el uso
    df["Categorias"] = df["Categorias"].str.split(",")
    return df


@st.cache_data
def cargar_users() -> pd.DataFrame:
    """900 usuarios con atributos demográficos."""
    return pd.read_csv(PATH_USERS)


@st.cache_data
def cargar_visits() -> pd.DataFrame:
    """Visitas (señal implícita)."""
    df = pd.read_csv(PATH_VISITS, parse_dates=["VisitDate"])
    return df


@st.cache_data
def cargar_reviews() -> pd.DataFrame:
    """Reviews con rating 1-5 (señal explícita)."""
    df = pd.read_csv(PATH_REVIEWS, parse_dates=["VisitDate"])
    return df


@st.cache_data
def cargar_ground_truth() -> pd.DataFrame:
    """Arquetipo verdadero de cada usuario (oculto al modelo)."""
    return pd.read_csv(PATH_GROUND_TRUTH)


# =============================================================================
# INFERENCIA: RECOMENDACIONES TOP-K
# =============================================================================

def obtener_user_idx(user_id: int) -> int:
    """Convierte un UserID público en el índice interno del embedding.

    El mapping es identidad (UserID-1) pero pasamos por el JSON
    para mantener la robustez (si el mapping cambiara en el futuro).
    """
    metadata = cargar_metadata()
    return metadata["mapeo_user_id_to_idx"][str(user_id)]


def obtener_dest_idx(dest_id: int) -> int:
    """Convierte un DestinationID público en el índice interno del embedding."""
    metadata = cargar_metadata()
    return metadata["mapeo_dest_id_to_idx"][str(dest_id)]


def predecir_ratings(user_id: int) -> np.ndarray:
    """Predice el rating estimado del usuario para los 30 destinos.

    Devuelve un array de shape (30,) con los ratings predichos,
    en el orden de DestinationID (1, 2, ..., 30).
    """
    modelo = cargar_modelo()
    metadata = cargar_metadata()
    n_dests = metadata["dataset"]["n_dests"]

    u_idx = obtener_user_idx(user_id)

    # Tensores: el usuario se repite, los destinos van de 0 a n_dests-1
    u_tensor = torch.full((n_dests,), u_idx, dtype=torch.long)
    d_tensor = torch.arange(n_dests, dtype=torch.long)

    with torch.no_grad():
        ratings = modelo(u_tensor, d_tensor).numpy()

    return ratings


def top_k_recomendaciones(user_id: int, k: int = 10,
                          excluir_visitados: bool = True) -> pd.DataFrame:
    """Genera las top-K recomendaciones para un usuario.

    Por defecto, excluye los destinos que el usuario ya ha visitado:
    no tiene sentido recomendarle algo que ya conoce.

    Devuelve un DataFrame con columnas:
        DestinationID, Name, Ruta, Departamento, Categorias, RatingPredicho
    ordenado por RatingPredicho descendente.
    """
    destinations = cargar_destinations()
    visits = cargar_visits()

    ratings = predecir_ratings(user_id)

    # Construir el DataFrame de candidatos
    candidatos = destinations.copy()
    candidatos["RatingPredicho"] = ratings

    # Excluir los destinos ya visitados
    if excluir_visitados:
        visitados = set(visits.loc[visits["UserID"] == user_id, "DestinationID"])
        candidatos = candidatos[~candidatos["DestinationID"].isin(visitados)]

    return candidatos.nlargest(k, "RatingPredicho").reset_index(drop=True)


# =============================================================================
# HISTORIAL DEL USUARIO
# =============================================================================

def obtener_historial(user_id: int) -> pd.DataFrame:
    """Recupera el historial de visitas del usuario, enriquecido con ratings y destinos.

    Devuelve un DataFrame con: Name, Ruta, Categorias, VisitDate, Rating (NaN si no review).
    Ordenado por fecha descendente.
    """
    visits = cargar_visits()
    reviews = cargar_reviews()
    destinations = cargar_destinations()

    # Visitas del usuario
    h = visits[visits["UserID"] == user_id].copy()

    # Join con reviews para añadir el rating (si existe)
    h = h.merge(
        reviews[["UserID", "DestinationID", "VisitDate", "Rating"]],
        on=["UserID", "DestinationID", "VisitDate"],
        how="left",
    )

    # Join con destinations para obtener el nombre y la ruta
    h = h.merge(
        destinations[["DestinationID", "Name", "Ruta", "Departamento", "Categorias"]],
        on="DestinationID",
        how="left",
    )

    # Ordenar por fecha descendente
    return h.sort_values("VisitDate", ascending=False).reset_index(drop=True)


# =============================================================================
# PCA DE LOS EMBEDDINGS DE USUARIO (análisis creativo)
# =============================================================================

@st.cache_data
def calcular_pca_embeddings() -> pd.DataFrame:
    """Calcula la proyección PCA 2D de los embeddings de usuario aprendidos.

    Devuelve un DataFrame con columnas:
        UserID, PC1, PC2, Arquetipo

    El arquetipo proviene del ground truth, oculto al modelo durante el
    entrenamiento. Esta es la prueba creativa: ¿el modelo redescubrió
    la estructura latente del dataset?

    El resultado se cachea para no recalcular el PCA a cada interacción.
    """
    modelo = cargar_modelo()
    ground_truth = cargar_ground_truth()

    # Extraer la matriz de embeddings de usuario
    user_emb = modelo.user_emb.weight.detach().numpy()  # (900, 8)

    # Proyección PCA 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(user_emb)

    # Construir el DataFrame
    df_pca = pd.DataFrame({
        "UserID":   np.arange(1, len(coords) + 1),  # mapping identidad
        "PC1":      coords[:, 0],
        "PC2":      coords[:, 1],
    })
    df_pca = df_pca.merge(ground_truth, on="UserID")

    # Atributo adicional: varianza explicada (útil para mostrar en la UI)
    df_pca.attrs["varianza_explicada"] = pca.explained_variance_ratio_

    return df_pca
