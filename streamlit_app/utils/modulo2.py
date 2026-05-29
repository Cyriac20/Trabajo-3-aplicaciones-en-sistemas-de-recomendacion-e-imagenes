"""
Funciones de carga y inferencia para el Módulo 2 (Clasificación de
Conducción Distraída).

Centraliza:
  - La definición del modelo (ResNet18 con cabeza reentrenada) idéntica
    a la usada en modulo2_clasificacion_imagenes/train.py
  - La carga del modelo entrenado (.pth)
  - Las transformaciones de preprocesamiento (val_transforms del entrenamiento)
  - La inferencia top-K sobre una imagen subida por el usuario

Funciones costosas usan @st.cache_resource para no recargar
a cada interacción.
"""

from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# =============================================================================
# RUTAS DE LOS ARTEFACTOS
# =============================================================================

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets" / "modulo2"

PATH_MODEL = MODELS_DIR / "modelo_cnn.pth"

PATH_CONFUSION   = ASSETS_DIR / "matriz_confusion.png"
PATH_PRED_OK     = ASSETS_DIR / "predicciones_correctas.png"
PATH_PRED_ERROR  = ASSETS_DIR / "predicciones_erroneas.png"


# =============================================================================
# CLASES DEL DATASET
# =============================================================================
# Orden alfabético devuelto por torchvision.datasets.ImageFolder durante
# el entrenamiento — debe coincidir EXACTAMENTE con el orden usado por Juan José
# (verificado contra los ejes de matriz_confusion.png).

CLASES = [
    "other_activities",
    "safe_driving",
    "talking_phone",
    "texting_phone",
    "turning",
]

CLASES_DISPLAY = {
    "safe_driving":     ("Conducción segura", ""),
    "texting_phone":    ("Escribiendo SMS", ""),
    "talking_phone":    ("Hablando por teléfono", ""),
    "turning":          ("Girándose", ""),
    "other_activities": ("Otras actividades", ""),
}


# =============================================================================
# DEFINICIÓN DEL MODELO
# =============================================================================
# Debe ser IDÉNTICA a la de modulo2_clasificacion_imagenes/model.py
# (ResNet18 con la capa fc reemplazada por una Linear(512, num_classes)).

def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


# =============================================================================
# TRANSFORMS DE INFERENCIA
# =============================================================================
# Idénticas a val_transforms en train.py:
#   Resize(224,224) + ToTensor + Normalize(media/std de ImageNet)

_INFER_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# =============================================================================
# CARGA DEL MODELO
# =============================================================================

def modelo_disponible() -> bool:
    """Indica si el archivo de pesos está presente en el repo."""
    return PATH_MODEL.exists()


@st.cache_resource(show_spinner="Cargando modelo CNN...")
def cargar_modelo():
    """Carga ResNet18 con los pesos entrenados por Juan José.

    Devuelve el modelo en modo eval, sobre CPU (Streamlit Cloud no tiene GPU).
    """
    model = build_model(num_classes=len(CLASES))
    state = torch.load(PATH_MODEL, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# =============================================================================
# INFERENCIA
# =============================================================================

def predecir(imagen_pil: Image.Image, top_k: int = 3):
    """Clasifica una imagen y devuelve las top_k clases con sus probabilidades.

    Parameters
    ----------
    imagen_pil : PIL.Image
        Imagen RGB cargada por el usuario.
    top_k : int
        Número de clases a devolver, ordenadas de mayor a menor probabilidad.

    Returns
    -------
    list[tuple[str, float]]
        Lista de (nombre_clase, probabilidad). Por ejemplo:
        [("texting_phone", 0.82), ("talking_phone", 0.11), ("safe_driving", 0.04)]
    """
    modelo = cargar_modelo()

    # Asegurar RGB (algunas imágenes PNG vienen en RGBA o en escala de grises)
    if imagen_pil.mode != "RGB":
        imagen_pil = imagen_pil.convert("RGB")

    tensor = _INFER_TRANSFORMS(imagen_pil).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = modelo(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # Ordenar índices por probabilidad descendente y quedarnos con top_k
    indices_orden = probs.argsort()[::-1][:top_k]
    return [(CLASES[i], float(probs[i])) for i in indices_orden]
