"""
Módulo 2 — Clasificación de Conducción Distraída
=================================================

Página *placeholder* en espera de que Juan José termine el entrenamiento
del modelo CNN. La UI está completa: cuando el modelo esté disponible,
solo hay que reemplazar la sección de inferencia.

Atendido por la persona "Diana" del Design Thinking.
"""

import streamlit as st
from PIL import Image

from utils.style import aplicar_estilo, COLORS


# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Conducción Distraída · RutaViva",
    page_icon="🛣️",
    layout="wide",
)
aplicar_estilo()


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
        <div style="font-size: 3rem;">🛣️</div>
        <div>
            <h1 style="margin: 0;">Conducción Distraída</h1>
            <p style="color: {COLORS['texto_soft']}; margin: 0; font-size: 1.05rem;">
                Detección automática de comportamientos distraídos en conductores —
                <em>al servicio de Diana, Jefa de Seguridad Vial</em>
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# BANNER DE ESTADO
# =============================================================================

st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, {COLORS['amarillo']}, {COLORS['arq_mochilero']});
                color: white; padding: 1.5rem 2rem; border-radius: 16px;
                margin: 1rem 0 2rem 0;
                box-shadow: 0 6px 24px rgba(244, 162, 97, 0.25);">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="font-size: 2rem;">⏳</div>
            <div>
                <strong style="font-size: 1.1rem;">Módulo en preparación</strong><br>
                <span style="font-size: 0.95rem; opacity: 0.95;">
                    El modelo de clasificación de imágenes está actualmente en fase
                    de entrenamiento. La interfaz mostrada a continuación está lista
                    para integrar el modelo en cuanto esté disponible.
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SECCIÓN 1 — DESCRIPCIÓN DEL MODELO PREVISTO
# =============================================================================

st.markdown("### 1 · El modelo previsto")

col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    st.markdown(
        f"""
        <div style="background: white; padding: 1.5rem; border-radius: 14px;
                    border-left: 5px solid {COLORS['turquesa']};
                    box-shadow: 0 4px 16px rgba(0,0,0,0.05);">
            <div style="font-size: 1.8rem;">🧠</div>
            <h4 style="margin: 0.3rem 0;">Arquitectura</h4>
            <div style="color: {COLORS['texto_soft']}; font-size: 0.9rem;">
                Transfer learning con <strong>ResNet18</strong> o
                <strong>MobileNetV2</strong>: congelado del backbone
                + reentrenamiento de la cabeza de clasificación.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_d2:
    st.markdown(
        f"""
        <div style="background: white; padding: 1.5rem; border-radius: 14px;
                    border-left: 5px solid {COLORS['coral']};
                    box-shadow: 0 4px 16px rgba(0,0,0,0.05);">
            <div style="font-size: 1.8rem;">📊</div>
            <h4 style="margin: 0.3rem 0;">Dataset</h4>
            <div style="color: {COLORS['texto_soft']}; font-size: 0.9rem;">
                <strong>Multi-class Driver Behavior</strong> (Kaggle):
                imágenes etiquetadas en varias categorías de
                comportamiento distraído.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_d3:
    st.markdown(
        f"""
        <div style="background: white; padding: 1.5rem; border-radius: 14px;
                    border-left: 5px solid {COLORS['verde']};
                    box-shadow: 0 4px 16px rgba(0,0,0,0.05);">
            <div style="font-size: 1.8rem;">🎯</div>
            <h4 style="margin: 0.3rem 0;">Métricas previstas</h4>
            <div style="color: {COLORS['texto_soft']}; font-size: 0.9rem;">
                Accuracy, F1-score, precision, recall y matriz
                de confusión sobre el conjunto de prueba.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("&nbsp;")


# =============================================================================
# SECCIÓN 2 — UI DE INFERENCIA (PLACEHOLDER)
# =============================================================================

st.markdown("### 2 · Probar el clasificador")
st.caption(
    "Sube una imagen de un conductor para clasificarla automáticamente. "
    "Esta interfaz está lista; conectaremos el modelo cuando el entrenamiento "
    "haya terminado."
)

col_up_1, col_up_2 = st.columns([1, 1])

with col_up_1:
    archivo = st.file_uploader(
        "Imagen del conductor",
        type=["jpg", "jpeg", "png"],
        help="Formatos soportados: JPG, PNG.",
    )

    if archivo is not None:
        imagen = Image.open(archivo)
        st.image(imagen, caption="Imagen cargada", use_container_width=True)

with col_up_2:
    if archivo is not None:
        st.markdown(
            f"""
            <div style="background: {COLORS['crema_dark']}; padding: 2rem;
                        border-radius: 16px; text-align: center;
                        border: 2px dashed {COLORS['arq_mochilero']};">
                <div style="font-size: 2rem;">🔒</div>
                <h4 style="margin: 0.5rem 0;">Predicción pendiente</h4>
                <p style="color: {COLORS['texto_soft']}; margin: 0;">
                    El modelo de clasificación aún no está disponible.<br>
                    Vuelve pronto para ver el resultado.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background: {COLORS['crema_dark']}; padding: 3rem 2rem;
                        border-radius: 16px; text-align: center;">
                <div style="font-size: 2.5rem;">👈</div>
                <p style="color: {COLORS['texto_soft']}; margin-top: 1rem;">
                    Sube una imagen para previsualizarla aquí.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("&nbsp;")
st.markdown("---")
st.caption(
    "Módulo 2 a cargo de Juan José · Stack previsto: PyTorch + torchvision · "
    "Conexión a la app al finalizar el entrenamiento."
)
