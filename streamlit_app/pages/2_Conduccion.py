"""
Módulo 2 — Clasificación de Conducción Distraída
=================================================

Página que integra el modelo CNN (ResNet18 con transfer learning)
entrenado por Juan José sobre el Multi-Class Driver Behavior Image Dataset.

Atendido por la persona "Diana" del Design Thinking.
"""

import streamlit as st
from PIL import Image

from utils.style import aplicar_estilo, COLORS
from utils import modulo2


# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Conducción Distraída · RutaViva",
    layout="wide",
)
aplicar_estilo()


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
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
# SECCIÓN 1 — DESCRIPCIÓN DEL MODELO
# =============================================================================

st.markdown("### 1 · El modelo")

col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    st.markdown(
        f"""
        <div style="background: white; padding: 1.5rem; border-radius: 14px;
                    border-left: 5px solid {COLORS['turquesa']};
                    box-shadow: 0 4px 16px rgba(0,0,0,0.05);">
            <h4 style="margin: 0.3rem 0;">Arquitectura</h4>
            <div style="color: {COLORS['texto_soft']}; font-size: 0.9rem;">
                Transfer learning con <strong>ResNet18</strong> preentrenado
                en ImageNet, con la capa final reemplazada por una
                <code>Linear(512, 5)</code>.
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
            <h4 style="margin: 0.3rem 0;">Dataset</h4>
            <div style="color: {COLORS['texto_soft']}; font-size: 0.9rem;">
                <strong>Multi-Class Driver Behavior</strong> (Kaggle):
                5 clases — <em>safe_driving, texting_phone, talking_phone,
                turning, other_activities</em>.
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
            <h4 style="margin: 0.3rem 0;">Entrenamiento</h4>
            <div style="color: {COLORS['texto_soft']}; font-size: 0.9rem;">
                Backbone congelado, optimización Adam sobre la cabeza,
                data augmentation (flip, rotación, color jitter) y
                split 80/20.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("&nbsp;")


# =============================================================================
# SECCIÓN 2 — INFERENCIA EN VIVO
# =============================================================================

st.markdown("### 2 · Probar el clasificador")

modelo_listo = modulo2.modelo_disponible()

if not modelo_listo:
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['amarillo']}, {COLORS['arq_mochilero']});
                    color: white; padding: 1.2rem 1.5rem; border-radius: 14px;
                    margin: 0.5rem 0 1.5rem 0;
                    box-shadow: 0 6px 24px rgba(244, 162, 97, 0.25);">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 1.6rem;">⏳</div>
                <div>
                    <strong>Pesos del modelo no disponibles</strong><br>
                    <span style="font-size: 0.9rem; opacity: 0.95;">
                        El archivo <code>modelo_cnn.pth</code> aún no está en
                        <code>streamlit_app/models/</code>. La inferencia en vivo
                        se activará automáticamente cuando se añada el archivo;
                        mientras tanto puedes consultar los resultados de
                        entrenamiento más abajo.
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "Sube una imagen de un conductor. El modelo devuelve la categoría más probable "
    "junto a las dos siguientes alternativas para evaluar su confianza."
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
    if archivo is None:
        st.markdown(
            f"""
            <div style="background: {COLORS['crema_dark']}; padding: 3rem 2rem;
                        border-radius: 16px; text-align: center;">
                <p style="color: {COLORS['texto_soft']}; margin-top: 1rem;">
                    Sube una imagen para previsualizarla aquí.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif not modelo_listo:
        st.markdown(
            f"""
            <div style="background: {COLORS['crema_dark']}; padding: 2rem;
                        border-radius: 16px; text-align: center;
                        border: 2px dashed {COLORS['arq_mochilero']};">
                <h4 style="margin: 0.5rem 0;">Predicción no disponible</h4>
                <p style="color: {COLORS['texto_soft']}; margin: 0;">
                    Falta el archivo de pesos <code>modelo_cnn.pth</code>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        try:
            top = modulo2.predecir(imagen, top_k=3)
        except Exception as e:
            st.error(f"Error durante la inferencia: {e}")
            top = None

        if top:
            clase_top, prob_top = top[0]
            nombre, _ = modulo2.CLASES_DISPLAY[clase_top]

            # Color de fondo según si es safe o distraído
            color_bg = (
                COLORS["verde"] if clase_top == "safe_driving" else COLORS["coral"]
            )

            st.markdown(
                f"""
                <div style="background: {color_bg}; color: white;
                            padding: 1.5rem; border-radius: 16px;
                            text-align: center;
                            box-shadow: 0 6px 24px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0.3rem 0; color: white;">{nombre}</h3>
                    <p style="margin: 0; font-size: 1.4rem; font-weight: 600;">
                        {prob_top*100:.1f}% de confianza
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Alternativas consideradas**")
            for clase, prob in top[1:]:
                nombre_alt, _ = modulo2.CLASES_DISPLAY[clase]
                st.markdown(f"**{nombre_alt}** — {prob*100:.1f}%")
                st.progress(float(prob))


# =============================================================================
# SECCIÓN 3 — RESULTADOS DE ENTRENAMIENTO
# =============================================================================

st.markdown("&nbsp;")
st.markdown("### 3 · Resultados del entrenamiento")
st.caption(
    "Visualizaciones generadas durante la evaluación del modelo sobre el "
    "conjunto de validación (20% del dataset, ~1 456 imágenes)."
)

tab_cm, tab_ok, tab_err = st.tabs([
    "Matriz de confusión",
    "Predicciones correctas",
    "Predicciones erróneas",
])

with tab_cm:
    st.image(
        str(modulo2.PATH_CONFUSION),
        caption="Matriz de confusión — eje vertical: etiqueta real, eje horizontal: predicción del modelo.",
        use_container_width=True,
    )
    st.markdown(
        f"""
        <div style="background: {COLORS['crema_dark']}; padding: 1rem 1.2rem;
                    border-radius: 12px; border-left: 4px solid {COLORS['turquesa']};
                    font-size: 0.92rem;">
            <strong>Lectura rápida</strong> · El modelo distingue muy bien
            <em>safe_driving</em> y <em>texting_phone</em>. La clase
            <em>other_activities</em> es la más difícil — sus errores se reparten
            sobre todo hacia <em>safe_driving</em> y <em>talking_phone</em>,
            lo cual sugiere que esa categoría es heterogénea y podría
            beneficiarse de subdivisiones más finas.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_ok:
    st.image(
        str(modulo2.PATH_PRED_OK),
        caption="Muestras donde el modelo acertó la categoría.",
        use_container_width=True,
    )

with tab_err:
    st.image(
        str(modulo2.PATH_PRED_ERROR),
        caption="Muestras donde el modelo se equivocó — útil para identificar patrones de error.",
        use_container_width=True,
    )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("&nbsp;")
st.markdown("---")
st.caption(
    "Módulo 2 a cargo de Juan José · Stack: PyTorch + torchvision (ResNet18) · "
    "Entrenamiento sobre Multi-Class Driver Behavior Image Dataset (Kaggle)."
)
