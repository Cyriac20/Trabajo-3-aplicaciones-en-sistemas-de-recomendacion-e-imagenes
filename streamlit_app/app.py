"""
RutaViva — Sistema Inteligente Integrado
=========================================

Página de inicio de la aplicación.

Proyecto académico — Redes Neuronales Artificiales
Equipo: Cyriac y Juan José
"""

import streamlit as st
from utils.style import aplicar_estilo, COLORS


# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="RutaViva · Sistema Inteligente",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

aplicar_estilo()


# =============================================================================
# SIDEBAR — INFORMACIÓN DEL PROYECTO
# =============================================================================

with st.sidebar:
    st.markdown("### 🚌 RutaViva")
    st.markdown(
        "**Sistema Inteligente Integrado**  \n"
        "para una PME de transporte intermunicipal colombiana."
    )
    st.markdown("---")
    st.markdown(
        "**Curso:** Redes Neuronales Artificiales  \n"
        "**Equipo:** Cyriac y Juan José"
    )
    st.markdown("---")
    st.caption(
        "Navega por los módulos del proyecto usando el menú superior."
    )


# =============================================================================
# HERO PRINCIPAL
# =============================================================================

st.markdown(
    """
    <div class="rutaviva-hero">
        <h1>RutaViva</h1>
        <p>
            Tres soluciones de deep learning al servicio de una PME de transporte
            intermunicipal colombiana: anticipar la demanda, proteger a los pasajeros
            y personalizar la experiencia de viaje.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# CONTEXTO DE LA EMPRESA
# =============================================================================

st.markdown("### 📍 El contexto")

col_ctx_1, col_ctx_2 = st.columns([2, 1])

with col_ctx_1:
    st.markdown(
        """
        **RutaViva** es una PME colombiana de transporte terrestre de pasajeros
        que opera con una flota mixta de buses, minibuses y vans entre Medellín,
        el Eje Cafetero, Bogotá, Cartagena y destinos turísticos antioqueños.

        Como toda PME del sector, RutaViva enfrenta tres desafíos críticos que
        este sistema busca resolver mediante deep learning :
        """
    )

with col_ctx_2:
    st.markdown(
        f"""
        <div style="background: {COLORS['crema_dark']}; padding: 1.2rem;
                    border-radius: 12px; text-align: center;">
            <div style="font-size: 2rem; font-weight: 800;
                        color: {COLORS['coral']}; font-family: 'Fraunces', serif;">6</div>
            <div style="color: {COLORS['texto_soft']}; font-size: 0.9rem;">rutas operadas</div>
            <div style="font-size: 2rem; font-weight: 800;
                        color: {COLORS['turquesa']}; font-family: 'Fraunces', serif;
                        margin-top: 0.5rem;">30</div>
            <div style="color: {COLORS['texto_soft']}; font-size: 0.9rem;">destinos turísticos</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("&nbsp;")


# =============================================================================
# LAS 3 PERSONAS (DESIGN THINKING)
# =============================================================================

st.markdown("### 👥 Tres personas, tres necesidades")
st.markdown(
    "Identificadas durante la fase **Empathize** del Design Thinking, "
    "estas tres personas guían cada decisión técnica del proyecto."
)

st.markdown("&nbsp;")

col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    st.markdown(
        f"""
        <div class="persona-card" style="--accent: {COLORS['verde']};">
            <div style="font-size: 2.5rem;">👨‍💼</div>
            <h3>Carlos</h3>
            <div class="role">Gerente de Operaciones · 45 años</div>
            <div class="quote">
                « Planifico la flota con hojas de Excel y mi experiencia.
                Sufro por exceso de capacidad en temporadas bajas y por
                falta de vehículos en picos imprevistos. »
            </div>
            <div style="margin-top: 1rem; padding-top: 1rem;
                        border-top: 1px solid #eee; font-size: 0.85rem;
                        color: {COLORS['texto_soft']};">
                → Módulo 1 : Predicción de demanda
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_p2:
    st.markdown(
        f"""
        <div class="persona-card" style="--accent: {COLORS['turquesa']};">
            <div style="font-size: 2.5rem;">👩‍💼</div>
            <h3>Diana</h3>
            <div class="role">Jefa de Seguridad Vial · 38 años</div>
            <div class="quote">
                « Los reportes de conducción distraída llegan demasiado tarde,
                después de un incidente. No existe forma sistemática
                de detectar comportamientos de riesgo. »
            </div>
            <div style="margin-top: 1rem; padding-top: 1rem;
                        border-top: 1px solid #eee; font-size: 0.85rem;
                        color: {COLORS['texto_soft']};">
                → Módulo 2 : Detección de distracción
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_p3:
    st.markdown(
        f"""
        <div class="persona-card" style="--accent: {COLORS['coral']};">
            <div style="font-size: 2.5rem;">🧳</div>
            <h3>Andrés</h3>
            <div class="role">Cliente frecuente · 28 años</div>
            <div class="quote">
                « Pierdo tiempo navegando destinos en la plataforma
                y no descubro rutas nuevas que podrían interesarme.
                Quiero sugerencias personalizadas. »
            </div>
            <div style="margin-top: 1rem; padding-top: 1rem;
                        border-top: 1px solid #eee; font-size: 0.85rem;
                        color: {COLORS['texto_soft']};">
                → Módulo 3 : Recomendaciones personalizadas
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("&nbsp;")
st.markdown("&nbsp;")


# =============================================================================
# LOS 3 MÓDULOS
# =============================================================================

st.markdown("### 🧠 Los tres módulos de deep learning")

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.markdown(
        f"""
        <div class="modulo-card">
            <div class="icono">📈</div>
            <h3>Módulo 1</h3>
            <div style="font-weight: 600; color: {COLORS['coral']};
                        margin-bottom: 0.5rem;">Predicción de demanda</div>
            <div class="descripcion">
                Red LSTM con embeddings de ruta que predice la demanda
                de pasajeros a 30 días sobre las 6 rutas.
            </div>
            <div style="margin-top: 1rem;">
                <span class="badge badge-ok">Operacional</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m2:
    st.markdown(
        f"""
        <div class="modulo-card">
            <div class="icono">🛣️</div>
            <h3>Módulo 2</h3>
            <div style="font-weight: 600; color: {COLORS['coral']};
                        margin-bottom: 0.5rem;">Conducción distraída</div>
            <div class="descripcion">
                CNN con transfer learning (ResNet18/MobileNetV2) que clasifica
                imágenes de conductores en categorías de comportamiento.
            </div>
            <div style="margin-top: 1rem;">
                <span class="badge badge-pending">En preparación</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m3:
    st.markdown(
        f"""
        <div class="modulo-card">
            <div class="icono">🧭</div>
            <h3>Módulo 3</h3>
            <div style="font-weight: 600; color: {COLORS['coral']};
                        margin-bottom: 0.5rem;">Recomendaciones</div>
            <div class="descripcion">
                Filtrado colaborativo por factorización matricial
                que sugiere destinos según el perfil del usuario.
            </div>
            <div style="margin-top: 1rem;">
                <span class="badge badge-ok">Operacional</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("&nbsp;")
st.info(
    "👈 **Usa el menú lateral** para navegar entre los módulos y explorar "
    "las predicciones, clasificaciones y recomendaciones generadas por los modelos."
)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("&nbsp;")
st.markdown("---")
st.caption(
    "Proyecto académico · Curso de Redes Neuronales Artificiales · "
    "Equipo: Cyriac y Juan José · 2025"
)
