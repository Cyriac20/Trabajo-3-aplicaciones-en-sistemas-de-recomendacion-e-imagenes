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
    layout="wide",
    initial_sidebar_state="expanded",
)

aplicar_estilo()


# =============================================================================
# SIDEBAR — INFORMACIÓN DEL PROYECTO
# =============================================================================

with st.sidebar:
    st.markdown("### RutaViva")
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

st.markdown("### Contexto")

col_ctx_1, col_ctx_2 = st.columns([2, 1])

with col_ctx_1:
    st.markdown(
        """
        **RutaViva** es una PME colombiana de transporte terrestre de pasajeros
        que opera con una flota mixta de buses, minibuses y vans entre Medellín,
        el Eje Cafetero, Bogotá, Cartagena y destinos turísticos antioqueños.

        Como muchas empresas del sector, RutaViva enfrenta tres desafíos
        operativos que este sistema aborda mediante modelos de aprendizaje
        profundo:
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

st.markdown("### Personas y necesidades")
st.markdown(
    "Las personas definidas en la etapa de análisis ayudan a conectar cada "
    "módulo con una necesidad concreta del negocio."
)

st.markdown("&nbsp;")

col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    st.markdown(
        f"""
        <div class="persona-card" style="--accent: {COLORS['verde']};">
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
                Módulo 1: predicción de demanda
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_p2:
    st.markdown(
        f"""
        <div class="persona-card" style="--accent: {COLORS['turquesa']};">
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
                Módulo 2: detección de distracción
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_p3:
    st.markdown(
        f"""
        <div class="persona-card" style="--accent: {COLORS['coral']};">
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
                Módulo 3: recomendaciones personalizadas
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

st.markdown("### Módulos del sistema")

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.markdown(
        f"""
        <a href="/Demanda" target="_self" style="text-decoration: none; color: inherit;">
        <div class="modulo-card">
            <h3>Módulo 1</h3>
            <div style="font-weight: 600; color: {COLORS['coral']};
                        margin-bottom: 0.5rem;">Predicción de demanda</div>
            <div class="descripcion">
                Red LSTM con embeddings de ruta que predice la demanda
                de pasajeros a 30 días sobre las 6 rutas.
            </div>
            <div style="margin-top: 1rem;">
                <span class="badge badge-ok">Operacional</span>
                <span style="float:right; color: {COLORS['turquesa']}; font-weight: 600;">Abrir →</span>
            </div>
        </div>
        </a>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Demanda.py", label="Ir al Módulo 1", icon="📈")

with col_m2:
    st.markdown(
        f"""
        <a href="/Conduccion" target="_self" style="text-decoration: none; color: inherit;">
        <div class="modulo-card">
            <h3>Módulo 2</h3>
            <div style="font-weight: 600; color: {COLORS['coral']};
                        margin-bottom: 0.5rem;">Conducción distraída</div>
            <div class="descripcion">
                CNN con transfer learning (ResNet18/MobileNetV2) que clasifica
                imágenes de conductores en categorías de comportamiento.
            </div>
            <div style="margin-top: 1rem;">
                <span class="badge badge-ok">Operacional</span>
                <span style="float:right; color: {COLORS['turquesa']}; font-weight: 600;">Abrir →</span>
            </div>
        </div>
        </a>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/2_Conduccion.py", label="Ir al Módulo 2", icon="🛣️")

with col_m3:
    st.markdown(
        f"""
        <a href="/Recomendaciones" target="_self" style="text-decoration: none; color: inherit;">
        <div class="modulo-card">
            <h3>Módulo 3</h3>
            <div style="font-weight: 600; color: {COLORS['coral']};
                        margin-bottom: 0.5rem;">Recomendaciones</div>
            <div class="descripcion">
                Filtrado colaborativo por factorización matricial
                que sugiere destinos según el perfil del usuario.
            </div>
            <div style="margin-top: 1rem;">
                <span class="badge badge-ok">Operacional</span>
                <span style="float:right; color: {COLORS['turquesa']}; font-weight: 600;">Abrir →</span>
            </div>
        </div>
        </a>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/3_Recomendaciones.py", label="Ir al Módulo 3", icon="🧭")

st.markdown("&nbsp;")
st.info(
    "**Usa el menú lateral** para navegar entre los módulos y explorar "
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
