"""
Módulo 1 — Predicción de Demanda de Pasajeros
==============================================

Página de la app dedicada a la predicción de demanda. Permite:
  1. Seleccionar una de las 6 rutas RutaViva
  2. Elegir el modo: evaluación (vs realidad) o predicción del futuro
  3. Visualizar el historial reciente + la predicción a 30 días
  4. Explorar la estructura latente aprendida por el modelo (PCA de embeddings)

Atendido por la persona "Carlos" del Design Thinking.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.style import aplicar_estilo, COLORS
from utils.modulo1 import (
    cargar_metadata,
    cargar_demanda,
    predecir,
    calcular_pca_embeddings_rutas,
)


# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Predicción de Demanda · RutaViva",
    page_icon="📈",
    layout="wide",
)
aplicar_estilo()


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
        <div style="font-size: 3rem;">📈</div>
        <div>
            <h1 style="margin: 0;">Predicción de Demanda</h1>
            <p style="color: {COLORS['texto_soft']}; margin: 0; font-size: 1.05rem;">
                Anticipar la demanda de pasajeros a 30 días sobre las 6 rutas RutaViva —
                <em>al servicio de Carlos, Gerente de Operaciones</em>
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# CARGA DE METADATOS
# =============================================================================

metadata = cargar_metadata()
df_demanda = cargar_demanda()


# =============================================================================
# SECCIÓN 1 — CONTROLES
# =============================================================================

st.markdown("### 1 · Configurar la predicción")

col_ctrl_1, col_ctrl_2 = st.columns([1, 1])

with col_ctrl_1:
    rutas_ordenadas = sorted(metadata["rutas"])
    ruta = st.selectbox(
        "Ruta a analizar",
        rutas_ordenadas,
        index=rutas_ordenadas.index("Medellin-Cartagena"),  # default: la más estacional
        help="Cada ruta tiene su propio embedding aprendido y sus parámetros de normalización.",
    )

with col_ctrl_2:
    modo = st.radio(
        "Modo de predicción",
        options=["evaluacion", "prediccion"],
        format_func=lambda m: {
            "evaluacion": "🎯 Evaluación (predecir nov 2024 vs datos reales)",
            "prediccion": "🔮 Predicción (futuro a partir del fin de dataset)",
        }[m],
        horizontal=False,
    )


# =============================================================================
# SECCIÓN 2 — GRÁFICO PRINCIPAL
# =============================================================================

st.markdown("&nbsp;")
st.markdown("### 2 · Visualización de la predicción")

resultado = predecir(ruta=ruta, modo=modo)

# Construcción del gráfico Plotly
fig = go.Figure()

# Trace 1: historial (los 60 días de entrada)
fig.add_trace(
    go.Scatter(
        x=resultado["fechas_historial"],
        y=resultado["valores_historial"],
        mode="lines",
        name="Historial (entrada del modelo)",
        line=dict(color=COLORS["verde"], width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Pasajeros: %{y:.0f}<extra></extra>",
    )
)

# Trace 2: predicción
fig.add_trace(
    go.Scatter(
        x=resultado["fechas_prediccion"],
        y=resultado["valores_prediccion"],
        mode="lines+markers",
        name="Predicción LSTM",
        line=dict(color=COLORS["coral"], width=3),
        marker=dict(size=6),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Predicho: %{y:.0f}<extra></extra>",
    )
)

# Trace 3 (modo evaluación): valores reales
if modo == "evaluacion" and resultado["valores_reales"] is not None:
    fig.add_trace(
        go.Scatter(
            x=resultado["fechas_prediccion"][: len(resultado["valores_reales"])],
            y=resultado["valores_reales"],
            mode="lines+markers",
            name="Valores reales",
            line=dict(color=COLORS["turquesa"], width=2.5, dash="dot"),
            marker=dict(size=6, symbol="diamond"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Real: %{y:.0f}<extra></extra>",
        )
    )

# Línea vertical separando historial y predicción
fecha_separacion = resultado["fechas_prediccion"][0]
fig.add_vline(
    x=fecha_separacion,
    line=dict(color=COLORS["texto_soft"], width=1, dash="dash"),
    annotation_text="Inicio predicción",
    annotation_position="top",
    annotation_font=dict(size=11, color=COLORS["texto_soft"]),
)

fig.update_layout(
    plot_bgcolor=COLORS["crema"],
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color=COLORS["texto"]),
    height=480,
    xaxis=dict(
        title="Fecha",
        gridcolor="#E5E5E5",
        zerolinecolor="#D0D0D0",
    ),
    yaxis=dict(
        title="Pasajeros / día",
        gridcolor="#E5E5E5",
        zerolinecolor="#D0D0D0",
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        bgcolor="rgba(255,255,255,0.5)",
    ),
    margin=dict(l=20, r=20, t=50, b=20),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)


# Métricas resumen (solo en modo evaluación, calculadas en directo)
if modo == "evaluacion" and resultado["valores_reales"] is not None:
    reales = resultado["valores_reales"]
    pred = resultado["valores_prediccion"][: len(reales)]
    rmse = float(np.sqrt(np.mean((pred - reales) ** 2)))
    mae = float(np.mean(np.abs(pred - reales)))
    error_relativo = float(mae / np.mean(reales) * 100)

    col_met = st.columns(3)
    col_met[0].metric("RMSE", f"{rmse:.1f}", help="Raíz del error cuadrático medio")
    col_met[1].metric("MAE", f"{mae:.1f}", help="Error absoluto medio")
    col_met[2].metric(
        "Error relativo medio",
        f"{error_relativo:.1f} %",
        help="MAE dividido por el promedio de pasajeros reales",
    )

elif modo == "prediccion":
    fecha_inicio = resultado["fechas_prediccion"][0].strftime("%d de %B %Y")
    fecha_fin = resultado["fechas_prediccion"][-1].strftime("%d de %B %Y")
    media_pred = float(np.mean(resultado["valores_prediccion"]))

    st.markdown(
        f"""
        <div style="background: {COLORS['crema_dark']}; padding: 1rem 1.5rem;
                    border-radius: 12px; margin-top: 1rem;">
            🔮 <strong>Predicción operacional para Carlos.</strong>
            El modelo anticipa una demanda media de
            <strong>{media_pred:.0f} pasajeros/día</strong> sobre la ruta
            <em>{ruta}</em> entre el {fecha_inicio} y el {fecha_fin}.
            Esta predicción puede orientar la asignación de vehículos y personal.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("&nbsp;")


# =============================================================================
# SECCIÓN 3 — PCA DE LOS EMBEDDINGS DE RUTA
# =============================================================================

st.markdown("### 3 · ¿Qué aprendió el modelo? · Análisis PCA de los embeddings de ruta")

with st.expander("ℹ️ ¿Qué significa esta visualización?", expanded=False):
    st.markdown(
        """
        El modelo LSTM asocia a cada una de las **6 rutas** un vector latente de
        **4 dimensiones** que captura sus particularidades temporales.

        A diferencia de los hiperparámetros que nosotros definimos (`base`,
        `peso_fin_de_semana`, etc.), estos embeddings son **aprendidos por el modelo
        durante el entrenamiento**, optimizando únicamente la calidad de la predicción.

        El PCA proyecta estos 6 vectores en 2D. **Si rutas con perfiles similares
        aparecen cerca, significa que el modelo descubrió las regularidades del
        portafolio sin que se le hayan dicho explícitamente.**

        Este análisis hace eco al PCA de los embeddings de usuario del Módulo 3:
        ambos validan la calidad interpretativa de las representaciones latentes
        aprendidas.
        """
    )

df_pca = calcular_pca_embeddings_rutas()
varianza = df_pca.attrs["varianza_explicada"]

# Colores: una paleta diversificada para las 6 rutas
COLORES_RUTAS = {
    "Medellin-Bogota":            COLORS["arq_negocios"],
    "Medellin-Cartagena":         COLORS["coral"],
    "Medellin-Guatape":           COLORS["turquesa"],
    "Medellin-Manizales":         COLORS["arq_mochilero"],
    "Medellin-Pereira":           COLORS["amarillo"],
    "Medellin-SantaFeAntioquia":  COLORS["rosa"],
}

fig_pca = go.Figure()

for _, fila in df_pca.iterrows():
    es_seleccionada = (fila["Ruta"] == ruta)
    fig_pca.add_trace(
        go.Scatter(
            x=[fila["PC1"]],
            y=[fila["PC2"]],
            mode="markers+text",
            marker=dict(
                size=28 if es_seleccionada else 22,
                color=COLORES_RUTAS.get(fila["Ruta"], COLORS["texto_soft"]),
                line=dict(
                    color=COLORS["coral"] if es_seleccionada else "white",
                    width=4 if es_seleccionada else 2,
                ),
            ),
            text=[fila["Ruta"].replace("Medellin-", "")],
            textposition="top center",
            textfont=dict(
                size=13,
                family="Fraunces, serif",
                color=COLORS["texto"],
            ),
            name=fila["Ruta"],
            showlegend=False,
            hovertemplate=(
                f"<b>{fila['Ruta']}</b><br>"
                f"PC1 = {fila['PC1']:.3f}<br>"
                f"PC2 = {fila['PC2']:.3f}<extra></extra>"
            ),
        )
    )

fig_pca.update_layout(
    plot_bgcolor=COLORS["crema"],
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color=COLORS["texto"]),
    height=480,
    xaxis=dict(
        title=f"PC1 ({varianza[0]*100:.1f}% varianza)",
        gridcolor="#E5E5E5",
        zerolinecolor="#D0D0D0",
    ),
    yaxis=dict(
        title=f"PC2 ({varianza[1]*100:.1f}% varianza)",
        gridcolor="#E5E5E5",
        zerolinecolor="#D0D0D0",
    ),
    margin=dict(l=20, r=20, t=20, b=20),
)

st.plotly_chart(fig_pca, use_container_width=True)

st.markdown(
    f"""
    <div style="background: {COLORS['crema_dark']}; padding: 1rem 1.5rem;
                border-radius: 12px; margin-top: 1rem;">
        <strong>Varianza explicada:</strong> PC1 = {varianza[0]*100:.1f}%, PC2 = {varianza[1]*100:.1f}%
        (total {(varianza[0]+varianza[1])*100:.1f}%).
        Como demostramos en el reporte, PC1 captura principalmente la
        intensidad de la <em>estacionalidad semanal</em>, mientras que PC2
        opone las rutas sensibles a la <em>temporada alta turística</em>
        a las rutas de demanda regular.
    </div>
    """,
    unsafe_allow_html=True,
)
