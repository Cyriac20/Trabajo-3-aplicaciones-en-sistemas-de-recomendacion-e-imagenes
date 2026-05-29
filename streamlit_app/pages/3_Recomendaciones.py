"""
Módulo 3 — Sistema de Recomendación de Destinos
================================================

Página de la app dedicada al sistema de recomendación. Permite:
  1. Seleccionar un usuario de la base de RutaViva
  2. Visualizar su perfil (arquetipo del ground truth) e historial de visitas
  3. Generar y mostrar las top-10 recomendaciones del modelo MF
  4. Explorar la estructura latente aprendida (PCA de embeddings)

Atendido por la persona "Andrés" del Design Thinking.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.style import aplicar_estilo, color_arquetipo, COLORS
from utils.modulo3 import (
    cargar_metadata,
    cargar_users,
    cargar_ground_truth,
    obtener_historial,
    top_k_recomendaciones,
    calcular_pca_embeddings,
)


# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Recomendaciones · RutaViva",
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
            <h1 style="margin: 0;">Sistema de Recomendación</h1>
            <p style="color: {COLORS['texto_soft']}; margin: 0; font-size: 1.05rem;">
                Sugerencias personalizadas de destinos turísticos —
                <em>al servicio de Andrés y de los clientes RutaViva</em>
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# CARGA DE DATOS
# =============================================================================

users = cargar_users()
ground_truth = cargar_ground_truth()
metadata = cargar_metadata()

# Construir un DataFrame combinado para el selector
users_full = users.merge(ground_truth, on="UserID")


# =============================================================================
# SECCIÓN 1 — SELECCIÓN DEL USUARIO
# =============================================================================

st.markdown("### 1 · Seleccionar un cliente")
st.caption(
    "El filtro por arquetipo es **solo un atajo para encontrar usuarios** en la lista de 900 clientes; "
    "no condiciona qué destinos se recomiendan. Las recomendaciones provienen siempre del modelo MF, "
    "que nunca vio el arquetipo durante el entrenamiento (ver sección 4)."
)

col_sel_1, col_sel_2 = st.columns([2, 1])

with col_sel_1:
    # Opción de filtrar por arquetipo para facilitar la exploración
    arquetipos_disponibles = ["Todos"] + sorted(ground_truth["Arquetipo"].unique())
    arquetipo_filtro = st.selectbox(
        "Atajo: filtrar la lista de usuarios por arquetipo (opcional)",
        arquetipos_disponibles,
        index=0,
        help="Solo acorta la lista del selector. No modifica las recomendaciones del modelo.",
    )

    # Aplicar el filtro
    if arquetipo_filtro == "Todos":
        users_filtrados = users_full
    else:
        users_filtrados = users_full[users_full["Arquetipo"] == arquetipo_filtro]

    # Selector de usuario con etiqueta informativa
    opciones_user = users_filtrados.apply(
        lambda r: f"#{r['UserID']:>3} · {r['Nombre']:<12} · {r['Edad']} años · {r['Arquetipo']}",
        axis=1,
    ).tolist()
    user_ids_filtrados = users_filtrados["UserID"].tolist()

    seleccion = st.selectbox(
        f"Elige un cliente ({len(users_filtrados)} disponibles)",
        options=range(len(opciones_user)),
        format_func=lambda i: opciones_user[i],
    )
    user_id = int(user_ids_filtrados[seleccion])

with col_sel_2:
    # Mostrar la "ficha" del usuario seleccionado
    user = users_full[users_full["UserID"] == user_id].iloc[0]
    color_arq = color_arquetipo(user["Arquetipo"])

    st.markdown(
        f"""
        <div style="background: white; padding: 1.5rem; border-radius: 16px;
                    border-left: 6px solid {color_arq};
                    box-shadow: 0 4px 20px rgba(0,0,0,0.05);">
            <div style="font-size: 0.8rem; color: {COLORS['texto_soft']};
                        text-transform: uppercase; letter-spacing: 0.1em;
                        font-weight: 600;">Usuario #{user['UserID']}</div>
            <div style="font-family: 'Fraunces', serif; font-size: 1.8rem;
                        font-weight: 600; margin: 0.3rem 0;">{user['Nombre']}</div>
            <div style="color: {COLORS['texto_soft']}; margin-bottom: 0.8rem;">
                {user['Edad']} años · {user['Genero']}
            </div>
            <div style="background: {color_arq}; color: white;
                        display: inline-block; padding: 0.4rem 1rem;
                        border-radius: 100px; font-weight: 600; font-size: 0.9rem;">
                {user['Arquetipo']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("&nbsp;")


# =============================================================================
# SECCIÓN 2 — HISTORIAL DEL USUARIO
# =============================================================================

st.markdown("### 2 · Historial de viajes")

historial = obtener_historial(user_id)

if len(historial) == 0:
    st.info("Este usuario aún no tiene historial de viajes.")
else:
    n_visitas = len(historial)
    n_reviews = historial["Rating"].notna().sum()
    rating_medio = historial["Rating"].mean()

    col_h1, col_h2, col_h3 = st.columns(3)
    col_h1.metric("Destinos visitados", n_visitas)
    col_h2.metric("Reviews dejadas", int(n_reviews))
    col_h3.metric(
        "Rating medio",
        f"{rating_medio:.2f} / 5" if not np.isnan(rating_medio) else "—",
    )

    st.markdown("&nbsp;")

    # Mostrar las 5 visitas más recientes en cards horizontales
    st.markdown("**Visitas más recientes**")
    historial_top = historial.head(5)
    cols_hist = st.columns(5)
    for col, (_, row) in zip(cols_hist, historial_top.iterrows()):
        rating_display = (
            f"{int(row['Rating'])}/5" if not pd.isna(row["Rating"]) else "Sin review"
        )
        fecha_display = pd.to_datetime(row["VisitDate"]).strftime("%b %Y")
        with col:
            st.markdown(
                f"""
                <div style="background: white; padding: 1rem; border-radius: 12px;
                            height: 130px; box-shadow: 0 2px 10px rgba(0,0,0,0.04);">
                    <div style="font-size: 0.7rem; color: {COLORS['texto_soft']};
                                text-transform: uppercase; letter-spacing: 0.08em;">
                        {fecha_display}
                    </div>
                    <div style="font-weight: 600; font-size: 0.9rem;
                                margin: 0.3rem 0; line-height: 1.2;">
                        {row['Name'][:35]}
                    </div>
                    <div style="font-size: 0.85rem; color: {COLORS['coral']};
                                font-weight: 600;">
                        {rating_display}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Tabla completa expandible
    with st.expander(f"Ver historial completo ({n_visitas} visitas)"):
        historial_display = historial.copy()
        historial_display["VisitDate"] = historial_display["VisitDate"].dt.strftime(
            "%Y-%m-%d"
        )
        historial_display["Categorias"] = historial_display["Categorias"].apply(
            lambda lst: ", ".join(lst) if isinstance(lst, list) else ""
        )
        st.dataframe(
            historial_display[
                ["VisitDate", "Name", "Ruta", "Departamento", "Categorias", "Rating"]
            ].rename(
                columns={
                    "VisitDate": "Fecha",
                    "Name": "Destino",
                    "Ruta": "Ruta RutaViva",
                    "Departamento": "Depto.",
                    "Categorias": "Categorías",
                    "Rating": "Rating",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

st.markdown("&nbsp;")


# =============================================================================
# SECCIÓN 3 — TOP-10 RECOMENDACIONES
# =============================================================================

st.markdown("### 3 · Top-10 destinos recomendados")
st.caption(
    "Generadas por el modelo de factorización matricial (MF). "
    "Se excluyen los destinos ya visitados por el usuario."
)

top10 = top_k_recomendaciones(user_id, k=10, excluir_visitados=True)

# Mostrar las 10 recomendaciones en una grilla 2x5
for fila in range(2):
    cols_rec = st.columns(5)
    for col_idx, col in enumerate(cols_rec):
        rank = fila * 5 + col_idx + 1
        if rank > len(top10):
            continue
        rec = top10.iloc[rank - 1]
        cats_html = "".join(
            f'<span style="background: {COLORS["crema_dark"]}; '
            f'padding: 0.15rem 0.5rem; border-radius: 100px; '
            f'font-size: 0.7rem; margin-right: 0.25rem; '
            f'color: {COLORS["verde"]};">{c}</span>'
            for c in rec["Categorias"][:2]  # máximo 2 categorías para ahorrar espacio
        )
        with col:
            st.markdown(
                f"""
                <div style="background: white; padding: 1rem; border-radius: 14px;
                            height: 200px; box-shadow: 0 4px 16px rgba(0,0,0,0.06);
                            display: flex; flex-direction: column;">
                    <div style="display: flex; justify-content: space-between;
                                align-items: center; margin-bottom: 0.6rem;">
                        <div style="background: linear-gradient(135deg, {COLORS['coral']}, {COLORS['amarillo']});
                                    color: white; width: 32px; height: 32px;
                                    border-radius: 50%; display: flex;
                                    align-items: center; justify-content: center;
                                    font-weight: 700; font-family: 'Fraunces', serif;">
                            {rank}
                        </div>
                        <div style="background: {COLORS['turquesa']}; color: white;
                                    padding: 0.2rem 0.6rem; border-radius: 100px;
                                    font-size: 0.75rem; font-weight: 600;">
                            {rec['RatingPredicho']:.2f}
                        </div>
                    </div>
                    <div style="font-weight: 600; font-size: 0.95rem;
                                line-height: 1.25; flex-grow: 1;">
                        {rec['Name']}
                    </div>
                    <div style="font-size: 0.75rem; color: {COLORS['texto_soft']};
                                margin: 0.4rem 0;">
                        {rec['Departamento']}
                    </div>
                    <div style="line-height: 1.6;">
                        {cats_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("&nbsp;")


# =============================================================================
# SECCIÓN 4 — VISUALIZACIÓN PCA DE LOS EMBEDDINGS
# =============================================================================

st.markdown("### 4 · Análisis PCA de los embeddings")

with st.expander("Cómo leer esta visualización", expanded=False):
    st.markdown(
        """
        Durante el entrenamiento, el modelo MF asocia a cada uno de los **900 usuarios**
        un vector latente de **8 dimensiones** que captura sus preferencias.

        **Crucialmente, el modelo nunca vio el arquetipo verdadero de cada usuario**
        (negocios, familia, mochilero, pareja, aventurero). Solo vio las interacciones
        usuario × destino × rating.

        La visualización siguiente proyecta estos 900 embeddings en 2D mediante PCA,
        coloreados a posteriori por el arquetipo del ground truth. **Si vemos clusters
        bien separados, eso significa que el modelo redescubrió la estructura latente
        del dataset sin supervisión directa** — una validación de la calidad de las
        representaciones aprendidas.
        """
    )

df_pca = calcular_pca_embeddings()
varianza = df_pca.attrs["varianza_explicada"]

# Identificar el usuario seleccionado
usuario_actual = df_pca[df_pca["UserID"] == user_id].iloc[0]

# Plot Plotly: scatter con todos los usuarios + punto destacado para el seleccionado
fig = px.scatter(
    df_pca,
    x="PC1",
    y="PC2",
    color="Arquetipo",
    color_discrete_map={
        "Negocios":   COLORS["arq_negocios"],
        "Familia":    COLORS["arq_familia"],
        "Mochilero":  COLORS["arq_mochilero"],
        "Pareja":     COLORS["arq_pareja"],
        "Aventurero": COLORS["arq_aventurero"],
    },
    hover_data={"UserID": True, "PC1": ":.2f", "PC2": ":.2f"},
    opacity=0.55,
    title=None,
)
fig.update_traces(marker=dict(size=7, line=dict(width=0)))

# Añadir un punto destacado para el usuario seleccionado
fig.add_trace(
    go.Scatter(
        x=[usuario_actual["PC1"]],
        y=[usuario_actual["PC2"]],
        mode="markers",
        marker=dict(
            size=22,
            color="rgba(0,0,0,0)",
            line=dict(color=COLORS["coral"], width=4),
        ),
        name=f"Usuario seleccionado #{user_id}",
        showlegend=True,
        hovertemplate=(
            f"<b>Usuario #{user_id}</b><br>"
            f"PC1 = {usuario_actual['PC1']:.2f}<br>"
            f"PC2 = {usuario_actual['PC2']:.2f}<br>"
            f"Arquetipo: {usuario_actual['Arquetipo']}<extra></extra>"
        ),
    )
)

fig.update_layout(
    plot_bgcolor=COLORS["crema"],
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color=COLORS["texto"]),
    height=520,
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
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02,
        bgcolor="rgba(255,255,255,0.5)",
    ),
    margin=dict(l=20, r=20, t=20, b=20),
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"""
    <div style="background: {COLORS['crema_dark']}; padding: 1rem 1.5rem;
                border-radius: 12px; margin-top: 1rem;">
        <strong>Varianza explicada:</strong> PC1 = {varianza[0]*100:.1f}%, PC2 = {varianza[1]*100:.1f}%
        (total {(varianza[0]+varianza[1])*100:.1f}%).
        La separación visible entre arquetipos confirma que el modelo aprendió a
        agrupar a los usuarios según sus preferencias latentes, <strong>sin haber
        recibido nunca esta información como input</strong>.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("&nbsp;")


# =============================================================================
# FOOTER: MÉTRICAS DEL MODELO
# =============================================================================

with st.expander("Métricas del modelo MF (sobre el conjunto de prueba)"):
    metricas_mf = metadata["metricas"]["MF"]
    metricas_baseline = metadata["metricas"]["baseline_popularidad"]

    col_met = st.columns(4)
    col_met[0].metric("RMSE", f"{metricas_mf['RMSE']:.3f}")
    col_met[1].metric("MAE", f"{metricas_mf['MAE']:.3f}")
    col_met[2].metric(
        "Precision@10",
        f"{metricas_mf['Precision@10']:.3f}",
        f"+{(metricas_mf['Precision@10']/metricas_baseline['Precision@10'] - 1)*100:.0f}% vs baseline",
    )
    col_met[3].metric(
        "Recall@10",
        f"{metricas_mf['Recall@10']:.3f}",
        f"+{(metricas_mf['Recall@10']/metricas_baseline['Recall@10'] - 1)*100:.0f}% vs baseline",
    )

    st.caption(
        f"Modelo entrenado sobre {metadata['dataset']['n_train_reviews']} reviews "
        f"({metadata['hiperparametros']['n_epochs']} épocas, "
        f"embedding_dim = {metadata['hiperparametros']['embedding_dim']}, "
        f"learning_rate = {metadata['hiperparametros']['lr']})."
    )
