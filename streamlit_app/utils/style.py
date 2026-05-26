"""
Constantes visuales y estilos compartidos entre todas las páginas de la app RutaViva.

Centralizamos aquí la paleta de colores, las fuentes y el CSS personalizado
para garantizar coherencia visual a través de la aplicación.
"""

import streamlit as st


# =============================================================================
# PALETA DE COLORES RUTAVIVA
# =============================================================================
# Inspirada en los colores del Caribe colombiano y del Eje Cafetero
# Combinación cálida y festiva, alejada de los azules corporativos genéricos

COLORS = {
    # Colores principales
    "coral":      "#E63946",  # Rojo coral del atardecer caribeño (color primario)
    "amarillo":   "#F4A261",  # Amarillo/naranja del sol andino
    "turquesa":   "#2A9D8F",  # Turquesa del mar Caribe
    "verde":      "#264653",  # Verde profundo del cafetal
    "rosa":       "#F4978E",  # Rosa suave (acentos)

    # Fondos
    "crema":      "#FFF8F0",  # Crema cálido (fondo principal)
    "crema_dark": "#FFE8D6",  # Crema más oscuro (cards, sidebar)

    # Texto
    "texto":      "#1D1D1F",  # Negro suave
    "texto_soft": "#6B7280",  # Gris medio para texto secundario

    # Colores por arquetipo (usados en el módulo 3)
    "arq_negocios":    "#264653",  # Verde corporativo
    "arq_familia":     "#F4A261",  # Amarillo cálido
    "arq_mochilero":   "#E76F51",  # Naranja terracota
    "arq_pareja":      "#E63946",  # Coral
    "arq_aventurero":  "#2A9D8F",  # Turquesa
}


# =============================================================================
# CSS PERSONALIZADO
# =============================================================================

CUSTOM_CSS = """
<style>
/* Importar fuentes desde Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&family=DM+Sans:wght@400;500;700&display=swap');

/* Tipografía global */
html, body, [class*="css"]  {
    font-family: 'DM Sans', sans-serif;
}

/* Títulos en Fraunces (serif moderno con carácter) */
h1, h2, h3, h4 {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}

h1 {
    font-weight: 800 !important;
    color: #264653;
}

/* Hero principal de la página de inicio */
.rutaviva-hero {
    background: linear-gradient(135deg, #E63946 0%, #F4A261 100%);
    padding: 3rem 2.5rem;
    border-radius: 24px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(230, 57, 70, 0.2);
}

.rutaviva-hero h1 {
    color: white !important;
    font-size: 3rem;
    margin: 0;
    line-height: 1.1;
}

.rutaviva-hero p {
    color: rgba(255, 255, 255, 0.95);
    font-size: 1.15rem;
    margin-top: 0.8rem;
    max-width: 600px;
}

/* Cards de personas */
.persona-card {
    background: white;
    padding: 1.8rem;
    border-radius: 16px;
    border-left: 6px solid var(--accent);
    height: 100%;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.persona-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
}

.persona-card h3 {
    margin: 0 0 0.3rem 0;
    color: #1D1D1F;
}

.persona-card .role {
    color: #6B7280;
    font-size: 0.9rem;
    margin-bottom: 1rem;
    font-style: italic;
}

.persona-card .quote {
    color: #1D1D1F;
    font-size: 0.95rem;
    line-height: 1.5;
}

/* Cards de módulos */
.modulo-card {
    background: #FFE8D6;
    padding: 2rem;
    border-radius: 20px;
    border: 2px solid transparent;
    margin: 0.5rem 0;
    transition: all 0.2s ease;
}

.modulo-card:hover {
    border-color: #E63946;
    transform: translateY(-2px);
}

.modulo-card .icono {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.modulo-card h3 {
    margin: 0.5rem 0;
}

.modulo-card .descripcion {
    color: #6B7280;
    font-size: 0.95rem;
}

/* Badge de estado de módulo */
.badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-ok       { background: #2A9D8F; color: white; }
.badge-pending  { background: #F4A261; color: white; }

/* Sidebar — botones de navegación más bonitos */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFE8D6 0%, #FFF8F0 100%);
}

/* Métricas */
[data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif;
    font-weight: 600;
}

/* Reducir padding superior de la página */
.block-container {
    padding-top: 2rem !important;
}
</style>
"""


def aplicar_estilo():
    """Aplica el CSS personalizado a la página actual.

    Llamar al inicio de cada página de Streamlit.
    """
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def color_arquetipo(arquetipo: str) -> str:
    """Devuelve el color asociado a un arquetipo de viajero."""
    mapa = {
        "Negocios":   COLORS["arq_negocios"],
        "Familia":    COLORS["arq_familia"],
        "Mochilero":  COLORS["arq_mochilero"],
        "Pareja":     COLORS["arq_pareja"],
        "Aventurero": COLORS["arq_aventurero"],
    }
    return mapa.get(arquetipo, COLORS["texto_soft"])
