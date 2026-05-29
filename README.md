<div align="center">
  <h1>🚌 RutaViva — Sistema Inteligente Integrado</h1>
  <p><em>Predicción de demanda, detección de conducción distraída y recomendación de destinos con Deep Learning</em></p>

<!-- Badges -->

<a href="https://trabajo-3-aplicaciones-en-sistemas-de-recomendacion-e-imagenes.streamlit.app/">
    <img src="https://img.shields.io/badge/Aplicación_Web-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="App Web"/>
  </a>
  <a href="https://cyriac20.github.io/Trabajo-3-aplicaciones-en-sistemas-de-recomendacion-e-imagenes/reporte_tecnico_rutaviva.html">
    <img src="https://img.shields.io/badge/Reporte_Técnico-Publicado-0F172A?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Reporte Técnico"/>
  </a>
  <a href="https://youtube.com/">
    <img src="https://img.shields.io/badge/Video_Demostración-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Video"/>
  </a>
</div>

<br>

Proyecto desarrollado para el curso de **Redes Neuronales Artificiales y Algoritmos Bioinspirados** (UNAL Medellín, 2026-1). Integra tres modelos de aprendizaje profundo al servicio de **RutaViva**, una PME colombiana ficticia de transporte intermunicipal, para abordar tres retos del sector: la planificación de flota, la seguridad vial y la fidelización de pasajeros.

## 🎯 Objetivo y Misión

Construir un sistema integrado que combine **series de tiempo**, **visión por computador** y **filtrado colaborativo** en una única herramienta web, demostrando cómo el deep learning puede aportar valor operativo a una empresa de transporte real. Cada módulo se diseñó a partir de una *persona* concreta (Carlos en operaciones, Diana en seguridad, Andrés como pasajero) bajo metodología Design Thinking.

---

## 🚀 Entregables Públicos

1. **[🖥️ Aplicación Web Interactiva](https://trabajo-3-aplicaciones-en-sistemas-de-recomendacion-e-imagenes.streamlit.app/)**: Permite predecir demanda por ruta a 30 días, subir imágenes de conductores para clasificar comportamientos distractivos y generar recomendaciones de destinos por usuario.
2. **[📄 Reporte Técnico / Blog Post](https://cyriac20.github.io/Trabajo-3-aplicaciones-en-sistemas-de-recomendacion-e-imagenes/reporte_tecnico_rutaviva.html)**: Documentación completa del proceso (Design Thinking, datasets, arquitecturas, evaluación, ética).
3. **[🎬 Video de Demostración](https://youtube.com/)**: Recorrido por los tres módulos en la herramienta web y contribuciones individuales del equipo.

---

## 📊 Resultados por Módulo

| Módulo | Modelo | Métricas clave |
| --- | --- | --- |
| **1. Predicción de demanda** | LSTM global con embedding de ruta | RMSE **19.92** · MAE **14.36** |
| **2. Conducción distraída** | ResNet18 (transfer learning) sobre 5 clases | Accuracy **71.15%** · F1 ponderado **70.76%** |
| **3. Recomendación de destinos** | Matrix Factorization (PyTorch) | RMSE **1.040** · Recall@10 **0.767** |

---

## 🛠️ Instalación y Replicación (Desarrollo Local)

### 1. Requisitos Previos

- Python 3.10 o superior.
- (Opcional) Acceso a GPU CUDA para reentrenar el clasificador de imágenes.

### 2. Configuración y Dependencias

```bash
# Clonar el proyecto
git clone https://github.com/Cyriac20/Trabajo-3-aplicaciones-en-sistemas-de-recomendacion-e-imagenes.git
cd Trabajo-3-aplicaciones-en-sistemas-de-recomendacion-e-imagenes

# Instalar dependencias de la app
pip install -r streamlit_app/requirements.txt
```

### 3. Lanzar la Aplicación Web

La app utiliza los modelos pre-entrenados que ya viven en `streamlit_app/models/`.

```bash
cd streamlit_app
streamlit run app.py
```

Tras arrancar, se habilitan las páginas:

- **Demanda** — selección de ruta, modo evaluación o predicción a 30 días, métricas y gráfica real vs predicho.
- **Conduccion** — carga de imagen e inferencia del clasificador CNN con probabilidades por clase.
- **Recomendaciones** — selección de usuario, historial de viajes y top-10 de destinos sugeridos.

---

## 📂 Arquitectura del Repositorio

```text
├── streamlit_app/
│   ├── app.py                          # Página de inicio (hero + personas + módulos)
│   ├── pages/
│   │   ├── 1_Demanda.py                # Módulo 1 — series de tiempo
│   │   ├── 2_Conduccion.py             # Módulo 2 — clasificación de imágenes
│   │   └── 3_Recomendaciones.py        # Módulo 3 — recomendación
│   ├── utils/                          # Carga de modelos, preprocesamiento, estilo
│   ├── models/                         # Pesos entrenados (.pth) y metadatos
│   ├── data/                           # CSVs usados por la app
│   ├── assets/                         # Imágenes y visualizaciones
│   └── requirements.txt
│
├── notebooks/
│   ├── generacion_dataset_modulo1.ipynb    # Dataset sintético de demanda
│   ├── modulo1_series_temporales.ipynb     # EDA + entrenamiento LSTM
│   ├── eda_modulo2.ipynb                   # EDA del dataset de conducción
│   ├── generacion_dataset_modulo3.ipynb    # Dataset sintético usuario-destino
│   ├── modelo3_recomendacion.ipynb         # Matrix Factorization y evaluación
│   └── figures/                            # Gráficas exportadas para el reporte
│
├── modulo2_clasificacion_imagenes/
│   ├── train.py                        # Entrenamiento ResNet18
│   ├── evaluate_and_plot.py            # Matriz de confusión y ejemplos
│   └── model.py
│
├── reporte/                            # Versión simplificada del reporte
├── reporte_tecnico_rutaviva.html       # Reporte técnico completo (blog post)
└── video/                              # Videos de demostración (entregable)
```

---

## 🧠 Decisiones Técnicas Destacadas

- **Módulo 1** usa una sola LSTM **global** con embeddings de ruta, en lugar de un modelo por ruta. Permite compartir patrones estacionales (festivos, fines de semana) entre rutas con pocos datos.
- **Módulo 2** parte de **ResNet18 pre-entrenada en ImageNet** y descongela las últimas capas. Se priorizó F1 ponderado por el desbalance de clases en el dataset público de Kaggle.
- **Módulo 3** implementa **Matrix Factorization en PyTorch** (no librería externa), lo que permite incorporar embeddings ajustables y evaluar Precision@K / Recall@K sobre un split temporal de los ratings.

---

## ⚠️ Notas sobre los Datos

- **Módulos 1 y 3** usan datasets **sintéticos** generados por nosotros, diseñados con estructura realista (estacionalidad semanal/mensual, festivos colombianos, perfiles de usuario) para el caso RutaViva.
- **Módulo 2** usa el dataset público [Multi-Class Driver Behavior Image Dataset](https://www.kaggle.com/datasets/arafatsahinafridi/multi-class-driver-behavior-image-dataset/data) de Kaggle.

---

## 📚 Bibliografía y Referencias Clave

- Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory*. Neural Computation, 9(8), 1735–1780.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. CVPR.
- Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix factorization techniques for recommender systems*. Computer, 42(8), 30–37.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

---

<div align="center">
  <sub>Universidad Nacional de Colombia · Sede Medellín · Redes Neuronales Artificiales · 2026-1</sub><br>
  <sub>Equipo: <b>Juan José Zapata Moreno</b> · <b>Cyriac Salignat</b></sub>
</div>
