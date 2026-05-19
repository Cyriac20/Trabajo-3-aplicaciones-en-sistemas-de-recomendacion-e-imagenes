# Proyecto 3 — Sistema Inteligente Integrado para Empresa de Transporte

**Curso:** Redes Neuronales Artificiales
**Equipo:** Cyriac y Juan José

## 1. Resumen del proyecto

Sistema integrado para una empresa de transporte con tres módulos de deep learning más una interfaz web:

1. **Módulo 1** — Predicción de la demanda de transporte (series temporales)
2. **Módulo 2** — Clasificación de imágenes de conducción distraída (CNN)
3. **Módulo 3** — Sistema de recomendación de destinos de viaje
4. **Módulo 4** — Herramienta web (Streamlit) que integra los tres módulos

## 2. División del trabajo

### Juan José
- **Módulo 2** — Clasificación de imágenes de conducción distraída (Dataset: Multi-class Driver Behavior Image Dataset)
- Secciones del reporte correspondientes al Módulo 2
- Sección de aspectos éticos, sesgos y privacidad
- Apoyo en la grabación del video final

### Cyriac
- **Módulo 1** — Series temporales (LSTM en PyTorch)
- **Módulo 3** — Sistema de recomendación (filtrado colaborativo)
- Coordinación general del reporte y del repositorio
- Secciones del reporte de Módulos 1 y 3 + introducción y conclusión

### Trabajo flexible — Módulo 4 (Streamlit)
- La interfaz web la asume quien vaya más adelantado con sus módulos.

## 3. Stack técnico

- **Lenguaje:** Python 3.10+
- **Deep Learning:** PyTorch + torchvision
- **Data science:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Interfaz web:** Streamlit
- **Entorno:** Google Colab
- **Versionado:** GitHub

## 4. Estructura del repositorio
```
proyecto3-transporte/
├── README.md
├── requirements.txt
├── data/                  # (gitignored, Datasets descargados)
├── notebooks/             # Notebooks para Análisis Exploratorio (EDA)
├── modulo1_series_temporales/
├── modulo2_clasificacion_imagenes/
├── modulo3_recomendacion/
├── app_web/               # Aplicación de Streamlit
├── reporte/               # Reporte técnico final (formato blog)
└── video/                 # Video de demostración
```

## Recomendaciones adicionales de IA

- **Modularización**: Mantengan el código de entrenamiento (`train.py`), la definición de red neuronal (`model.py`), y el uso para inferencia (`predict.py` o similar) lo más independientes y limpios posibles.
- **Reproducibilidad**: Guarden o documenten el estado del `random seed` para PyTorch y Numpy, permitiendo que sus pruebas siempre den el mismo resultado.
- **Paths relativos**: Usen librerías como `pathlib` o `os.path` para cargar datos. Esto evita que los directorios absolutos de Colab vs su PC choquen al ejecutar el script en otro entorno.
- **Descarga de Datos en Local**: No commitear datos, especialmente archivos grandes. Asegúrense de usar `gdown` o la API de Kaggle para bajarlos dentro de Colab rápidamente.

---
*Nota: Para instalar las dependencias, ejecutar `pip install -r requirements.txt`*
