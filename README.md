# RutaViva — Sistema Inteligente Integrado

Aplicación web (Streamlit) del Proyecto 3 del curso de **Redes Neuronales Artificiales**.

Integra tres módulos de deep learning al servicio de una PME ficticia de transporte intermunicipal colombiana :

| Módulo | Tarea | Modelo |
|--------|-------|--------|
| 1 | Predicción de demanda de pasajeros a 30 días | LSTM con embeddings de ruta |
| 2 | Clasificación de conducción distraída | CNN (transfer learning) |
| 3 | Recomendación de destinos turísticos | Matrix Factorization |

## Equipo

- Cyriac
- Juan José

## Estructura del repo

```
streamlit_app/
├── app.py                          # Página de inicio
├── pages/
│   ├── 1_📈_Demanda.py             # Módulo 1
│   ├── 2_🛣️_Conducción.py          # Módulo 2 (placeholder)
│   └── 3_🧭_Recomendaciones.py     # Módulo 3
├── utils/                          # Funciones compartidas (estilo, carga de modelos)
├── models/                         # Pesos de los modelos (.pth) y metadatos
├── data/                           # CSVs necesarios para las páginas
├── .streamlit/config.toml          # Tema visual personalizado
└── requirements.txt
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue

La aplicación está desplegada en [Streamlit Community Cloud](https://streamlit.io/cloud).
