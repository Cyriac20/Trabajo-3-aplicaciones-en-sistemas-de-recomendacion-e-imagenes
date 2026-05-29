# Modulo 1 — Prediccion de la demanda de transporte (series temporales)

**Proyecto 3 — Sistema Inteligente Integrado para RutaViva**
**Curso:** Redes Neuronales Artificiales
**Equipo:** Cyriac y Juan José

---

## 1. Introduccion y objetivos

En este notebook desarrollamos el **Modulo 1** del proyecto: la prediccion de la demanda diaria de pasajeros para las 6 rutas operadas por **RutaViva**, sobre un horizonte de **30 dias**.

### Problema

> *Como Gerente de Operaciones, necesito anticipar la demanda de pasajeros por ruta y por dia durante los proximos 30 dias, para optimizar la asignacion de vehiculos y personal, reducir costos y evitar tanto la sobrecapacidad como la subcapacidad.*

### Decisiones tecnicas adoptadas

| Decision | Eleccion | Justificacion |
|---|---|---|
| Arquitectura | **Un solo modelo global con embedding de ruta** | Mas datos disponibles, transferencia de patrones entre rutas, mantenimiento simplificado |
| Horizonte | **Salida directa a 30 dias (multi-step direct)** | Mas simple que la prediccion recursiva, evita la acumulacion de error |
| Entrada | **Ventana de 60 dias** | Captura al menos un ciclo estacional semanal × 8 semanas |
| Features | **pasajeros normalizados + sin/cos dia de semana + sin/cos mes + festivo + embedding de ruta** | Patrones temporales explicitos + identidad de ruta latente |
| Normalizacion | **Min-Max por ruta, calculada sobre el train** | Pone todas las rutas en la misma escala [0, 1], evita que el modelo sobre-pondere las rutas de alto volumen |
| Split | **Temporal:** train 2023-01-01 → 2024-10-31 / test 2024-11-01 → 2024-12-31 | Simula despliegue real, evaluacion sobre temporada alta de fin de ano |

### Metricas

- **RMSE** y **MAE** en escala original (pasajeros), por ruta y global.
- **Comparacion con un baseline naive** (promedio por dia de la semana) para verificar si la LSTM mejora una regla simple.


---

## 2. Configuracion del entorno y montaje de Google Drive

---

## 3. Carga del dataset y exploracion inicial

El dataset `rutaviva_demanda.csv` fue generado sinteticamente (ver `justificacion_dataset_modulo1.md`). Contiene la demanda diaria de pasajeros para 6 rutas intermunicipales sobre 2 anos (2023-2024).

**Observaciones esperadas:** cada ruta tiene su propio nivel medio de demanda (algunas son de alto volumen, otras pequenas). Estas diferencias de escala (potencialmente un factor 5-8x entre la ruta mas grande y la mas pequena) justifican la **normalizacion por ruta** que aplicaremos mas adelante.

---

## 4. Analisis exploratorio (EDA)

Visualizamos los patrones temporales para entender:

1. El **nivel** y la **forma general** de cada ruta.
2. La **estacionalidad semanal** (efecto del dia de la semana).
3. La **estacionalidad anual** (meses, festivos, temporada alta).

**Lectura del grafico:** se observa la **estacionalidad anual** (picos en diciembre, alza en Semana Santa y vacaciones de mitad de ano), una ligera **tendencia** en algunas rutas, y las variaciones de alta frecuencia (de un dia a otro) que corresponden a la **estacionalidad semanal** y al **ruido aleatorio**.

**Lectura del grafico:** los **viernes, sabados y domingos** suelen mostrar picos en rutas turisticas (Guatape, Santa Fe de Antioquia, Cartagena), mientras que las rutas mas de "negocio" tienen un perfil mas plano o concentrado a principios de semana. Esta heterogeneidad entre rutas justifica el uso del **embedding de ruta** en el modelo (cada ruta tiene su propia identidad).

**Lectura del grafico:** el heatmap confirma la **temporada alta de diciembre** y los picos secundarios de Semana Santa (marzo-abril) y vacaciones de mitad de ano (junio-julio). Estos patrones justifican la inclusion del **mes como feature** (codificado de forma ciclica).

---

## 5. Preprocesamiento

Las etapas son:

1. **Codificacion ciclica** del dia de la semana y del mes (sin/cos).
2. **Indicador de festivo colombiano** (feature binaria).
3. **Codificacion entera** de la ruta (`ruta_id` de 0 a 5) — sera usada como entrada del embedding.
4. **Normalizacion Min-Max por ruta** de la variable objetivo (calculada solo sobre el train).
5. **Creacion de las secuencias** (X de 60 dias × features, junto con `ruta_id`, e y de 30 dias).
6. **Split temporal** train/test.

### ¿Por que codificacion ciclica (sin/cos) en lugar de one-hot?

Para variables ciclicas como el dia de la semana, el lunes (0) y el domingo (6) son **adyacentes** en el tiempo, pero numericamente estan en los extremos opuestos. Si pasaramos el valor crudo (0 a 6), el modelo asumiria que el domingo esta "muy lejos" del lunes, lo cual es falso. La codificacion **(sin, cos)** sobre un circulo soluciona este problema: cualquier dia tiene exactamente dos vecinos a la misma distancia.

### ¿Por que normalizar la variable objetivo *por ruta*?

Las 6 rutas tienen escalas muy diferentes (un factor ~8x entre la mas grande y la mas pequena). Si entrenamos un modelo global sobre valores brutos:

- La loss MSE penalizaria proporcionalmente mas las rutas de alto volumen (errores absolutos mas grandes ⇒ errores al cuadrado mucho mayores).
- El optimizador concentraria el aprendizaje en esas rutas grandes y subestimaria las pequenas.

Al normalizar **por ruta** en [0, 1], todas las rutas comparten la misma escala relativa. El modelo aprende patrones temporales **relativos** ("el sabado esta 30 % por encima de la media") que son universales, y el **embedding de ruta** se encarga de capturar las especificidades de cada una.

### ¿Por que calcular min/max solo sobre el train?

Calcular los parametros de normalizacion sobre todo el dataset (train + test) introduciria un **data leakage**: el modelo tendria acceso indirecto a informacion del futuro (los picos de diciembre 2024 informarian la normalizacion durante el entrenamiento). Calculamos `min`/`max` solo sobre el train, simulando un despliegue real donde el futuro es desconocido.

---

## 6. Modelo LSTM con embedding de ruta

### Arquitectura

El modelo combina dos tipos de entrada:

1. **Entrada secuencial** `(batch, 60, 6)` — la dinamica temporal (pasajeros_norm + 5 features temporales).
2. **Entrada estatica** `(batch,)` — el `ruta_id` (entero entre 0 y 5).

**Flujo del forward:**

```
ruta_id ──► nn.Embedding(6, 4) ──► vector_ruta (batch, 4)
                                        │
                                        │  repetir en cada paso de tiempo
                                        ▼
secuencia (batch, 60, 6) ──► concatenar ──► (batch, 60, 10) ──► LSTM ──► fc ──► (batch, 30)
```

### ¿Por que repetir el embedding en cada paso de tiempo?

Asi, el LSTM "sabe" en cada paso de tiempo de que ruta esta hablando. La identidad de la ruta influye en como el LSTM interpreta cada nueva observacion (por ejemplo: un pico de demanda no significa lo mismo en una ruta de fin de semana turistico vs. en una ruta de negocio entre semana).

### Visualizacion de embeddings

Despues del entrenamiento se proyectan los **embeddings aprendidos** a 2D mediante PCA. Esta visualizacion permite revisar si el modelo ubica cerca las rutas con comportamientos parecidos, por ejemplo rutas turisticas de fin de semana frente a rutas mas regulares.

---

## 7. Entrenamiento

**Una sola fase de entrenamiento** sobre todo el dataset concatenado. El modelo ve secuencias de todas las rutas en cada batch (mezcladas aleatoriamente), lo que le permite aprender patrones universales.

### Hiperparametros

| Parametro | Valor | Justificacion |
|---|---|---|
| Optimizador | Adam | Estandar, adaptativo, robusto |
| Learning rate | 1e-3 | Valor estandar que funciona bien con LSTM |
| Loss | MSE | Coherente con la metrica RMSE |
| Batch size | 64 | Mas grande que para 6 modelos (mas datos), acelera el entrenamiento |
| Epocas | 50 | Suficiente para converger |
| Hidden size LSTM | 64 | Recomendado en el plan del proyecto |
| Embedding dim | 4 | Suficiente para 6 rutas |

---

## 8. Evaluacion

### Metricas

- **RMSE** y **MAE** sobre el conjunto de test, **en escala original** (pasajeros), por ruta y global.
- **Comparacion con un baseline naive**: la prediccion = promedio del mismo dia de la semana sobre el train.

### ¿Por que comparar con un baseline?

Sin baseline, no sabemos si el LSTM aporta valor real. Si apenas iguala (o pierde frente a) un promedio simple por dia de la semana, eso indicaria que el modelo no aprendio nada util y deberiamos cuestionar la arquitectura.

---

## 9. Visualizaciones

Tres tipos de visualizacion:

1. **Prediccion vs. realidad** por ruta (primera secuencia de test).
2. **Comparacion del RMSE** por ruta (LSTM vs baseline).
3. **Embeddings aprendidos** (proyeccion 2D) para interpretar similitudes entre rutas.

### Visualizacion de los embeddings aprendidos

Extraemos los vectores de embedding aprendidos por el modelo (uno de 4 dimensiones por ruta) y los proyectamos a 2D con PCA. La lectura se centra en verificar si las distancias del grafico tienen sentido frente al comportamiento operativo de las rutas.

### Interpretacion cuantitativa de los embeddings

Para validar que la estructura visual observada en la proyeccion 2D corresponde a una organizacion semantica real (y no a un artefacto de la reduccion dimensional), calculamos la correlacion de Pearson entre las coordenadas PCA de cada ruta y los parametros narrativos utilizados durante la generacion sintetica del dataset (`base`, `peso_fin_de_semana`, `peso_temporada_alta`).

Si el modelo aprendio una representacion significativa, cada eje principal deberia relacionarse con algun parametro operativo, como la intensidad de fin de semana, la temporada alta o el nivel base de demanda.

**Interpretacion:**

- ¿Que rutas estan agrupadas juntas en el espacio de embedding?
- ¿Aparecen claramente las rutas turisticas (Guatape, Santa Fe) separadas de las rutas de larga distancia (Bogota, Cartagena)?
- ¿La estructura aprendida es coherente con la intuicion del negocio (RutaViva)?

Si la respuesta es positiva, el PCA sirve como evidencia adicional de que el modelo no solo ajusta valores numericos, sino que tambien organiza las rutas de forma coherente con sus patrones de demanda.

---

## 10. Analisis, discusion y conclusiones

### Resultados clave a revisar

- ¿En que rutas el LSTM supera claramente al baseline? ¿En cuales el baseline es igual de bueno?
- ¿La normalizacion por ruta + embedding permite manejar correctamente la heterogeneidad de las escalas?
- ¿Los embeddings aprendidos tienen una estructura interpretable?

### Ventajas observadas del enfoque "modelo global + embedding"

1. **Eficiencia computacional**: un solo entrenamiento en lugar de seis.
2. **Mejor performance esperada**: el modelo dispone de 6x mas datos.
3. **Representacion latente de las rutas**: los embeddings ofrecen una interpretacion semantica del portafolio.
4. **Despliegue simplificado**: un solo archivo de modelo (.pth) a cargar en la app Streamlit.

### Limitaciones identificadas

1. **Dataset sintetico** — los patrones aprendidos son los que se inyectaron en la generacion (ver `justificacion_dataset_modulo1.md`). Discutido en la seccion de aspectos eticos.
2. **Horizonte fijo de 30 dias** — el modelo no genera intervalos de confianza, solo una prediccion puntual.
3. **Festivos colombianos hardcodeados** — un uso en produccion requeriria una libreria como `holidays`.
4. **Embedding fijo durante la inferencia** — si RutaViva agrega una nueva ruta, habria que reentrenar el modelo.

### Trabajo futuro

- Validar el modelo sobre datos reales del DANE o de una empresa real de transporte intermunicipal colombiano.
- Anadir variables exogenas (clima, eventos especiales, precio del combustible).
- Probar arquitecturas mas avanzadas (GRU, Transformer ligero).
- Generar intervalos de confianza mediante MC Dropout o Quantile Regression.
- Permitir embeddings "cold-start" para nuevas rutas (basados en metadatos: distancia, tipo turistico/business, etc.).

### Integracion con el resto del proyecto

El modelo entrenado (`lstm_modelo_global.pth`) y sus metadatos (`metadata_modelo_global.json`) se guardaron en `MODELS_DIR` y seran cargados por la **aplicacion web Streamlit (Modulo 4)** para que el Gerente de Operaciones (persona Carlos) pueda visualizar las predicciones de demanda directamente desde la interfaz.
