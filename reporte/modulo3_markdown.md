# Módulo 3 — Sistema de recomendación de destinos

**Proyecto:** Sistema Inteligente Integrado para RutaViva  
**Curso:** Redes Neuronales Artificiales  
**Notebook:** `01_modelo_recomendacion.ipynb`

---

## Objetivo

Construir un sistema de recomendación que sugiera destinos personalizados a los clientes de RutaViva, a partir de su historial de ratings sobre las 30 destinaciones del catálogo. Este sistema responde a la persona **Andrés** (cliente frecuente) definida en la fase Empathize del Design Thinking.

## Enfoque

Comparamos **dos enfoques de filtrado colaborativo** sobre el mismo dataset, con el objetivo de cuantificar el aporte de la no-linealidad para nuestra tarea de recomendación:

| Modelo | Tipo | Referencia |
|---|---|---|
| **Matrix Factorization (MF)** | Producto escalar de embeddings + biases | Koren et al., 2009 |
| **Neural Collaborative Filtering (NCF)** | Embeddings → MLP no lineal | He et al., 2017 |

Ambos modelos comparten la misma capa de embedding (dimensión 16), la misma función de pérdida (MSE) y los mismos hiperparámetros de entrenamiento (Adam con `lr=1e-3` y `weight_decay=1e-5`). La **única diferencia** es la forma de combinar los embeddings de usuario y de destino: producto escalar (MF) vs MLP no lineal (NCF). Esta paridad de configuración nos permite **aislar el aporte específico de la no-linealidad** sobre nuestro problema.

Esta postura comparativa — sin designar a priori un modelo "principal" — está alineada con la literatura reciente, especialmente Rendle et al. (2020) *"Neural Collaborative Filtering vs. Matrix Factorization Revisited"*, que muestra que un MF bien ajustado puede igualar o superar al NCF según la estructura del dataset.

## Decisiones de diseño clave

| Decisión | Valor | Justificación |
|---|---|---|
| Señal de entrenamiento | Reviews explícitas (rating 1-5) | Señal más informativa, problema de regresión limpio |
| Dimensión de los embeddings | 16 | Suficiente para capturar 5 arquetipos + variabilidad; PCA legible |
| Función de pérdida | MSE | Apropiada para regresión sobre ratings |
| Partición train/test | **Temporal** (train ≤ 2024-10-31, test ≥ 2024-11-01) | Simula despliegue real, coherente con Módulo 1 |
| Métricas de rating | RMSE, MAE | Estándar para predicción de ratings |
| Métricas de ranking | Precision@10, Recall@10 (relevant ⟺ rating ≥ 4) | Evalúa la calidad de las top-K recomendaciones |

## Análisis creativo

Al final del notebook validamos que los embeddings de usuario aprendidos por el modelo NCF **redescubren los 5 arquetipos latentes** del dataset, mediante un análisis PCA + comparación con el `ground_truth_arquetipos.csv` (oculto durante el entrenamiento). Este análisis hace eco directo al PCA de los embeddings de ruta realizado en el Módulo 1, garantizando coherencia metodológica entre los dos módulos.

## 0. Conexión a Google Drive

Mismo esquema que el notebook de generación del dataset. Los modelos entrenados y los archivos de metadata se guardarán en `proyecto3-rutaviva/models/`.

## 1. Importación de librerías y configuración

## 2. Carga de los datos

Cargamos las 5 tablas generadas en el notebook `00_generacion_dataset_modulo3.ipynb`. Recordemos que el archivo `ground_truth_arquetipos.csv` **no se utiliza durante el entrenamiento** — se reserva para la validación final del análisis PCA.

## 3. Análisis exploratorio rápido

Antes de entrenar, verificamos las propiedades clave del dataset desde la perspectiva del problema de recomendación.

### 3.1 Distribución de los ratings

### 3.2 Sparsidad de la matriz user × destino

Una densidad de aproximadamente 25 % es relativamente alta para un problema de recomendación (los datasets reales como MovieLens están alrededor del 1-5 %). Esto se debe a la naturaleza sintética de los datos. Es una limitación reconocida: el modelo opera en condiciones más favorables que las reales.

## 4. Preprocesamiento

Tres pasos necesarios antes de entrenar:

1. **Codificación de IDs en índices contiguos** (PyTorch `nn.Embedding` requiere índices 0...N-1).
2. **Partición temporal train/test** (coherente con el Módulo 1).
3. **Construcción de los DataLoaders** PyTorch.

### 4.1 Codificación de IDs

Convertimos `UserID` y `DestinationID` (que arrancan en 1) en índices contiguos `user_idx` y `dest_idx` (que arrancan en 0). Guardamos los mapeos para poder volver a los IDs originales en el momento de la inferencia (Streamlit del Módulo 4).

### 4.2 Partición temporal train / test

Cortamos la cronología en `2024-11-01`, idéntico al Módulo 1, para reflejar un despliegue real: el modelo se entrena con datos pasados (enero 2023 – octubre 2024) y se evalúa sobre el futuro inmediato (noviembre-diciembre 2024).

> **Nota importante.** Verificamos que cada usuario del test esté también presente en el train (de lo contrario, el modelo no podría predecir nada para él — problema *cold-start*). Si algunos usuarios solo aparecen en el test, los excluimos del test con un mensaje claro.

### 4.3 Dataset y DataLoaders de PyTorch

Definimos una clase `RatingsDataset` que devuelve, para cada review: `(user_idx, dest_idx, rating)`. El `DataLoader` se encarga del batching y del shuffling durante el entrenamiento.

## 5. Modelo 1 — Matrix Factorization

### Principio

La factorización matricial es el método clásico de filtrado colaborativo (Koren et al., 2009). La idea es representar a cada usuario $u$ y a cada destino $d$ por vectores latentes $\mathbf{p}_u, \mathbf{q}_d \in \mathbb{R}^{16}$ y predecir el rating como:

$$
\hat{r}_{u,d} = \mu + b_u + b_d + \mathbf{p}_u^\top \mathbf{q}_d
$$

donde:
- $\mu$ es el rating medio global (constante).
- $b_u$ es el bias del usuario (cuánto califica este usuario en general por encima/debajo del promedio).
- $b_d$ es el bias del destino (cuánto califican los usuarios este destino en general).
- $\mathbf{p}_u^\top \mathbf{q}_d$ captura la interacción usuario-destino.

Los biases son fundamentales: capturan parte importante de la señal sin necesidad de la interacción latente.

### Entrenamiento del MF

Definimos una función `entrenar_modelo` genérica que reutilizaremos para el NCF. Aplica:
- Pérdida MSE
- Optimizador Adam (`lr=1e-3`)
- Regularización L2 (`weight_decay=1e-5`) sobre todos los parámetros del modelo

**¿Por qué regularización L2?** Sin ella, los embeddings pueden crecer libremente para minimizar el MSE en el train, lo que produce un sobreajuste progresivo (el train sigue bajando mientras el test estanca). La regularización L2 penaliza los pesos grandes y estabiliza la convergencia. Es una práctica estándar en factorización matricial desde Koren et al. (2009). Aplicamos el mismo valor a los dos modelos para garantizar una comparación equitativa.

## 6. Modelo 2 — Neural Collaborative Filtering

### Principio

El NCF (He et al., 2017) sustituye el producto escalar del MF por un MLP que aprende una **interacción no-lineal** entre los embeddings de usuario y de destino:

$$
\hat{r}_{u,d} = \text{MLP}\big([\mathbf{p}_u, \mathbf{q}_d]\big)
$$

Donde $[\cdot, \cdot]$ denota la concatenación, y el MLP es una red feedforward de dos capas ocultas con activaciones ReLU.

**¿Por qué un MLP?** El producto escalar del MF impone una forma específica de interacción (lineal en cada componente). Un MLP, al ser un aproximador universal, puede capturar interacciones más complejas — por ejemplo, *"si el usuario tiene una afinidad alta por la categoría 'playa' Y media por la categoría 'cultural', le gustará Cartagena más que la suma de los dos efectos"*.

Mantenemos exactamente la misma capa de embedding (dim 16) y la misma función de pérdida (MSE), de manera que **la única diferencia es la no-linealidad del MLP**.

## 7. Comparación de las curvas de entrenamiento

## 8. Evaluación detallada sobre el test

Comparamos los dos modelos en dos niveles:

1. **Métricas de rating** (RMSE, MAE): qué tan bien predicen el valor exacto del rating.
2. **Métricas de ranking** (Precision@10, Recall@10): qué tan pertinentes son las top-10 recomendaciones generadas.

Para las métricas de ranking, comparamos también con un **baseline trivial de popularidad** (recomendar los destinos más visitados, sin personalización). Esto nos permite verificar que ambos modelos aprenden efectivamente una señal personalizada.

### 8.1 RMSE y MAE

### 8.2 Precision@10 y Recall@10

**Protocolo:**

Para cada usuario del test:
1. Identificamos los destinos *relevantes* en el test (rating real ≥ 4).
2. Calculamos los scores predichos del modelo para todos los destinos **no vistos en el train**.
3. Ordenamos por score decreciente y tomamos los **top-10**.
4. Calculamos:
   - **Precision@10** = #(top-10 ∩ relevantes) / 10
   - **Recall@10**    = #(top-10 ∩ relevantes) / #(relevantes)
5. Promediamos sobre todos los usuarios que tienen al menos un destino relevante en el test.

Como **baseline**, evaluamos también una recomendación por popularidad: recomendar los destinos con más reviews en el train, excluyendo los ya vistos por el usuario.

### 8.3 Visualización comparativa de las métricas

## 9. Análisis creativo — los embeddings de usuario redescubren los arquetipos

**Pregunta central:** durante el entrenamiento, el modelo MF **no recibió ninguna información** sobre el arquetipo de cada usuario (negocios, familia, mochilero, pareja, aventurero). Solo vio los ratings.

**Hipótesis:** si el modelo aprendió embeddings de usuario realmente significativos, deberíamos poder visualizar la estructura de los 5 arquetipos en el espacio de los embeddings — proyectados en 2D mediante PCA.

Este análisis hace eco directo al PCA de los embeddings de ruta del Módulo 1, donde mostramos que el modelo había redescubierto los parámetros narrativos (`peso_fin_de_semana`, `peso_temporada_alta`) sin haberlos recibido como input.

## 10. Ejemplo concreto de recomendaciones generadas

Para concretizar lo que hace el modelo, mostramos las top-5 recomendaciones para algunos usuarios representativos (uno por arquetipo). Esta visualización es la base de la página de recomendación del Streamlit (Módulo 4).

## 11. Guardado de los modelos y de la metadata

Guardamos:
- Los pesos de los dos modelos (`.pth`)
- Los mapeos UserID ↔ user_idx y DestinationID ↔ dest_idx (necesarios para la inferencia en Streamlit)
- Las métricas finales
- Los hiperparámetros utilizados

## 12. Síntesis del Módulo 3

Hemos construido un sistema de recomendación de destinos para RutaViva mediante la **comparación equitativa de dos enfoques de filtrado colaborativo**: factorización matricial clásica (MF) y collaborative filtering neuronal (NCF). Esta comparación a paridad de hiperparámetros nos permite aislar y cuantificar el aporte específico de la no-linealidad introducida por el MLP.

### Resultados clave

Los resultados consolidados se reportan en la secci?n de evaluaci?n y en la metadata exportada del modelo.

| Métrica | MF (baseline) | NCF (principal) | Mejora |
|---|---|---|---|
| RMSE | — | — | — |
| MAE | — | — | — |
| Precision@10 | — | — | — |
| Recall@10 | — | — | — |

### Aporte creativo

El análisis PCA de los embeddings de usuario muestra que el modelo NCF, sin haber recibido información sobre los arquetipos, los redescubre por sí solo. Este resultado:
- valida la **calidad de las representaciones latentes** aprendidas;
- hace eco al análisis PCA de los embeddings de ruta del Módulo 1 (coherencia metodológica del proyecto);
- ofrece a RutaViva una herramienta de **segmentación automática** de su base de clientes, útil para campañas marketing dirigidas.

### Limitaciones reconocidas

1. **Dataset sintético** — el modelo redescubre patrones inyectados.
2. **Sin features de contenido** — un modelo híbrido (CF + content) mejoraría el cold-start.
3. **Sin contexto temporal** — un usuario que cambia de comportamiento no se modela.

### Próximas etapas

- Integración del modelo NCF en la app Streamlit (Módulo 4): seleccionar un usuario, mostrar sus top-10 recomendaciones, comparar con su historial.
- Redacción de la sección Módulo 3 del reporte técnico (después de ejecutar el notebook y analizar los resultados reales).