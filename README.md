# Análisis Supervisado y No Supervisado

Este repositorio contiene ejemplos prácticos de aprendizaje supervisado y no supervisado utilizando Python y librerías populares como scikit-learn, pandas y matplotlib.

## Estructura del repositorio

```
Aprendizaje No Supervisado/
  01_kmeans/
    agrupacion.py
    clientes_entrenamiento.csv
    entrenamiento.py
    modelo_segmentacion_clientes.pkl
    requirements.txt
    graficas/
      clusters.png
  02_texto/
    noticias.csv
    noticias_entrnamiento.py
    modelo_entrenado.pkl
clasificacion/
    train.py
```

## Requisitos

Instala las dependencias necesarias ejecutando:

```sh
pip install -r "Aprendizaje No Supervisado/01_kmeans/requirements.txt"
```
Para el análisis de texto, también necesitas:
```sh
pip install pandas nltk scikit-learn joblib
```

## Aprendizaje No Supervisado

### Segmentación de clientes (KMeans)

En la carpeta [`Aprendizaje No Supervisado/01_kmeans`](Aprendizaje%20No%20Supervisado/01_kmeans):

- [`entrenamiento.py`](Aprendizaje%20No%20Supervisado/01_kmeans/entrenamiento.py): Entrena un modelo KMeans para segmentar clientes y guarda el modelo y una gráfica de los clusters.
- [`agrupacion.py`](Aprendizaje%20No%20Supervisado/01_kmeans/agrupacion.py): Usa el modelo entrenado para predecir el cluster de nuevos datos.
- [`clientes_entrenamiento.csv`](Aprendizaje%20No%20Supervisado/01_kmeans/clientes_entrenamiento.csv): Datos de entrenamiento.

#### Ejecución

Para entrenar el modelo y generar la gráfica:

```sh
cd "Aprendizaje No Supervisado/01_kmeans"
python entrenamiento.py
```

Para predecir el cluster de nuevos clientes:

```sh
python agrupacion.py
```

### Agrupamiento de noticias por texto

En la carpeta [`Aprendizaje No Supervisado/02_texto`](Aprendizaje%20No%20Supervisado/02_texto):

- [`noticias.csv`](Aprendizaje%20No%20Supervisado/02_texto/noticias.csv): Archivo con noticias (solo columna `Noticia`).
- [`noticias_entrnamiento.py`](Aprendizaje%20No%20Supervisado/02_texto/noticias_entrnamiento.py): Agrupa noticias por similitud de texto usando KMeans y muestra los resultados ordenados por cluster.

#### Ejecución

```sh
cd "Aprendizaje No Supervisado/02_texto"
python noticias_entrnamiento.py
```

## Aprendizaje Supervisado

En la carpeta [`clasificacion`](ejemplos/clasificacion):

- [`train.py`](ejemplos/clasificacion/train.py): Ejemplo de clasificación usando regresión logística.

### Ejecución

```sh
cd clasificacion
python train.py
```

## Notas

- Los modelos y gráficos generados se guardan automáticamente en sus respectivas carpetas.
- Asegúrate de tener Python 3.7+ instalado.

# Spider-Man Sentiment Analysis Project 🕷️

## Descripción 📄

Este proyecto realiza **análisis de sentimientos** sobre comentarios relacionados con Spider-Man (películas, actores, videojuegos, cómics) utilizando tres enfoques diferentes:

1. **Modelo supervisado** (Naive Bayes) entrenado con un dataset específico
2. **Análisis léxico** (VADER) basado en diccionario de palabras
3. **Modelo Transformer** (BETO para español) de última generación

## Características principales ✨

- 🎯 **Análisis por aspectos**: Detecta menciones a actores, películas, trajes, etc.
- 📊 **Visualización comparativa**: Gráficos que muestran resultados de los diferentes métodos
- 🤖 **Tecnologías avanzadas**: Combina modelos tradicionales con transformers
- 🕷️ **Temática especializada**: Optimizado para contenido sobre Spider-Man

## Requisitos 📋

```bash
pip install pandas numpy scikit-learn nltk matplotlib transformers torch
```

## Cómo usar 🚀

1. Clona el repositorio
2. Descarga el dataset `spiderman_comentarios.csv`
3. Ejecuta:

```bash
python analizador_spiderman.py
```

4. Ingresa tus comentarios cuando se solicite

## Estructura de archivos 📂

```
spiderman-sentiment-analysis/
├── analizador_spiderman.py    # Código principal
└──spiderman_comentarios.csv  # Dataset de entrenamiento
```

## Ejemplos de comentarios para probar 💡

```python
"Tom Holland dio vida al Peter Parker perfecto"
"Las escenas de acción en No Way Home son increíbles"
"El traje negro de Spider-Man 3 era innecesario"
"Los DLCs del juego de PS4 merecían más contenido"
```

## Licencia 📜

MIT License - JairVaz13

---

**"Un gran poder conlleva una gran responsabilidad"** - Tío Ben