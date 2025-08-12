import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import nltk
from collections import defaultdict
import warnings

# Configuración inicial
warnings.filterwarnings('ignore')
nltk.download('vader_lexicon')

# Cargar dataset
try:
    df = pd.read_csv('spiderman_comentarios.csv')
    comentarios = df['comentario'].tolist()
    etiquetas = df['etiqueta'].tolist()
except FileNotFoundError:
    print("Error: El archivo 'spiderman_comentarios.csv' no se encontró.")
    exit()

# Preprocesamiento y vectorización
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(comentarios)

# Entrenamiento del modelo supervisado
clf = MultinomialNB()
clf.fit(X, etiquetas)

# Inicializar analizadores
sia = SentimentIntensityAnalyzer()

try:
    analyzer_transformer = pipeline("sentiment-analysis", 
                                  model="finiteautomata/beto-sentiment-analysis")
except Exception as e:
    print(f"Error al cargar el modelo transformer: {e}")
    analyzer_transformer = None

# Diccionario de aspectos
aspectos = {
    'actores': ['tobey', 'maguire', 'andrew', 'garfield', 'tom', 'holland', 'zendaya', 'kirsten', 'dunst', 'emma', 'stone'],
    'peliculas': ['homecoming', 'far from home', 'no way home', 'spiderman', 'amazing', 'civil war', 'spider-verse'],
    'trajes': ['traje', 'diseño', 'iron spider', 'classic', 'suit'],
    'juegos': ['ps4', 'insomniac', 'game', 'juego', 'dlc'],
    'comics': ['cómic', 'historieta', 'marvel', 'stan lee', 'ditko', 'kraven', 'clone saga']
}

# Funciones de análisis
def analizar_sentimiento_supervisado(texto):
    texto_vectorizado = vectorizer.transform([texto])
    return clf.predict(texto_vectorizado)[0]

def analizar_sentimiento_no_supervisado(texto):
    scores = sia.polarity_scores(texto)
    if scores['compound'] >= 0.05:
        return "positivo"
    elif scores['compound'] <= -0.05:
        return "negativo"
    return "neutral"

def analizar_con_transformer(texto):
    if analyzer_transformer is None:
        return "Modelo no disponible"
    resultado = analyzer_transformer(texto)[0]
    sentimiento = resultado['label'].lower()
    return 'positivo' if sentimiento == 'positive' else 'negativo' if sentimiento == 'negative' else 'neutral'

def analizar_aspectos(texto):
    texto_lower = texto.lower()
    resultados = defaultdict(list)
    for aspecto, palabras in aspectos.items():
        for palabra in palabras:
            if palabra in texto_lower:
                sentimiento = analizar_sentimiento_no_supervisado(texto)
                resultados[aspecto].append((palabra, sentimiento))
    return dict(resultados)

def mostrar_grafico(resultados):
    methods = ['Supervisado', 'No Supervisado', 'Transformer']
    sentimientos = {'positivo': 0, 'neutral': 1, 'negativo': 2}
    data = np.zeros((3, 3))
    
    if resultados['transformer'] == "Modelo no disponible":
        methods = methods[:2]
        data = data[:2, :]
    
    for i, metodo in enumerate(['supervisado', 'no_supervisado', 'transformer']):
        if metodo == 'transformer' and resultados[metodo] == "Modelo no disponible":
            continue
        data[i, sentimientos[resultados[metodo].lower()]] = 1
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4CAF50', '#2196F3', '#F44336']
    labels = ['Positivo', 'Neutral', 'Negativo']
    
    bottom = np.zeros(len(methods))
    for idx, (color, label) in enumerate(zip(colors, labels)):
        ax.bar(methods, data[:, idx], label=label, color=color, bottom=bottom)
        bottom += data[:, idx]
    
    ax.set_title('Comparación de Métodos de Análisis', pad=20)
    ax.set_ylabel('Resultado')
    ax.legend(title='Sentimiento')
    plt.tight_layout()
    plt.show()

# Interfaz mejorada
def interfaz_analizador():
    print("🕷️" * 20)
    print("=== ANALIZADOR DE SENTIMIENTOS SPIDER-MAN ===")
    print("🕷️" * 20)
    print("\nEscribe tus opiniones sobre Spider-Man (películas, actores, juegos, cómics)")
    print("Presiona 'q' para salir\n")
    
    while True:
        comentario = input("🚀 Tu comentario: ")
        
        if comentario.lower() == 'q':
            break
            
        if not comentario.strip():
            print("Por favor, ingresa un comentario válido.\n")
            continue
            
        # Realizar análisis
        resultados = {
            'supervisado': analizar_sentimiento_supervisado(comentario),
            'no_supervisado': analizar_sentimiento_no_supervisado(comentario),
            'transformer': analizar_con_transformer(comentario)
        }
        
        aspectos_detectados = analizar_aspectos(comentario)
        
        # Mostrar resultados
        print("\n🔎 Resultados:")
        print(f"- 🧠 Modelo Supervisado (Naive Bayes): {resultados['supervisado']}")
        print(f"- 📊 Análisis Léxico (VADER): {resultados['no_supervisado']}")
        print(f"- 🤖 Modelo Transformer: {resultados['transformer']}")
        
        if aspectos_detectados:
            print("\n🎯 Aspectos detectados:")
            for aspecto, detalles in aspectos_detectados.items():
                print(f"  - {aspecto.upper()}:")
                for palabra, sentimiento in detalles:
                    print(f"    · '{palabra}' → {sentimiento}")
        
        mostrar_grafico(resultados)
        print("\n" + "🕸️" * 40 + "\n")
    
    print("\n¡Gracias por usar el analizador! 'Un gran poder conlleva una gran responsabilidad'")

if __name__ == "__main__":
    interfaz_analizador()