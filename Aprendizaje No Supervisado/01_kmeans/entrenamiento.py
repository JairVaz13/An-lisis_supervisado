import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  

# 1. Importar el dataset con los datos de entrenamiento
df_datos_clientes = pd.read_csv("clientes_entrenamiento.csv")

# 2. Mostrar los primeros registros del DataFrame
print(df_datos_clientes.head())

# 3. Mostrar información general del DataFrame
print(df_datos_clientes.info())

# 4. Seleccionar solo columnas numéricas y convertir a array de NumPy
x = df_datos_clientes.select_dtypes(include=[np.number]).values
print(x)

# 5. Entrenamiento del modelo
modelo = KMeans(n_clusters=3, random_state=1234, n_init=10)
modelo.fit(x)

# 6. Análisis del modelo
df_datos_clientes["cluster"] = modelo.labels_
analisis = df_datos_clientes.groupby("cluster").mean()
print(analisis)

