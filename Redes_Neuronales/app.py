import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Datos
numeros = np.array([8, 4, 5, 6, 3, 10, 15, 78, 35, 42], dtype=float)
par_impar = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)

# Definir capa y modelo
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

# Compilar modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# Entrenar modelo y guardar historial
historial = modelo.fit(numeros, par_impar, epochs=500, verbose=True)

# Resumen del modelo
modelo.summary()

# Graficar la pérdida
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.plot(historial.history['loss'])
plt.savefig('perdida.png')

