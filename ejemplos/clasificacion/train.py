import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = {
    "tiempo_minutos": [5, 8, 12, 2, 20, 15, 3],
    "paginas_visitadas": [3, 5, 7, 1, 10, 8, 2],
    "compro": [0, 0, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

x = df[["tiempo_minutos", "paginas_visitadas"]]
y = df["compro"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = LogisticRegression()  # ✅ instancia correcta
model.fit(x_train, y_train)   # ✅ entrenar modelo

y_pred = model.predict(x_test)  # ✅ predicción

print("Predicciones:", y_pred)
print("Entrada de prueba:", x_test.values)
