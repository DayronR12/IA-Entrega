import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generar datos ficticios
np.random.seed(0)
tamaño_casa = np.random.rand(100) * 100  # Tamaño en metros cuadrados
precio_casa = 300 + 15 * tamaño_casa + np.random.randn(100) * 10  # Precio en miles de dólares

# Convertir a DataFrame
data = pd.DataFrame({'Tamaño': tamaño_casa, 'Precio': precio_casa})

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data[['Tamaño']], data['Precio'], test_size=0.2, random_state=0)

# Crear y entrenar el modelo
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train, y_train)

# Predicción
y_pred = modelo_lineal.predict(X_test)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
print("Error cuadrático medio (MSE):", mse)

# Gráfica
plt.scatter(X_test, y_test, color='blue', label='Datos reales')
plt.plot(X_test, y_pred, color='red', label='Predicción')
plt.xlabel('Tamaño de la casa (m²)')
plt.ylabel('Precio de la casa (miles de $)')
plt.title('Regresión Lineal')
plt.legend()
plt.show()
