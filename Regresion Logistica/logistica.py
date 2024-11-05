import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tkinter as tk
from tkinter import scrolledtext

# Generar datos ficticios
np.random.seed(1)
horas_estudio = np.random.rand(100) * 10  # Horas de estudio
resultado_examen = (horas_estudio > 5).astype(int)  # 1 si estudia más de 5 horas, 0 si no

# Convertir a DataFrame
data_log = pd.DataFrame({'Horas de Estudio': horas_estudio, 'Resultado del Examen': resultado_examen})

# Dividir en conjunto de entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    data_log[['Horas de Estudio']], data_log['Resultado del Examen'], test_size=0.2, random_state=1
)

# Crear y entrenar el modelo
modelo_logistico = LogisticRegression()
modelo_logistico.fit(X_entrenamiento, y_entrenamiento)

# Predicción
y_prediccion = modelo_logistico.predict(X_prueba)

# Calcular métricas para cada clase
precision_no_aprobado = precision_score(y_prueba, y_prediccion, pos_label=0)
recall_no_aprobado = recall_score(y_prueba, y_prediccion, pos_label=0)
f1_no_aprobado = f1_score(y_prueba, y_prediccion, pos_label=0)
support_no_aprobado = sum(y_prueba == 0)

precision_aprobado = precision_score(y_prueba, y_prediccion, pos_label=1)
recall_aprobado = recall_score(y_prueba, y_prediccion, pos_label=1)
f1_aprobado = f1_score(y_prueba, y_prediccion, pos_label=1)
support_aprobado = sum(y_prueba == 1)

# Precisión total del modelo
precision_total = accuracy_score(y_prueba, y_prediccion)

# Crear informe personalizado
informe = f"""
Precisión Total: {precision_total:.2f}

Informe de Clasificación:
               Precisión  Exhaustividad  F1-Score  Apoyo
No Aprobado      {precision_no_aprobado:.2f}         {recall_no_aprobado:.2f}         {f1_no_aprobado:.2f}       {support_no_aprobado}
Aprobado         {precision_aprobado:.2f}         {recall_aprobado:.2f}         {f1_aprobado:.2f}      {support_aprobado}
"""

# Función para mostrar los resultados en la interfaz
def mostrar_resultados():
    resultado_texto.delete(1.0, tk.END)  # Limpiar el área de texto
    resultado_texto.insert(tk.END, informe)

# Configuración de la interfaz con Tkinter
ventana = tk.Tk()
ventana.title("Resultados de la Regresión Logística")

# Establecer tamaño de la ventana
ventana.geometry("600x400")

# Etiqueta y botón para mostrar los resultados
etiqueta = tk.Label(ventana, text="Resultados de Evaluación del Modelo:", font=("Arial", 14))
etiqueta.pack(pady=10)

# Área de texto desplazable para los resultados
resultado_texto = scrolledtext.ScrolledText(ventana, width=70, height=15, font=("Courier New", 12))
resultado_texto.pack(pady=10)

# Botón para cargar los resultados en el área de texto
boton_mostrar = tk.Button(ventana, text="Mostrar Resultados", font=("Arial", 12), command=mostrar_resultados)
boton_mostrar.pack(pady=10)

# Ejecutar la aplicación
ventana.mainloop()
