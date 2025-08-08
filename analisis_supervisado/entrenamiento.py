import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Importar el Dataset con los datos de entrenamiento
df_datos_clientes = pd.read_csv("clientes_entrenamiento.csv")

# 2. Convertir el DataFrame a un Array de Numpy
X = df_datos_clientes.values

# 3. Entrenar el modelo
modelo = KMeans(n_clusters=3, random_state=96, n_init=10)
modelo.fit(X)

# 4. Análisis del modelo
df_datos_clientes['clouster'] = modelo.labels_
analisis = df_datos_clientes.groupby('clouster').mean()

print(analisis)

# Exportar el modelo
joblib.dump(modelo, "modelo_segmentacion_clientes.pkl")

# 5. Graficar los clusters
centroides = modelo.cluster_centers_
etiquetas = modelo.labels_

cluster0 = X[etiquetas == 0]
cluster1 = X[etiquetas == 1]
cluster2 = X[etiquetas == 2]

# 6. Colocar los puntos de cada cluster
plt.scatter(cluster0[:, 0], cluster0[:, 1], c='red', label='Clientes de temporada 0')
plt.scatter(cluster1[:, 0], cluster1[:, 1], c='blue', label='Clientes VIP 1')
plt.scatter(cluster2[:, 0], cluster2[:, 1], c='green', label='Clientes de ofertas 2')

# 7. Colocar los centroides de cada cluster
plt.scatter(centroides[:, 0], centroides[:, 1], c='black', marker='X', s=100, label='Centroides')

# 8. Colocar Títulos y etiquetas
plt.title('Segmentación de clientes')
plt.xlabel('Gasto total')
plt.ylabel('Frecuencia de compra')
plt.legend()
plt.grid(True)

# 9. Guardar gráfico
plt.savefig('graficas/graficas.png')
plt.show()