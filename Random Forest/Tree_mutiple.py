import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Crear el DataFrame original
data = {
    'id': [0, 1, 2, 3, 4, 5],
    'x0': [4.3, 6.5, 2.7, 6.6, 6.5, 2.7],
    'x1': [4.9, 4.1, 4.8, 4.4, 2.9, 6.7],
    'x2': [4.1, 5.9, 4.1, 4.5, 4.7, 4.2],
    'x3': [4.7, 5.5, 5.0, 3.9, 4.0, 5.3],
    'x4': [5.5, 5.9, 5.6, 5.9, 4.1, 4.8],
    'y': [0, 0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

# Definir los cuatro conjuntos de índices basados en 'id'
indices_sets = [
    [0, 2, 2, 5, 0, 3],
    [2, 4, 4, 2, 1, 0],
    [0, 1, 5, 2, 4, 0],
    [5, 0, 1, 3, 2, 3]
]

# Definir las características 'x' correspondientes a cada conjunto
feature_sets = [
    ['x0', 'x1'],
    ['x2', 'x3'],
    ['x2', 'x4'],
    ['x1', 'x3']
]

# Crear subplots para los árboles de decisión con menos espacio entre ellos
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)  # Ajustar el espacio entre subplots

for i, (indices, features) in enumerate(zip(indices_sets, feature_sets)):
    subset_df = df[df['id'].isin(indices)]
    x = subset_df[features]
    y = subset_df['y']

    clf = DecisionTreeClassifier()
    clf.fit(x, y)

    ax = axes[i // 2, i % 2]
    plot_tree(clf, filled=True, feature_names=list(x.columns), class_names=['y=0', 'y=1'], impurity=True, fontsize=10, ax=ax)
    ax.set_title(f"Árbol de Decisión {i + 1}\nCaracterísticas: {', '.join(features)}")

plt.tight_layout()
plt.show()

# Crear un DataFrame con el nuevo conjunto de datos
new_data = {
    'x0': [6.5],
    'x1': [4.7],
    'x2': [4.2],
    'x3': [1.3],
    'x4': [5.8]
}
new_df = pd.DataFrame(new_data)

# Inicializar una lista para almacenar las predicciones de los cuatro árboles
predictions = []

# Realizar una predicción para cada árbol
for i, (indices, features) in enumerate(zip(indices_sets, feature_sets)):
    subset_df = new_df[features]  # Utilizar solo las características relevantes para este árbol
    clf = DecisionTreeClassifier()
    clf.fit(df[df['id'].isin(indices)][features], df[df['id'].isin(indices)]['y'])
    prediction = clf.predict(subset_df)
    predictions.append(prediction[0])

# Imprimir las predicciones para cada árbol
for i, prediction in enumerate(predictions):
    print(f"Predicción Árbol {i + 1}: y={prediction}")
