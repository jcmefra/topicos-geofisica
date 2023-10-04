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

# Iterar a través de los cuatro conjuntos de índices y crear y visualizar árboles de decisión
for i, indices in enumerate(indices_sets):
    subset_df = df[df['id'].isin(indices)]
    X = subset_df[['x0', 'x1']]  # Cambia las características según el conjunto
    y = subset_df['y']

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    plt.figure(figsize=(10, 6))
    plot_tree(clf, filled=True, feature_names=list(X.columns), class_names=['y=0', 'y=1'])
    plt.title(f"Árbol de Decisión {i + 1}")
    plt.show()
