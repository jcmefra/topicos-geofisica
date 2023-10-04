import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Crear un DataFrame desde los datos proporcionados
data = {
    'x0': [4.3, 6.5, 2.7, 6.6, 6.5, 2.7],
    'x1': [4.9, 4.1, 4.8, 4.4, 2.9, 6.7],
    'y': [0, 0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

# Dividir los datos en características (X) y etiquetas (y)
X = df[['x0', 'x1']]
y = df['y']

# Crear un clasificador de árbol de decisión
clf = DecisionTreeClassifier()

# Entrenar el modelo en los datos
clf.fit(X, y)

# Visualizar el árbol de decisión con solo la clase y las características 'x'
plt.figure(figsize=(10, 6))
plot_tree(clf, filled=True, feature_names=list(X.columns), class_names=['y=0', 'y=1'], impurity=False, fontsize=10)
plt.title("Árbol de Decisión")
plt.show()
