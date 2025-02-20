import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os

# Definir el archivo CSV
csv_file_path = 'csv/tarea1.csv'
columns = ['Criterion', 'Depth', 'Splitter', 'Min_to_Split', 'Node', 'Leaf', 'Train', 'Test']

# Crear el archivo si no existe
if not os.path.exists(csv_file_path):
    pd.DataFrame(columns=columns).to_csv(csv_file_path, index=False)

# Cargar conjunto de datos
train_data = pd.read_csv('csv/adult.csv')
test_data = pd.read_csv('csv/adult_test.csv')

X_train = train_data.drop(columns=['income'])
y_train = train_data['income']
X_test = test_data.drop(columns=['income'])
y_test = test_data['income']

# Convertir variables categóricas en X_train y X_test
for column in X_train.select_dtypes(include=['object']).columns:
    encoder = LabelEncoder()
    X_train[column] = encoder.fit_transform(X_train[column])
    X_test[column] = encoder.transform(X_test[column])

# Parámetros a iterar
Criterion = ["entropy"] # Criterion = ["gini", "entropy", "log_loss"]
Depth = list(range(1, 21)) + [None]
Splitter = list(range(98, 101)) #Splitter = list(range(50, 101))
Min_to_SplitLeaf = list(range(131, 151))

# Iterar sobre cada combinación de hiperparámetros
data = []
for criterio in Criterion:
    for split in Splitter:
        for min_split in Min_to_SplitLeaf:
            for prof in Depth:
                arbol = DecisionTreeClassifier(
                    criterion=criterio, max_depth=prof,
                    min_samples_split=split, min_samples_leaf=min_split
                )
                arbol.fit(X_train, y_train)

                # Obtener datos
                train_acc = arbol.score(X_train, y_train)
                test_acc = arbol.score(X_test, y_test)
                nodes = arbol.tree_.node_count
                leaves = arbol.tree_.n_leaves

                # Guardar en CSV
                new_data = pd.DataFrame([[criterio, prof, split, min_split, nodes, leaves, train_acc, test_acc]], columns=columns)
                new_data.to_csv(csv_file_path, mode='a', header=False, index=False)
