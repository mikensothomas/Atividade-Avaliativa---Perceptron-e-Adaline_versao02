import numpy as np
import pandas as pd

def step_function(x):
    return np.where(x >= 0, 1, -1)

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = step_function(linear_output)
                update = self.learning_rate * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = step_function(linear_output)
        return y_pred

class Adaline:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            linear_output = np.dot(X, self.weights) + self.bias
            errors = y - linear_output
            self.weights += self.learning_rate * np.dot(X.T, errors)
            self.bias += self.learning_rate * errors.sum()
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)

data = np.array([
    [-0.6508, 0.1097, 4.0009, -1],
    [-1.4492, 0.8896, 4.4005,-1],
    [2.0850, 0.6876, 12.0710, -1],
    [0.2626, 1.1476, 7.7985, 1],
    [0.6418, 1.0234, 7.0427, 1],
    [0.2569, 0.6730, 8.3265, -1],
    [1.1155, 0.6043, 7.4446, 1],
    [0.0914, 0.3399, 7.0677, -1],
    [0.0121, 0.5256, 4.6316, 1],
    [-0.0429, 0.4660, 5.4323, 1],
    [0.4340, 0.6870, 8.2287, -1],
    [0.2735, 1.0287, 7.1934, 1],
    [0.4839, 0.4851, 7.4850, -1],
    [0.4089, -0.1267, 5.5019, -1],
    [1.4391, 0.1614, 8.5843, -1],
    [-0.9115, -0.1973, 2.1962, -1],
    [0.3654, 1.0475, 7.4858, 1],
    [0.2144, 0.7515, 7.1699, 1],
    [0.2013, 1.0014, 6.5489, 1],
    [0.6483, 0.2183, 5.8991, 1],
    [-0.1147, 0.2242, 7.2435, -1],
    [-0.7970, 0.8795, 3.8762, 1],
    [-1.0625, 0.6366, 2.4707, 1],
    [0.5307, 0.1285, 5.6883, 1],
    [-1.2200, 0.7777, 1.7252, 1],
    [0.3957, 0.1076, 5.6623, -1],
    [-0.1013, 0.5989, 7.1812, -1],
    [2.4482, 0.9455, 11.2095, 1],
    [2.0149, 0.6192, 10.9263, -1],
    [0.2012, 0.2611, 5.4631, 1]

])

X = data[:, :-1]
y = data[:, -1]

perceptron = Perceptron(learning_rate=0.01, epochs=10)
perceptron.fit(X, y)
y_pred_perceptron = perceptron.predict(X)

adaline = Adaline(learning_rate=0.01, epochs=10)
adaline.fit(X, y)
y_pred_adaline = adaline.predict(X)

tabela_perceptron32 = pd.DataFrame({
    "x1": X[:, 0],
    "x2": X[:, 1],
    "x3": X[:, 2],
    "Classe Real": y,
    "Previsão (Perceptron)": y_pred_perceptron
})

tabela_adaline32 = pd.DataFrame({
    "x1": X[:, 0],
    "x2": X[:, 1],
    "x3": X[:, 2],
    "Classe Real": y,
    "Previsão (Adaline)": y_pred_adaline
})

print("Tabela - Resultados Perceptron:")
print(tabela_perceptron32)
print("\nTabela - Resultados Adaline:")
print(tabela_adaline32)

tabela_perceptron32.to_csv("tabela_resultados_perceptron.csv", index=False)
tabela_adaline32.to_csv("tabela_resultados_adaline.csv", index=False)