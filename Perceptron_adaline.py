import numpy as np
import pandas as pd

# Definindo a função de ativação para o Perceptron (degrau)
def step_function(x):
    return np.where(x >= 0, 1, -1)

# Definindo o Perceptron
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

# Definindo o Adaline
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

# Carregar dados da imagem em um array numpy (amostra dos dados)
data = np.array([
    [-0.6508, 0.1097, 4.0009, -1],
    [-1.4492, 0.8896, 4.4005, -1],
    [2.0850, 0.6876, 12.0710, 1],
    [0.2626, 1.1476, 7.7985, 1],
    [0.6418, 1.0234, 7.0427, 1],
    [0.2569, 0.6730, 8.3265, -1],
    [1.1155, 0.6043, 7.4446, 1],
    [0.0914, 0.3399, 7.0677, -1],
    [0.0121, 0.5256, 4.6316, -1],
    [0.4887, 0.8318, 5.8085, -1],
    [1.2217, 0.4822, 8.8573, 1],
    [0.6880, 0.6765, 7.4969, -1],
    [0.3557, 0.0222, 5.5214, 1],
    [0.6271, 0.2212, 5.5378, 1],
    [0.2455, 0.9313, 6.6924, 1]
])

# Dividir em características (X) e rótulos (y)
X = data[:, :-1]
y = data[:, -1]

# Instanciar e treinar o Perceptron
perceptron = Perceptron(learning_rate=0.01, epochs=10)
perceptron.fit(X, y)
y_pred_perceptron = perceptron.predict(X)

# Instanciar e treinar o Adaline
adaline = Adaline(learning_rate=0.01, epochs=10)
adaline.fit(X, y)
y_pred_adaline = adaline.predict(X)

# Criar DataFrames para resultados e exibir ou salvar
results_perceptron = pd.DataFrame({"y_true": y, "y_pred_perceptron": y_pred_perceptron})
results_adaline = pd.DataFrame({"y_true": y, "y_pred_adaline": y_pred_adaline})

# Exibir os resultados
print("Resultados Perceptron:")
print(results_perceptron)
print("\nResultados Adaline:")
print(results_adaline)

# Opcional: salvar os resultados em CSV
results_perceptron.to_csv("resultados_perceptron.csv", index=False)
results_adaline.to_csv("resultados_adaline.csv", index=False)