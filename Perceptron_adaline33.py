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

X = data[:, :-1]
y = data[:, -1]

perceptron = Perceptron(learning_rate=0.01, epochs=10)
perceptron.fit(X, y)

adaline = Adaline(learning_rate=0.01, epochs=10)
adaline.fit(X, y)

new_data = np.array([
    [-0.3665, 0.0620, 5.9891],
    [-0.7842, 1.1267, 5.5912],
    [0.3012, 0.5611, 5.8234],
    [0.7757, 1.0648, 8.0677],
    [0.1570, 0.8028, 6.3040],
    [-0.7014, 1.0316, 3.6005],
    [0.3748, 0.1536, 6.1537],
    [-0.6920, 0.9404, 4.4058],
    [-1.3970, 0.7141, 4.9263],
    [-1.8842, -0.2805, 1.2548]
])

new_y_pred_perceptron = perceptron.predict(new_data)
new_y_pred_adaline = adaline.predict(new_data)

tabela_novos_perceptron33 = pd.DataFrame({
    "x1": new_data[:, 0],
    "x2": new_data[:, 1],
    "x3": new_data[:, 2],
    "y (Perceptron)": new_y_pred_perceptron
})

tabela_novos_adaline33 = pd.DataFrame({
    "x1": new_data[:, 0],
    "x2": new_data[:, 1],
    "x3": new_data[:, 2],
    "y (Adaline)": new_y_pred_adaline
})

print("Tabela Novos Dados - Perceptron:")
print(tabela_novos_perceptron33)
print("\nTabela Novos Dados - Adaline:")
print(tabela_novos_adaline33)

tabela_novos_perceptron33.to_csv("tabela_novos_perceptron.csv", index=False)
tabela_novos_adaline33.to_csv("tabela_novos_adaline.csv", index=False)