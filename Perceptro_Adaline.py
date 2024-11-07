import numpy as np
import pandas as pd

# Parâmetros de configuração
learning_rate = 0.01  # Taxa de aprendizagem
epochs = 1000         # Número máximo de épocas para o treinamento

# Função de ativação degrau para o Perceptron
def step_function(x):
    return 1 if x >= 0 else -1

# Função de ativação linear para o Adaline
def linear_function(x):
    return x

# Perceptron: Treinamento com a regra de Hebb
def train_perceptron(X, y):
    weights = np.random.rand(X.shape[1] + 1)  # Inicializa pesos aleatórios
    initial_weights = weights.copy()  # Guarda pesos iniciais
    for epoch in range(epochs):
        for i, x in enumerate(X):
            x_with_bias = np.insert(x, 0, 1)  # Adiciona o bias (w0)
            net_input = np.dot(weights, x_with_bias)
            prediction = step_function(net_input)
            error = y[i] - prediction
            weights += learning_rate * error * x_with_bias  # Atualiza os pesos
        # Critério de parada para convergência
        if np.all(np.vectorize(step_function)(np.dot(np.insert(X, 0, 1, axis=1), weights)) == y):
            break
    return initial_weights, weights, epoch + 1  # Retorna pesos iniciais, finais e número de épocas

# Adaline: Treinamento com minimização do erro quadrático
def train_adaline(X, y):
    weights = np.random.rand(X.shape[1] + 1)  # Inicializa pesos aleatórios
    initial_weights = weights.copy()  # Guarda pesos iniciais
    for epoch in range(epochs):
        errors = []
        for i, x in enumerate(X):
            x_with_bias = np.insert(x, 0, 1)  # Adiciona o bias (w0)
            net_input = np.dot(weights, x_with_bias)
            prediction = linear_function(net_input)
            error = y[i] - prediction
            weights += learning_rate * error * x_with_bias  # Atualiza os pesos
            errors.append(error ** 2)
        if np.mean(errors) < 0.01:  # Critério de parada
            break
    return initial_weights, weights, epoch + 1  # Retorna pesos iniciais, finais e número de épocas

# Função de previsão para ambos os modelos
def predict(X, weights, model_type="perceptron"):
    predictions = []
    for x in X:
        x_with_bias = np.insert(x, 0, 1)
        net_input = np.dot(weights, x_with_bias)
        if model_type == "perceptron":
            predictions.append(step_function(net_input))
        elif model_type == "adaline":
            predictions.append(linear_function(net_input))
    return np.array(predictions)

# Conjunto de treinamento - substituir pelos valores reais do apêndice I
X_train = np.array([
    [0.5, 0.2, 0.1],
    [0.9, 0.4, 0.7],
    [0.4, 0.6, 0.5],
    [0.6, 0.7, 0.2]
])

y_train = np.array([-1, 1, -1, 1])

# Conjunto de teste - substituir pelos valores reais da Tabela 3.3
X_test = np.array([
    [0.5, 0.3, 0.8],
    [0.2, 0.6, 0.1],
    [0.9, 0.7, 0.5]
])

# Armazenar resultados para preenchimento da Tabela 3.2
results_perceptron = []
results_adaline = []

# Treinamento para preencher Tabela 3.2
for i in range(5):
    # Perceptron
    initial_weights_perceptron, final_weights_perceptron, epochs_perceptron = train_perceptron(X_train, y_train)
    results_perceptron.append((initial_weights_perceptron, final_weights_perceptron, epochs_perceptron))
    
    # Adaline
    initial_weights_adaline, final_weights_adaline, epochs_adaline = train_adaline(X_train, y_train)
    results_adaline.append((initial_weights_adaline, final_weights_adaline, epochs_adaline))

# Tabelas 3.2
table_3_2_perceptron = pd.DataFrame({
    "Treinamento": [f"T{i+1}" for i in range(5)],
    "Pesos Iniciais (w0, w1, w2, w3)": [result[0] for result in results_perceptron],
    "Pesos Finais (w0, w1, w2, w3)": [result[1] for result in results_perceptron],
    "Número de Épocas": [result[2] for result in results_perceptron]
})

table_3_2_adaline = pd.DataFrame({
    "Treinamento": [f"T{i+1}" for i in range(5)],
    "Pesos Iniciais (w0, w1, w2, w3)": [result[0] for result in results_adaline],
    "Pesos Finais (w0, w1, w2, w3)": [result[1] for result in results_adaline],
    "Número de Épocas": [result[2] for result in results_adaline]
})

# Preenchimento da Tabela 3.3
predictions_perceptron = [predict(X_test, result[1], model_type="perceptron") for result in results_perceptron]
predictions_adaline = [predict(X_test, result[1], model_type="adaline") for result in results_adaline]

table_3_3_perceptron = pd.DataFrame({
    "Amostra": [f"Amostra {i+1}" for i in range(len(X_test))],
    **{f"y (T{i+1})": predictions_perceptron[i] for i in range(5)}
})

table_3_3_adaline = pd.DataFrame({
    "Amostra": [f"Amostra {i+1}" for i in range(len(X_test))],
    **{f"y (T{i+1})": predictions_adaline[i] for i in range(5)}
})

# Exibição das Tabelas
print("Tabela 3.2 - Resultados Perceptron")
print(table_3_2_perceptron)

print("\nTabela 3.2 - Resultados Adaline")
print(table_3_2_adaline)

print("\nTabela 3.3 - Classificação das Amostras (Perceptron)")
print(table_3_3_perceptron)

print("\nTabela 3.3 - Classificação das Amostras (Adaline)")
print(table_3_3_adaline)