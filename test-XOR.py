"""

Obs: Este script é baseado na versão do livro http://neuralnetworksanddeeplearning.com/, com a devida autorização do autor.

    Código de teste para diferentes configurações de redes neurais.
    Adaptado para o Python 3.6

    Uso no shell:
         python test.py

    Parâmetros de rede:
         2º param é contagem de épocas
         O terceiro param é tamanho do lote
         4º param é a taxa de aprendizado (eta)
"""

# Imports
import mnist_loader
import network2
import numpy as np

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([1, 0, 0, 1])
training_data = list(zip(X, y))



net = network2.Network([2, 2, 1])
net.SGD(training_data, 1000, 10, 0.01)
print(net.feedforward(X[0]))

