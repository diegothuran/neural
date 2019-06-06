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

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)


net = network2.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
print(net.feedforward(test_data))

