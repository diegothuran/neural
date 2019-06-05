import numpy as np

class Neuron(object):

    def __init__(self, n_entradas):
        self.pesos = np.zeros(n_entradas)


    def activation(self, entradas):
        soma = np.dot(entradas, self.pesos)
        if soma > 0:
            return 1
        else:
            return 0


if __name__ == '__main__':
    n = Neuron(2)

    input = [1, 0]

    print(n.activation(input))