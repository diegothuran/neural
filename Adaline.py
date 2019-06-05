import numpy as np


class Adaline(object):
    """ADAptive LInear NEuron classifier

    Parameters
    ----------
    eta : float
        Taxa de aprendizagem (entre 0.0 and 1.0)
    n_iter : int
        número de épocas

    Attributes
    ----------
    weights : 1d-array
        pesos
    errors_ : list
        erros de classificação em cada época
    """

    def __init__(self, eta=0.01, n_iter=2000):
        self.eta = eta
        self.n_iter = n_iter

    def train(self, X, y):
        """Treinamento da rede

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Vetores de treinamento, em que n_samples é o número de amostras and n_features 
            é o número de características
        y : array-like, shape = [n_samples]
            Classes desejadas

        Returns
        -------
        self : object
        """

        self.weights = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.weights[1:] += self.eta * X.T.dot(errors)
            self.weights[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weights[1:] + self.weights[0])

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)
    
    

if __name__ == '__main__':
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([0, 1, 1, 1])

    p = Adaline()
    p.train(X, y)
    print(p.predict([0, 0]))