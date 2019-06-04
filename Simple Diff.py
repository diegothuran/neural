from mlxtend.data import iris_data
X, y = iris_data()
X = X[:, [0, 3]]
# standardize training data
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

from mlxtend.classifier import MultiLayerPerceptron as MLP
nn1 = MLP(hidden_layers=[50],
l2=0.00,
l1=0.0,
epochs=150,
eta=0.05,
momentum=0.1,
decrease_const=0.0,
minibatches=1,
random_seed=1,
print_progress=3)
nn1 = nn1.fit(X_std, y)


from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
fig = plot_decision_regions(X=X_std, y=y, clf=nn1, legend=2)
plt.title('Multi-layer Perceptron w. 1 hidden layer (logistic sigmoid)')
plt.show()
import matplotlib.pyplot as plt
plt.plot(range(len(nn1.cost_)), nn1.cost_)
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()