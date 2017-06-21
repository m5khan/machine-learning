import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy import optimize


def plot_decision_boundary(X, Z, W=None, b=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(X[:,0], X[:,1], c=Z, cmap=plt.cm.cool)
    ax.set_autoscale_on(False)

    a = - W[0, 0] / W[0, 1]
    xx = np.linspace(-30, 30)
    yy = a * xx - (b[0]) / W[0, 1]

    ax.plot(xx, yy, 'k-', c=plt.cm.cool(1.0/3.0))


def loadDataset(split, X=[] , XT=[], Z = [], ZT = []):
    dataset = datasets.load_iris()
    c = list(zip(dataset['data'], dataset['target']))
    np.random.seed(224)
    np.random.shuffle(c)
    x, t = zip(*c)
    sp = int(split*len(c))
    X = x[:sp]
    XT = x[sp:]
    Z = t[:sp]
    ZT = t[sp:]
    names = ['Sepal. length', 'Sepal. width', 'Petal. length', 'Petal. width']
    return np.array(X), np.array(XT), np.array(Z), np.array(ZT), names



def sigmoidFn(x):
    d = 1.0 + np.exp(-1.0 * x)
    return 1.0 / d

#Compute the probability of class 1 given the data and the parameters.
# arguments:
# X: data
# W: weight matrix, part of the parameters
# b: bias, part of the parameters
# returns:
# rate: probabiliy of the predicted class 1
def pred(X, W, b):
    l = X.dot(W.T) + b
    p = sigmoidFn(l)
    return p


# Compute the logarithm of the likelihood for logistic regression. The negative log-likelihood is our loss function.
# arguments:
# X: data
# Z: target
# W: weight matrix, part of the parameters
# b: bias, part of the parameters
# returns:
# log likelihood: logarithm of the likelihood
def loglikelihood(X, Z, W, b):
    y = pred(X, W, b)
    lny = np.log(y)
    ln1_y = np.log(1 - y)
    # ll = Z.T.dot(lny) + (1-Z).T.dot(ln1_y)
    zt = Z.reshape(y.shape[0], 1)
    ll = zt * lny + (1 - zt) * ln1_y
    return (ll.reshape(1, -1)[0])

# Compute the gradient of the loss with respect to the parameters
# arguments:
# * *X*: data
# * *Z*: target
# * *W*: weight matrix, part of the parameters
# * *b*: bias, part of the parameters
#
# returns:
# * *dLdW*: gradient of loss wrt to W
# * *dLdb*: gradient of loss wrt to b
def grad(X, Z, W, b):
    prob = pred(X, W, b)
    zt = Z.reshape(X.shape[0], 1)
    dldw = prob - zt
    dLdW = dldw.T.dot(X)
    # equal to the above dot product
    # dlw = (prob - zt) * X
    # dLdw = dlw.sum(axis=0)
    # print(dLdw)
    dLdb = (prob - zt).sum(axis=0)
    return dLdW, dLdb

# optimization using gradient descent
def optimizeWeights(X,Z,W,b):
    learning_rate = 0.001
    train_loss = []
    validation_loss = []

    for i in range(10000):
        dLdW, dLdb = grad(X, Z, W, b)

        W -= learning_rate * dLdW
        b -= learning_rate * dLdb
        train_loss.append(- loglikelihood(X, Z, W, b).mean())


if __name__ == "__main__":
    # prepare data
    split = 0.67
    X, XT, Z, ZT, names = loadDataset(split)

    # combine two of the 3 classes for a 2 class problem
    Z[Z==2] = 1
    ZT[ZT==2] = 1

    # only look at 2 dimensions of the input data for easy visualisation
    X = X[:,:2]
    XT = XT[:,:2]

    #-------------------------------

    W = np.random.randn(1, 2) * 0.01
    b = np.random.randn(1) * 0.01

    optimizeWeights(X,Z,W,b)

    #_ = plt.plot(train_loss)

    plot_decision_boundary(X, Z, W=W, b=b)

    #plot_decision_boundary(XT, ZT, W=W, b=b)

    plt.show()