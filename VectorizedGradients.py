import numpy as np

W = np.random.randn(5,10)
X = np.random.randn(10,3)
D = W.dot(X)
print("Forward pass = {}".format(D))
dD = np.random.randn(*D.shape) # The Gradient D From the following node.
dW = dD.dot(X.T)
print("Gradients of Ws = {}".format(dW))
dX = W.T.dot(dD)
print("Gradients of Xs = {}".format(dX))

