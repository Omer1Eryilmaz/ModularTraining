# Code for our meeting on the 15/06/2022. Forward mode scratch

import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Create data!
# Make deterministic
seed = 123
random.seed(seed)
np.random.seed(seed)

sample_size = 1000 # sample
D0 = 32 # Features
D1 = 512
D2 = 2
D3 = 10 # Class number
x = np.array(np.random.randn(sample_size,D0))
W0 = np.array(np.random.randn(D0,D1))
W1 = np.array(np.random.randn(D1,D2))
W2 = np.array(np.random.randn(D2,D3))

n_classes = 10 # Class number
n_examples = int(sample_size/n_classes)

y=[]
for i in range(n_classes):
    y += [i] * n_examples
y = np.random.permutation(y)
y = np.array(y)

def activationReLU(x):
    return np.maximum(0,x)

def tanhnorm(x):
    z = np.tanh(x)
    #Normalize the output of tanh (Project the Output on the unit circle)
    #z[z<1e-7&]=1e-7 # To avoid dividing by zero. 
    norm_tanh = np.zeros((1000,2))
    for i in range(len(x)):
        norm_tanh[i,0] = float(z[i,0])/float(math.sqrt(z[i,0]**2 + z[i,1]**2))
        norm_tanh[i,1] = float(z[i,1])/float(math.sqrt(z[i,0]**2 + z[i,1]**2))
    norm_tanh = norm_tanh
    return norm_tanh
# tanhnorm(f1(x,W0,W1))[0,0]**2 + tanhnorm(f1(x,W0,W1))[0,1]**2 = 1.0 (correct)    

def f1(x,W0,W1):
    return np.dot(activationReLU(np.dot(x,W0)),W1)

def k_mtrx(x,W0,W1):
    return np.dot(tanhnorm(f1(x,W0,W1)),tanhnorm(f1(x,W0,W1)).T) 

# K(f1(xi),f1(xj)) = tanhnorm(f1(xi))tanhnorm(f1(xj)).T
# Check the size of k_mtrx

def map_input(x,W0,W1):
    # Remove diagonals by changing 1s to -inf.
    return np.where(
        np.eye(len(x)) == 1,
        np.array(-float("inf")), 
        k_mtrx(x,W0,W1)
    )

k_min, k_max = -1., 1.
k_ideal = np.zeros((sample_size,sample_size))
for i in range(0,sample_size):
    for j in range (0,sample_size):
        if(y[i].item()!=y[j].item()):
            k_ideal[i,j] = k_min 
        else:
            k_ideal[i,j] = k_max

def srs_loss(x,W0,W1):
    xx = map_input(x,W0,W1)
    return np.mean(np.exp(xx[k_ideal == k_min]))

## Visuals from the tanh

#plt.scatter(tanhnorm(f1(x,W0,W1))[:,0],tanhnorm(f1(x,W0,W1))[:,1],c=y)
#plt.title(f"tanh_norm output {i}")
#plt.show()


def f2(x,W0,W1,W2):
    return np.dot(tanhnorm(f1(x,W0,W1)),W2)

def activationSoftmax(x):
    exp_values = np.exp(x- np.max(x, axis=1, keepdims=True))
    probabilities = exp_values/ np.sum(exp_values, axis = 1, keepdims=True)
    return probabilities
def Loss_CategoricalCrossentropy(x,y):
    samples = len(x)
    y_pred_clipped = np.clip(x, 1e-7,1-1e-7)

    if len(y.shape) == 1:
        correct_confidences = y_pred_clipped[range(samples),y]
    elif len(y.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y,
                axis=1
            )
    
    # Losses
    negative_log_likelihoods = -np.log(correct_confidences)
    return negative_log_likelihoods,correct_confidences


activation = f2(x,W0,W1,W2)
probabilities = activationSoftmax(activation)
negative_log_likelihoods,correct_confidences = Loss_CategoricalCrossentropy(probabilities,y)

#print(negative_log_likelihoods)
print(correct_confidences)