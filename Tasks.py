"""
Task 1: Implement a full two-layer pipeline
Task 2: Split into two separate layers according to the paper
Task 3: Implement cross entropy loss
Task 4: Implement CTS-NEO/SRS loss
Task 5: Demonstrate end-to-end cross entropy loss training for full two-layer pipeline

Task 5a: Find a suitable AD toolbox
Task 5b: Test that toolbox does what you expect on simple examples
Task 5c: Apply toolbox to your two-layer pipeline
Task 5d: Implement ADAM optimizer
ADAM - SGD with momentum?

Task 6: Demonstrate layer-wise training of split pipeline

"""

# Create Data
from locale import normalize
import os
import random
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
# Make deterministic
seed = 123
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

device = "cpu"
dtype = torch.float32

N=1000
D0 = 32

x = torch.randn(N,D0)

K = 10


# specify device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# make data
n_classes = 10
# balanced = False

sample_size = 1000
x = torch.rand(sample_size, 32).to(device)

    # random, but balanced, labels
if sample_size % n_classes: 
    raise('n_classes does not divide sample_size')
n_examples = int(sample_size / n_classes)
y = []
for i in range(n_classes):
    y += [i] * n_examples
y = np.random.permutation(y)
y = torch.tensor(y).to(device)

y_numpy = y.cpu().numpy().astype(np.int)
ylong = y.type(torch.LongTensor)


# Check the data type
#print(type(x))
#print(type(y))

# ***** Task1: Implement a full two-layer pipeline *****
# Two layer input module has been created.


D0 = 32
D1 = 512
D2 = 2
# Initialize weights 
W0u = torch.randn(D0,D1,dtype=dtype,device=device,requires_grad=True)
W1u = torch.randn(D1,D2,dtype=dtype,device=device,requires_grad=True)

W0 = W0u.clone().detach()
W1 = W1u.clone().detach()

# print(W0.size())
# print(W1.size())

def ReLU(x):
    return torch.maximum(torch.tensor(0),torch.tensor(x))
def F1(x,W0,W1):
    return torch.mm(ReLU(np.dot(x,W0)),W1)

#print(F1(x,W0,W1).size())

# ******* Taks2: Split into two separate layers according to the paper *******
# Subtask: Create the output module

D3 = 10 # Class number for classification

W2u = torch.randn(D2,D3, dtype=dtype, device=device, requires_grad=True)

W2 = W2u.clone().detach().double()
#Check the normalization.
def TanHnorm(x):
    z = torch.tanh(x)
    norm_matrix_unit = np.zeros((1000,2))
    for i in range(len(x)):
        norm_matrix_unit[i,0] = float(x[i,0]) / float(math.sqrt(x[i,0]**2 + x[i,1]**2))
        norm_matrix_unit[i,1] = float(x[i,1]) / float(math.sqrt(x[i,0]**2 + x[i,1]**2))
    return torch.tensor(norm_matrix_unit).double()

#new_repr = TanHnorm(F1(x,W0,W1))

#plt.scatter(new_repr[:,0],new_repr[:,1],c=y_numpy)
#plt.show()

def F2(x,W0,W1,W2):
    return torch.mm(TanHnorm(F1(x,W0,W1)),W2)


#print(Output.size())


#exp_output = torch.exp(Output - M)
#print(Output)


# ******** Task 3: Implement cross entropy loss ***********
def SoftMaxActivation(Output):
    Output = Output.numpy()
    exp_output = np.exp(Output-np.max(Output, axis=1, keepdims=True))
    probability = exp_output/ np.sum(exp_output, axis=1, keepdims= True)
    return probability
    
def CrossEntropyLoss(Output,y_numpy):
    Output = SoftMaxActivation(Output)
    Output = np.clip(Output, 1e-7, 1-1e-7)
    Samples = len(Output) 
    correct_confidences = Output[range(Samples),y_numpy]
    neg_log_likelihoods = -np.log(correct_confidences)

    return Output,neg_log_likelihoods

#Output, Loss = CrossEntropyLoss(F2(x,W0,W1,W2),y_numpy)
#predictions = np.argmax(Output,axis=1)
#print(predictions)
#accuracy = np.mean(predictions==y_numpy)
#print(accuracy)

# Task 4: Implement CTS-NEO/SRS loss

def k_mtrx(x):
    xx = torch.mm(x,x.t())
    # Remove diagonal because they are same.
    return torch.where(
        torch.eye(len(x)).to(x.device) == 1,
        torch.tensor(-float("inf")).to(x.device),
        xx
        )

# Expected output
#print(k_mtrx(TanHnorm(F1(x,W0,W1)).float()))

k_min, k_max = -1., 1.
k_ideal = torch.zeros(N,N)
for i in range(0,N):
    for j in range(0,N):
        if (y[i].item() != y[j].item()):
            k_ideal[i,j] = k_max
        else:
            k_ideal[i,j] = k_min

#print(k_ideal[:10,:10])

def CTS_NEO(x):
    xx=k_mtrx(x)
    return torch.mean(torch.exp(xx[k_ideal == k_min]))

#print(CTS_NEO(TanHnorm(F1(x,W0,W1)).float()))

#  **** Task 5: Demonstrate end-to-end cross entropy loss training for full two-layer pipeline

optimizer = torch.optim.Adam([W0u,W1u,W2u],lr=1e-3)

# %%
Rmax = 10
Rb = 1
lnew = float("inf")

for i in range(Rmax):
    loss = CrossEntropyLoss(F2(x,W0,W1,W2),y_numpy)
    if (i % Rb) == 0:
        lold = lnew
        lnew = loss.item()
        new_repr = TanHnorm(F1(x,W0,W1))
        plt.scatter(new_repr[:,0],new_repr[:,1],c=y_numpy)
        plt.title(f"e2e repr., epoch {i}")
        plt.show()
        
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()







