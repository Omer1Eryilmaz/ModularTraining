# Compare 2 backpropagation code
## Use NNFS spiral data with 2 classes
from re import L
from this import d
from urllib.parse import SplitResult
import numpy as np

seed = 123
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
    y += [i] * n_examples # Create balance labelling
y = np.random.permutation(y)
y = np.array(y)


class activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    def backward(self,dvalue):
        self.dinputs = dvalue.copy() # We do not want to remove negative values on the original gradient matrix.
        self.dinputs[self.inputs < 0]=0 # But only pass the the possitive values to the following layers.

class activation_Tanh:
    # Equal-norm condition required by Theorem 1 in (Modularizing Deep Learning via Pairwise Learning With Kernels)
    def forward(self,inputs):
        z = np.tanh(inputs)
        norm_tanh = np.zeros((z.shape)) # Normalize to unit vector.
        for i in range(len(x)):
            norm_tanh[i,0] = float(z[i,0])/float(np.sqrt(z[i,0]**2 + z[i,1]**2))
            norm_tanh[i,1] = float(z[i,1])/float(np.sqrt(z[i,0]**2 + z[i,1]**2))
        
        self.output = norm_tanh
    def backward(self,dvalues):
        self.dinputs = 1-np.tanh(dvalues)**2 # Derivative of tanh function 

class DenseLayer:
    def __init__(self,n_inputs,n_neurons):
        self.weigths=0.001* np.random.randn(n_inputs,n_neurons) #Randomly initilized weights
        self.biases= np.zeros((1,n_neurons)) # Initially 0 biases
    def forward(self,inputs):
        self.inputs = inputs # To remember the inputs we assign them self.inputs We use them in backpropogation.
        self.output = np.dot(inputs,self.weigths)+self.biases
        
    def backward(self,dvalues):
        self.dweigths = np.dot(self.inputs.T, dvalues) # 1st dvalues will be the output of loss function.
        self.dbiases = np.sum(dvalues,axis=0,keepdims=True) # Why do we sum column of gradients for biases?
        self.dinputs = np.dot(dvalues,self.weigths.T) # dinputs are the dvalues for the following layers.

# A general loss function class takes the prediction and labels then gives the average loss as the data loss.

class SRS_Loss:
    def calculate(self,output,y):
        data_loss = self.forward(output,y)
        return data_loss
    def forward(self,y_pred,y_true):
        sample_size = len(y_pred)
        # Create a symmetric kernel matrix
        kernel_y_pred = np.dot(y_pred,y_pred.T)
        # Change to diagonal values (1s) to -inf because e^-inf = 0 in the loss function. 
        #kernel_y_pred = np.where(np.eye(len(y_pred)) == 1, np.array(-float("inf")), kernel_y_pred)
    
        #k_min, k_max = -1., 1. # extreemes of tanh function
        NegativeOnlyLossMatrix = np.zeros((sample_size,sample_size))
        for i in range(0,sample_size):
            for j in range(0,sample_size):
                if(y_true[i].item()!=y_true[j].item()):
                    NegativeOnlyLossMatrix[i,j] =  kernel_y_pred[i,j] #Values
                else:
                    NegativeOnlyLossMatrix[i,j] = 0
        
        self.output = -np.mean(np.exp(NegativeOnlyLossMatrix))#CTS Loss
        
    def backward(self,dvalues,y_true):
        #sample_size = len(dvalues) # Why the size of dvalues are 900000?
        sample_size = 1000
        #Create a mask to remove samples from the same classes. We only consider distinct classes for loss calculation
        NegativeOnlyGradientMatrix = np.zeros((sample_size,sample_size))
        kernel_dvalues = np.dot(dvalues,dvalues.T)
        for i in range(0,sample_size):
            for j in range(0,sample_size):
                if(y_true[i].item()!=y_true[j].item()):
                    NegativeOnlyGradientMatrix[i,j] = kernel_dvalues[i,j] #gradients
                else:
                    NegativeOnlyGradientMatrix[i,j] = 0
        # Calculate gradient
        # derivative of exponential tahn x with respect to x. d exp(tanh(x))/dx
        self.dinputs = - np.mean(((1 - NegativeOnlyGradientMatrix ** 2) * np.exp(NegativeOnlyGradientMatrix)))
        

LayerClass1 = DenseLayer(D0,D1)
LayerClass1.forward(x)
activation1 = activation_ReLU() # () is important!
activation1.forward(LayerClass1.output)
LayerClass2 = DenseLayer(D1,D2)
LayerClass2.forward(activation1.output)
activation2 = activation_Tanh()
activation2.forward(LayerClass2.output)
#print(activation2.output[:3]) # Get the same outputs
loss_SRS = SRS_Loss()
#print(activation2.output.shape)
loss_SRS.forward(activation2.output,y)
#print(loss_SRS.output) # Returns a scaler value

# activation2.output.shape is (1000,2)
# 
loss_SRS.backward(activation2.output,y)
activation2.backward(loss_SRS.dinputs)
LayerClass2.backward(activation2.dinputs) # returns a scalar value
activation1.backward(LayerClass2.dinputs)
LayerClass1.backward(activation1.dinputs)


print(LayerClass1.dweigths)
#plt.scatter(tanhnorm(f1(x,W0,W1))[:,0],tanhnorm(f1(x,W0,W1))[:,1],c=y)
#plt.title(f"tanh_norm output {i}")
#plt.show()
