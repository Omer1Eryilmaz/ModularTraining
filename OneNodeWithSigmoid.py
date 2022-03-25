# Adapted from cs231n lecture example
import math

w = [2,-3, -1, -2] # assume some random weights and data
x = [-1, -2, 3, 4]

# forward pass
dot = w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + w[3]*x[3]
print("dot product (forward pass):{}".format(dot))

f = 1.0 / (1 + math.exp(-dot)) # sigmoid function
print("Sigmoid Output:{}".format(f))

# backward pass (backpropagation)
ddot = (1 - f) * f # gradient on dot variable, using the sigmoid gradient derivation
print("Derivative of the sigmoid function (1-Sigmoid)*Sigmoid = {}".format(ddot))
dx = [w[0] * ddot, w[1] * ddot, w[2]* ddot, w[3]*ddot] # backprop into x
print("X gradients ={}".format(dx))

dw = [x[0] * ddot, x[1] * ddot, x[2] * ddot, x[3]*ddot] # backprop into w
print("W gradients={}".format(dw))


#OneNodeWithSigmoid.py