import numpy as np
from sympy import arg
import math
import matplotlib.pyplot as plt

# The basic building block of an AD calculator: the input variable w

class ADTuple:
    def __init__(self, **kwargs) -> None:
        
        self.value=0
        self.deriv=1

        if 'value' in kwargs:
            self.value = kwargs['value']
        if 'deriv' in kwargs:
            self.deriv = kwargs['deriv']
    
def sin(a): # When we do not put self as input a.deriv is recognized why?
    b = ADTuple()
    b.value = np.sin(a.value)
    b.deriv = np.cos(a.value)*a.deriv
    return b

g = lambda w: sin(w)

a1 = ADTuple(value = 0)
a2 = ADTuple(value = 0.5)

result1 = g(a1)
result2 = g(a2)

print('function value at' + str(0) + '=' + str(result1.value))
print('derivative value at' + str(0) + '=' + str(result1.deriv))

print('function value at' + str(0.5) + '=' + str(result2.value))
print('derivative value at' + str(0.5) + '=' + str(result2.deriv))

w = np.linspace(-10,10,1000)

print(sin(w))