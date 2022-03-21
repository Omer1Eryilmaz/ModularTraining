import numpy as np
from sympy import arg
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
        b.value = np.sin(a)
        b.deriv = np.cos(a)*a.deriv

        return b

g = lambda w: sin(w)

a1 = ADTuple(val = 0)
a2 = ADTuple(val = 0.5)

result1 = g(a1)
result2 = g(a2)

print('function value at' + str(0) + '=' + str(result1.val))
print('derivative value at' + str(0) + '=' + str(result1.der))

print('function value at' + str(0.5) + '=' + str(result2.val))
print('derivative value at' + str(0.5) + '=' + str(result2.der))