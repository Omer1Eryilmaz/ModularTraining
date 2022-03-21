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


