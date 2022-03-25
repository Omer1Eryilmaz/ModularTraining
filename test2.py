import numpy as np
class MyTuple:
    '''
    The basic object representing the input variable 'w'
    represents the core of our AD calculator.  An instance 
    of this class is a tuple containining one function/derivative
    evaluation of the variable 'w'.  Because it is meant to 
    represent the simple variable 'w' the derivative 'der' is
    preset to 1.  The value 'val' can be set to 0 by default.  
    '''
    def __init__(self,**kwargs):
        # variables for the value (val) and derivative (der) of our input function 
        self.val = 0
        self.der = 1    
        
        # re-assign these default values 
        if 'val' in kwargs:
            self.val = kwargs['val']
        if 'der' in kwargs:
            self.der = kwargs['der']

# our implementation of the sinusoid rule from Table 1
def sin(a):
    # Create output evaluation and derivative object
    b = MyTuple()
    
    # Produce new function value
    b.val = np.sin(a.val)

    # Produce new derivative value - we need to use the chain rule here!
    b.der = np.cos(a.val)*a.der
    
    # Return updated object
    return b

# create instance of our function to differentiate - notice this uses our homemade sine function not numpy's
g = lambda w: sin(w)

# initialize our AutoDiff object at each point
a1 = MyTuple(val = 0); a2 = MyTuple(val = 0.5)

# evaluate
result1 = g(a1); result2 = g(a2)

# print results
#print ('function value at ' + str(0) + ' = ' + str(result1.val))

#print ('derivaive value at ' + str(0) + ' = ' + str(result1.der))
#print ('function value at ' + str(0.5) + ' = ' + str(result2.val))
#print ('derivaive value at ' + str(0.5) + ' = ' + str(result2.der))

print(sin(MyTuple(val = 5)).der)
