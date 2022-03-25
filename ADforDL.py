from calendar import c
from turtle import forward

from sympy import Mul

class CompNode:
    def __init__(self, tape):
        # Make sure that the gradient tape knows us :)
        tape.add(self)
    
    # Perform the intended operation
    # and store the result in self.output
    def forward(self):
        pass
    # assume that self.gradient has all the information from outgoing nodes
    # prior to calling backward -> perform the local gradient step with respect to inputs

    def backward(self):
        pass

    # Needed to be initialized to 0
    def set_gradient(self, gradient):
        self.gradient = gradient

    def add_gradient(self, gradient):
        self.gradient += gradient

class Tape:

    def __init__(self) -> None:
        self.computations = []
    def add(self, compNode: CompNode):
        self.computations.append(compNode)

    def forward(self):
        for compNode in self.computations:
            compNode.forward()
    def backward(self):
        # first initialize all gradients to zero
        for compNode in self.computations:
            compNode.set_gradient(0)
        
        # *** we need to invert the order ***
        self.computations.reverse()
        # Last node gets a default value of one for the gradient
        self.computations[0].set_gradient(1)
        for compNode in self.computations:
            compNode.backward()

class ConstantNode(CompNode):
    def __init__(self,value,tape):
        self.value = value
        super().__init__(tape) # how does this work?

    def forward(self):
        self.output = self.value

    def backward(self):
        # nothing to do here
        pass

t = Tape()
a = ConstantNode(2,t)
b = ConstantNode(3,t)

class Multiply(CompNode):

    def __init__(self, left: CompNode, right: CompNode, tape: Tape):
        self.left = left
        self.right = right
        super().__init__(t)
    def forward(self):
        self.output = self.left.output * self.right.output

    # Has to know how to locally differentiate multiplication    
    def backward(self):
        self.left.add_gradient(self.right.output * self.gradient)
        self.right.add_gradient(self.left.output * self.gradient)

t = Tape()
a = ConstantNode(2,t)
b = ConstantNode(3,t)

o = Multiply(a, b, t)
f = Multiply(ConstantNode(5, t), o, t)
t.forward()

print(f.output)

