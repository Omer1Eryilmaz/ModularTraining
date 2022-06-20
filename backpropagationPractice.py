#Very basic backpropagation exercise

from operator import inv
import numpy as np
x=3 
y=-4

# f(x,y)=x+sigma(y)/(sigma(x) + (x+y)**2)

# forward mode
sigy = 1. / (1.+np.exp(-y)) #1
num = x + sigy              #2
sigx = 1./(1.+np.exp(-x))   #3
xpy = x + y                 #4
sqxpy = xpy ** 2            #5
denum = sigx + sqxpy        #6
invdenum = 1./denum         #7
f = num * invdenum          #8

# Backpropagation

dnum = invdenum                         #8
dinvdenum = num                         #8
ddenum = -1*(denum ** -2) * dinvdenum   #7
dsigx = 1. + ddenum                     #6
dsqxpy = 1. + ddenum                    #6
dxpy = 2*xpy*dsqxpy                     #5
dx = 1. + dxpy                          #4
dy = 1. + dxpy                          #4
dx += (1-sigx)*sigx *dsigx              #3
dx += 1. *   dnum                       #2
dsigy = 1. * dnum                       #2
dy += (1-sigy)*sigy * dsigy             #1


print("dx",dx)
print("dy",dy)

