
# Questiom 1

import numpy as np
from numpy.linalg import \
    cond
from numpy.linalg import inv
from numpy.linalg import eig    
from numpy.linalg import det    
A = np.array([[1, 1/2, 1/3, 1/4],
              [1/2, 1/3, 1/4, 1/5], [1/3, 1/4, 1/5, 1/6], [1/4, 1/5, 1/6, 1/7]])
print('Condition number:\n', cond(A))
print('Inverse Matrix:\n', inv(A))


w, v = eig(A)
print('E-value:', w)
print('E-vector', v)

x = det(A)
print(x)

# Yes, It's Determinant is close to zero, So it is a ill-conditioned

# Question 2

import numpy as np

f = lambda t: 100 - 100*np.exp(-0.2*t) - 40*np.exp(-0.01*t) 
f_prime = lambda t: 100*0.2*np.exp(-0.2*t) + 40*0.01*np.exp(-0.01*t)  
newton_raphson = 2.5 - (f(2.5))/(f_prime(2.5))

print("newton_raphson =", newton_raphson)

print("The Temperature is: " , 100 - 100*np.exp(-0.2*newton_raphson))
