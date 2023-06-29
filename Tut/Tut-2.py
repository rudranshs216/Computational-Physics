# Group-9

import numpy as np
# Question 1

w = np.array([[2],
              [3],
              [5]])
wT = np.array([[2, 3, 5]])
I = np.array(
    [[2/3, -0.25, -0.25], [-0.25, 2/3, -0.25], [-0.25, -0.25, 2/3]])

R = wT.dot(I).dot(w)
print("Rotational Energy: ", 3.8592*R/2)
w, v = np.linalg.eig(I)

# 3.8592 is for ML^2 factor

print("EigenValues: ", 3.8592*w)

# Question2

A = np.array([[0, 0.1, 0.2, 0.3, 0.5, 1.0],
              [0.1, 0.2, 0.4, 0.6, 1, 0.5],
              [0.4, 0.5, 0.7, 1, 0.6, 0.3],
              [0.8, 0.8, 1, 0.7, 0.4, 0.2],
              [0.9, 1, 0.8, 0.5, 0.2, 0.1],
              [1, 0.9, 0.8, 0.4, 0.1, 0]])
y = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1])

x = np.linalg.solve(A, y)
print("Solution of System of Linear Equations:", x)