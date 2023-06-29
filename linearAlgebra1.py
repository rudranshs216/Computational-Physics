import numpy as np
from numpy.linalg import norm
from numpy.linalg import \
             cond, matrix_rank

vector_row = np.array([[1, -5, 3, 2, 4]])
vector_column = np.array([[1], 
                          [2], 
                          [3], 
                          [4]])

new_vector = vector_row.T
print(new_vector)
norm_1 = norm(new_vector, 1)
norm_2 = norm(new_vector, 2)
norm_inf = norm(new_vector, np.inf)
print('L_1 is: %.1f'%norm_1)
print('L_2 is: %.1f'%norm_2)
print('L_inf is: %.1f'%norm_inf)

A = np.array([[1.0001,1],
              [1,1.0001]])
print('Condition number:\n', cond(A)) 