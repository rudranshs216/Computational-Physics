# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import lagrange
# x = [5, 6, 9, 11]
# f = [12, 13, 14, 16]
# p = lagrange(x, f)
# x_new = np.arange(-1.0, 20, 1)
# fig = plt.figure(figsize=(10, 8))
# plt.plot(x_new, p(x_new), 'b', x, f, 'ro')
# plt.title('Lagrange Polynomial')
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('f(x)')
# # plt.savefig("lagrange1.png")
# plt.show()


# import numpy as np
# from numpy.linalg import qr
# from numpy.linalg import eig
# # a = np.array([[0, 2],
# # [2, 3]])
# #q, r = qr(a)
# #print('Q:', q)
# #print('R:', r)
# #b = np.dot(q, r)
# #print('QR:', b)
# a = np.array([[0, 2],
#               [2, 3]])
# p = [1, 5, 10, 20]
# for i in range(20):
#     q, r = qr(a)
#     a = np.dot(r, q)
#     if i+1 in p:
#         print(f'Iteration {i+1}:')
#         print(a)
# w, v = eig(a)
# print('E-value:', w)
# print('E-vector', v)
# a = np.array([[2, 2, 4],
#               [1, 3, 5],
#               [2, 3, 4]])
# w, v = eig(a)
# print('E-value:', w)
# print('E-vector', v)


# import numpy as np
# from numpy.linalg import inv
# def normalize(x):
#  fac=abs(x).max()
#  x_n=x / x.max()
#  return fac, x_n
# x=np.array([1, 1])
# a=np.array([[0, 2],
#  [2, 3]])
# n=100
# a_inv=inv(a)
# print(a)
# print(a_inv)
# for i in range(n):
#  x=np.dot(a_inv, x)
#  lambda_1, x=normalize(x)
# print('Eigenvalue:', lambda_1)
# print('Eigenvector:', x)


import numpy as np
# Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver


def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-
   _TDMA_(Thomas_algorithm)
    '''
    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]
    xc = bc


    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]
        return xc
A = np.array([[10, 2, 0, 0], [3, 10, 4, 0], [
             0, 1, 7, 5], [0, 0, 3, 4]], dtype=float)
a = np.array([3., 1, 3])
b = np.array([10., 10., 7., 4.])
c = np.array([2., 4., 5.])
d = np.array([3, 4, 5, 6.])
print(TDMAsolver(a, b, c, d))
# compare against numpy linear algebra library
print(np.linalg.solve(A, d))
