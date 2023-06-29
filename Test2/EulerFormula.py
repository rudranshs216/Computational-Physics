# import numpy as np
# import matplotlib.pyplot as plt

# plt.style.use('seaborn-poster')

# # Define parameters


# def f(t, s): return np.exp(-t)  # ODE


# h = 0.01  # Step size
# t = np.arange(0, 1 + h, h)  # Numerical grid
# s0 = -1  # Initial Condition

# # Explicit Euler Method
# s = np.zeros(len(t))
# s[0] = s0

# for i in range(0, len(t) - 1):
#     s[i + 1] = s[i] + h*f(t[i], s[i])

# plt.figure(figsize=(12, 8))
# plt.plot(t, s, 'bo--', label='Approximate')
# plt.plot(t, -np.exp(-t), 'g', label='Exact')
# plt.title('Approximate and Exact Solution \
# for Simple ODE')
# plt.xlabel('t')
# plt.ylabel('f(t)')
# plt.grid()
# plt.legend(loc='lower right')
# plt.show()


import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
# define step size
h = 0.1
# define numerical grid
t = np.arange(0, 5.1, h)
# oscillation freq. of pendulum
w = 4
s0 = np.array([[1], [0]])

m_e = np.array([[1, h],
               [-w**2*h, 1]])
m_i = inv(np.array([[1, -h],
                    [w**2*h, 1]]))
m_t = np.dot(inv(np.array([[1, -h/2],
                           [w**2*h/2, 1]])), np.array(
    [[1, h/2], [-w**2*h/2, 1]]))

s_e = np.zeros((len(t), 2))
s_i = np.zeros((len(t), 2))
s_t = np.zeros((len(t), 2))

# do integrations
s_e[0, :] = s0.T
s_i[0, :] = s0.T
s_t[0, :] = s0.T

for j in range(0, len(t)-1):
    s_e[j+1, :] = np.dot(m_e, s_e[j, :])
    s_i[j+1, :] = np.dot(m_i, s_i[j, :])
    s_t[j+1, :] = np.dot(m_t, s_t[j, :])

plt.figure(figsize=(12, 8))
plt.plot(t, s_e[:, 0], 'b-')
plt.plot(t, s_i[:, 0], 'g:')
plt.plot(t, s_t[:, 0], 'r--')
plt.plot(t, np.cos(w*t), 'k')
plt.ylim([-3, 3])
plt.xlabel('t')
plt.ylabel('$\Theta (t)$')
plt.legend(['Explicit', 'Implicit',
            'Trapezoidal', 'Exact'])
plt.show()
