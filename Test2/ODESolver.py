import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

plt.style.use('seaborn-poster')

F = lambda x, s: 100*(x*x*x - x)

t_eval = np.arange(-2, 2.01, 0.01)
sol = solve_ivp(F, [-2, 2.01], [0.1],method = 'RK45', t_eval=t_eval)

plt.plot(sol.t, sol.y[0])
plt.xlabel('x')
plt.ylabel('v')
plt.show()

# sol = solve_ivp(F, [0, np.pi], [0], t_eval=t_eval, \
#                 rtol = 1e-8, atol = 1e-8)

# plt.figure(figsize = (12, 4))
# plt.subplot(121)
# plt.plot(sol.t, sol.y[0])
# plt.xlabel('t')
# plt.ylabel('S(t)')
# plt.subplot(122)
# plt.plot(sol.t, sol.y[0] - np.sin(sol.t))
# plt.xlabel('t')
# plt.ylabel('S(t) - sin(t)')
# plt.tight_layout()
# plt.show()

# F = lambda t, s: -s

# t_eval = np.arange(0, 1.01, 0.01)
# sol = solve_ivp(F, [0, 1], [1], t_eval=t_eval)

# plt.figure(figsize = (12, 4))
# plt.subplot(121)
# plt.plot(sol.t, sol.y[0])
# plt.xlabel('t')
# plt.ylabel('S(t)')
# plt.subplot(122)
# plt.plot(sol.t, sol.y[0] - np.exp(-sol.t))
# plt.xlabel('t')
# plt.ylabel('S(t) - exp(-t)')
# plt.tight_layout()
# plt.show()


# F = lambda t, s: np.dot(np.array([[0, t**2], [-t, 0]]), s)

# t_eval = np.arange(0, 10.01, 0.01)
# sol = solve_ivp(F, [0, 10], [1, 1], t_eval=t_eval)

# plt.figure(figsize = (12, 8))
# plt.plot(sol.y.T[:, 0], sol.y.T[:, 1])
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()