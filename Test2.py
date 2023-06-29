import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt


def find_t(f, a, b, alpha, beta, t0, t1, maxI):

    sol1 = 0
    i = 0
    while abs(sol1-beta) > 10**-8 and i < maxI:
        sol0 = odeint(f, np.array([alpha, t0]), [a, b], atol=1e-10)[1, 0]
        sol1 = odeint(f, np.array([alpha, t1]), [a, b], atol=1e-10)[1, 0]
        t2 = t1 - (sol1 - beta)*(t1-t0)/(sol1-sol0)
        t0 = t1
        t1 = t2
        i = i+1
    if i == maxI:
        print("t not found")
    return t2


def solveSecant(f, X, a, b, alpha, beta, t0, t1, maxI):
    t = find_t(f, a, b, alpha, beta, t0, t1, maxI)
    sol = odeint(f, np.array([alpha, t]), X, atol=1e-10)[:, 0]
    return sol


def ode(y,x):
    return np.array([y[1], 5*y[1]+10*x + 5*y[0]])


X = np.linspace(0, 1, 100)
Y = solveSecant(ode, X, 0, 1, 0, 100, 0,1,40)
plt.plot(X, Y, '-k', linewidth=2)
plt.show()
