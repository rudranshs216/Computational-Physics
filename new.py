from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# LORENZ ATTRACTOR

def odes(a, t):
    x = a[0]
    y = a[1]


    dxdt = y
    dydt = -x -0.5*(x*x - 1)*y

    return [dxdt,dydt]

# initial conditions
a0 = [0,0]

# time interval
t = np.arange(0.0, 40.01, 0.01)

# ODE solution
a = odeint(odes , a0 ,t)

# Solution
X = a[:,0]
Y = a[:,1]

plt.title('X vs t')
plt.plot(t,X)
plt.show()

plt.title('Y vs t')
plt.plot(t,Y)
plt.show()

# plt.title('Z vs t')
# plt.plot(t,Z)
# plt.show()

# plt.title('dX/dt vs X')
# plt.plot(X, 10*(Y-X))
# plt.show()

# plt.title('dY/dt vs Y')
# plt.plot(Y, 28*X-Y-X*Z)
# plt.show()

# plt.title('dZ/dt vs Z')
# plt.plot(Z, -(8/3)*Z + X*Y)
# plt.show()

# fig = plt.figure()
# ax = plt.axes(projection ='3d')
# ax.plot3D(X, Y, Z, 'green')
# ax.set_title('Lorenz Attractor')
# plt.show()

    
# fig = plt.figure('Lorenz Attractor', facecolor = 'k', figsize = (10, 9))
# fig.tight_layout()
# ax = plt.axes(projection = '3d')

# def update(i):
#     ax.view_init(-6, -56 + i/2)
#     ax.clear()
#     ax.set(facecolor = 'k')
#     ax.set_axis_off()
#     ax.plot(X, Y, Z, color= 'lime', lw = 0.9)

# ani = animation.FuncAnimation(fig, update, np.arange(15000), interval = 2, repeat = False)
# plt.show()