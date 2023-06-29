import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
 
def lorenz(state, t, sigma, beta, rho):
    x, y, z = state
     
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
     
    return [dx, dy, dz]

sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0
 
p = (sigma, beta, rho)

y0 = [1.0, 1.0, 1.0]

t = np.arange(0.0, 100.01, 0.01)
 
result = odeint(lorenz, y0, t, p)
 
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot(result[:, 0], result[:, 1], result[:, 2])

# importing mplot3d toolkits, numpy and matplotlib
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()

# syntax for 3-D projection
ax = plt.axes(projection ='3d')

# plotting
ax.plot3D(result[:, 0], result[:, 1], result[:, 2], 'green')
ax.set_title('XYZ vs Time')
plt.show()




# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Create an image of the Lorenz attractor.
# # The maths behind this code is described in the scipython blog article
# # at https://scipython.com/blog/the-lorenz-attractor/
# # Christian Hill, January 2016.
# # Updated, January 2021 to use scipy.integrate.solve_ivp.

# WIDTH, HEIGHT, DPI = 1000, 750, 100

# # Lorenz paramters and initial conditions.
# sigma, beta, rho = 10, 2.667, 28
# u0, v0, w0 = 0, 1, 1.05

# # Maximum time point and total number of time points.
# tmax, n = 100, 10000

# def lorenz(t, X, sigma, beta, rho):
#     """The Lorenz equations."""
#     u, v, w = X
#     up = -sigma*(u - v)
#     vp = rho*u - v - u*w
#     wp = -beta*w + u*v
#     return up, vp, wp

# # Integrate the Lorenz equations.
# soln = solve_ivp(lorenz, (0, tmax), (u0, v0, w0), args=(sigma, beta, rho),
#                  dense_output=True)
# # Interpolate solution onto the time grid, t.
# t = np.linspace(0, tmax, n)
# x, y, z = soln.sol(t)


# # Plotting Part

# # Plot the Lorenz attractor using a Matplotlib 3D projection.
# fig = plt.figure(facecolor='k', figsize=(WIDTH/DPI, HEIGHT/DPI))
# # fig = plt.figure()

# # # syntax for 3-D projection
# ax = plt.axes(projection ='3d')
# ax.set_facecolor('k')
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# # Make the line multi-coloured by plotting it in segments of length s which
# # change in colour across the whole time series.
# s = 10
# cmap = plt.cm.winter
# for i in range(0,n-s,s):
#     ax.plot(x[i:i+s+1], y[i:i+s+1], z[i:i+s+1], color=cmap(i/n), alpha=0.4)

# # Remove all the axis clutter, leaving just the curve.
# ax.set_axis_off()

# plt.savefig('lorenz.png', dpi=DPI)
# plt.show()
