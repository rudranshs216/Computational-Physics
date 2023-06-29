from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')

# x = [0, 1, 2, 3]
# y = [0, 3, 2, 4]

x = [0, 2, 4, 6, 8, 10]
y = [0, 0.5767248078, -0.0660433280, -.2766838581, 0.2346363469, 0.0434727462]

# x = [0, 2, 4, 6]
# y = [0, 0.5767248078, -0.0660433280, -.2766838581]

# x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y = [0, 0.4400505857, 0.5767248078, 0.3390589585, -0.0660433280,
#      0.3275791376, -.2766838581, 0.0046828235,  0.2346363469, 0.2453117866, 0.0434727462]

f1 = CubicSpline(x, y, bc_type='natural')
f2 = CubicSpline(x, y, bc_type='clamped')

x_new = np.linspace(0, 10, 100)
y1_new = f1(x_new)
y2_new = f2(x_new)

plt.plot(x_new, y1_new, 'b', label="Natural")
plt.plot(x_new, y2_new, 'r', label="Clamped")
plt.legend()
plt.title('Cubic Spline Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
