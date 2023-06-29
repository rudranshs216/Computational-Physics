from cmath import sin
from math import cos
import matplotlib.pyplot as plt
import numpy as np

def x1(t) :
    return np.cos(3*1.16*t)*np.cos(3*4.32*t)
def v1(t) :
    return -3*1.16*np.sin(3*1.16*t)*np.cos(3*4.32*t)-3*4.32*np.cos(3*1.16*t)*np.sin(3*4.32*t)
def x2(t) :
    return np.sin(3*1.16*t)*np.sin(3*4.32*t)
def v2(t) :
    return 3*1.16*np.cos(3*1.16*t)*np.sin(3*4.32*t) + 3*4.32*np.sin(3*1.16*t)*np.cos(3*4.32*t)
def U(t) :
    return 4.5*x1(t)**2 + 4.5*(x2(t)-x1(t))**2 + 4.5*x2(t)**2  
def E(t) : 
    return U(t)+ 0.05*v1(t)**2 + 0.05*v2(t)**2     

# t = np.linspace(0, 1, 1000)
t = np.arange(0,1.01, 0.01) 

# diplacement vs time

# plt.plot(t, x1(t))       
# plt.plot(t, x2(t)) 
# plt.show() 

# velocity vs time 

# plt.plot(t, v1(t))       
# plt.plot(t, v2(t)) 
# plt.show() 

# momentum vs time   

# plt.plot(t, 0.1*v1(t))       
# plt.plot(t, 0.1*v2(t))     
# plt.show()                

# Potential vs time

# plt.plot(t, U(t))     
# plt.show()                

# Total Energy vs time

plt.plot(t, E(t))     
plt.show()