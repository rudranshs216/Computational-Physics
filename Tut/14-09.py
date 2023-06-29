import numpy as np


def func(x):
    return (x - np.cos(x))


a = 0
b = 1
print("\n")
for i in range(10000):
    c = (a * func(b) - b * func(a)) / (func(b) - func(a))
    if func(c) == 0:
        print("The Number of Iterations are (For Method of False Position): ", i+1)
        break
    elif func(c) * func(a) < 0:
        b = c
    else:
        a = c
print('The Root is (By Method of False Position): ', c)
print("\n")
for i in range(10000):
    c = (a+b)/2
    if func(c) == 0:
        print("The Number of Iterations are (For Bisection Method): ", i+1)
        break
    elif func(c) * func(a) < 0:
        b = c
    else:
        a = c
print('The Root is (By Bisection Method): ', c)
print("\n")
