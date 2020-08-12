import numpy as np

x = np.array([5]*5)
y = np.array([10]*5)

z = np.stack((x, y), axis=1)
print(z)
print(type(z))


