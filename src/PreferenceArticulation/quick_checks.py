import numpy as np

x1 = np.arange(9).reshape((3, 3))
x2 = np.arange(3)
print(x1)
print(x2)
print(np.add(x1, x2, x1))
print(x1)