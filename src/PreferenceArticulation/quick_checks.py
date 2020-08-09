import numpy as np

x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([1, 1, 1, 1, 1])

print(np.subtract(x1[:-1], x2[:-1]))
print(np.square(np.subtract(x1[:-1], x2[:-1])))


