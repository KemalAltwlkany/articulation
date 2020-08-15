import numpy as np
x1 = np.linspace(0, 5, 5)
x2 = np.linspace(0, 5, 5)
x3 = np.linspace(0, 1, 25)
x1, x2 = np.meshgrid(x1, x2)
x1 = np.reshape(x1, x1.size)
x2 = np.reshape(x2, x2.size)
v = np.column_stack((x1, x2, x3))
print(v)
