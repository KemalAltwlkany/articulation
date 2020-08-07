import numpy as np
import random as random

a = np.random.rand(5)
b = a/a.sum()
print(a)
print(b)
print(sum(b))

print("--------------")
x = np.random.uniform(low=-4, high=4, size=5)
print(x)