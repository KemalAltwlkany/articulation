import numpy as np

l = []
for i in range(100):
    l.append(i % 5)
print(l.count(0))
print(l.count(1))
print(l.count(2))
print(l.count(3))
print(l.count(4))
