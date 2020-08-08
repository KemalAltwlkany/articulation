import math as math
x1 = 1.02386434
x2 = 0.75532660
f1 = x1**2 + x2**2 - 0.1*math.cos(16*math.atan(x2/x1))
f2 = (x1 - 0.5)**2 + (x2 - 0.5)**2
print(f1)
print(f2)


