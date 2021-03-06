import numpy as np

x = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
inds = np.array([True]*len(x))
print(x[inds])
dists = -1*np.ones(len(x))
for i in range(len(x)):
    inds[i] = False
    print(np.sqrt(np.sum(np.square(x[i] - x[inds]), axis=1)))
    dists[i] = np.min(np.sqrt(np.sum(np.square(x[i] - x[inds]), axis=1)))
    inds[i] = True
print(dists)
print(np.max(dists))

x = np.arange(100)
print(type(x[::len(x)//4]))

# SPACING
# x = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
# y = np.array([1, 1])
# x_masked = np.ma.masked_array(x, mask=False)
# for i in range(len(x)):
#     print(np.sum(np.abs(x - x[i]), axis=1))
#     print(np.sum(np.abs(np.subtract(x, x[i]))))
#
#
# print('---------------------------')
# for i in range(len(x)):
#     x_masked.mask[i] = True
#     print(np.linalg.norm(x_masked - x[i], ord=1, axis=1))
#     x_masked.mask[i] = False
#
# print('---------------------------')
# for i in range(len(x)):
#     x_masked.mask[i] = True
#     print(np.ma.sum(np.ma.abs(x_masked - x[i]), axis=1))
#     print(np.ma.min(np.ma.sum(np.ma.abs(x_masked - x[i]), axis=1)))
#     x_masked.mask[i] = False


# x = np.array([1, 2, 3, 4, 5])
# y = np.array([10, 20, 30, 40, 50])
# v = np.stack((x, y), axis=1)
# print(v)

#x = np.array([[4, 4, 4], [10, 10, 10], [2, 2, 2], [1, 1, 1]])
#y = np.array([2, 2, 2])
#print(x-y)
#print(np.linalg.norm(x-y, axis=1))
#print(np.argmin(np.linalg.norm(x-y, axis=1)))

#print(np.linalg.norm(x-y, axis=1))


