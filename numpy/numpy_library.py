import numpy as np

# THE BASIC 
a = np.array([1,2,3])
print(a)
# [1 2 3]

b = np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]])
print(b)
# [[9. 8. 7.]
#  [6. 5. 4.]]

# Get Dimension(mang mau chieu?)
a.ndim
# 1
b.ndim
# 2

# Get Shape(so hang , so cot)
b.shape
# (2, 3)

# Get Type
a.dtype

# dtype('int32')

# Get Size(tra ve so byte:)
a.itemsize
# 4 (int 32 : 4byte)

# get total size
a.nbytes
#12(4*3)

# get number of elements
a.size
#3


# ACCESSING/CHANGING SPECIFIC ELEMENT,ROWS, =COLUMNS,ETC
a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
# [[ 1  2  3  4  5  6  7]
#  [ 8  9 10 11 12 13 14]]

# get element
a[1,5]
#13

#get row
a[0,:]
#[1,2,3,4,5,6,7]

# Get a specific column
a[:, 2]
#[3,10]

# Getting a little more fancy [startindex:endindex:stepsize]
a[0, 1:6:2]
#[2,4,6]

# change_element
a[1,5] = 20
#[[ 1  2  5  4  5  6  7]
#  [ 8  9  5 11 12 20 14]]

a[:,2] = [1,2]
# [[ 1  2  1  4  5  6  7]
#  [ 8  9  2 11 12 20 14]]

# 3-d-example
b = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
# [[[1 2]
#   [3 4]]

#  [[5 6]
#   [7 8]]]

# Get specific element (work outside in)(mat,hang,cot)
b[0,1,1]
# 4

# replace 
b[:,1,:] =[[9,9][8,8]]
# [[[1 2]
#   [9 9]]

#  [[5 6]
#   [8 8]]]

# INITIALIZING(khoi tao) DIFFERENT TYPES OD ARRAYS 

# All 0s matrix
np.zeros((2,3))
# array([[0., 0., 0.],
#       [0., 0., 0.]])

# All 1s matrix
np.ones((4,2,2), dtype='int32')

# Any other number(shape,number)
np.full((2,2), 99)
# array([[99., 99.],
#       [99., 99.]], dtype=float32)

# Any other number (full_like)
np.full_like(a, 4)
# array([[4, 4, 4, 4, 4, 4, 4],
#       [4, 4, 4, 4, 4, 4, 4]])

# Random decimal numbers(shape)
np.random.rand(4,2)
# array([[0.07805642, 0.53385716],
#       [0.02494273, 0.99955252],
#       [0.48588042, 0.91247437],
#       [0.27779213, 0.16597751]])

# Random Integer values(start value , end value)
np.random.randint(-4,8, size=(3,3))
# array([[-2, -4, -4],
#    [ 6,  6,  3],
#    [ 3,  2,  2]])

# The identity matrix
np.identity(5)
# array([[1., 0., 0., 0., 0.],
#       [0., 1., 0., 0., 0.],
#       [0., 0., 1., 0., 0.],
#       [0., 0., 0., 1., 0.],
#       [0., 0., 0., 0., 1.]])

# Repeat an array
arr = np.array([[1,2,3]])
r1 = np.repeat(arr,3, axis=0)
# [[1 2 3]
#  [1 2 3]
#  [1 2 3]]

# 
output = np.ones((5,5))
print(output)

z = np.zeros((3,3))
z[1,1] = 9
print(z)

output[1:-1,1:-1] = z
print(output)
# [[1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]]
# [[0. 0. 0.]
#  [0. 9. 0.]
#  [0. 0. 0.]]
# [[1. 1. 1. 1. 1.]
#  [1. 0. 0. 0. 1.]
#  [1. 0. 9. 0. 1.]
#  [1. 0. 0. 0. 1.]
#  [1. 1. 1. 1. 1.]]

# MATHEMATC
a = np.array([1,2,3,4])
print(a)
# [1 2 3 4]

a + 2
# array([5, 6, 7, 8])

a - 2
# array([-1,  0,  1,  2])

a * 2
# array([2, 4, 6, 8])

a / 2
# array([0.5, 1. , 1.5, 2. ])

b = np.array([1,0,1,0])
a + b
# array([1, 0, 3, 0])

a ** 2
# array([ 1,  4,  9, 16], dtype=int32)

# Take the sin
np.cos(a)
# For a lot more (https://docs.scipy.org/doc/numpy/reference/routines.math.html)

#LINEAR ALGEBRA
a = np.ones((2,3))
print(a)

b = np.full((3,2), 2)
print(b)

np.matmul(a,b)
# [[1. 1. 1.]
#  [1. 1. 1.]]
# [[2 2]
#  [2 2]
#  [2 2]]
# array([[6., 6.],
#       [6., 6.]])

# Find the determinant
c = np.identity(3)
np.linalg.det(c)
#1.0

## Reference docs (https://docs.scipy.org/doc/numpy/reference/routines.linalg.html)

# Determinant
# Trace
# Singular Vector Decomposition
# Eigenvalues
# Matrix Norm
# Inverse
# Etc...

# STATISTICS
stats = np.array([[1,2,3],[4,5,6]])
stats
# array([[1, 2, 3],
#       [4, 5, 6]])
np.min(stats)
# 1
np.max(stats, axis=1)
# array([3, 6])
np.sum(stats, axis=0)
# array([5, 7, 9])

# REORDANIZING ARRAYS
before = np.array([[1,2,3,4],[5,6,7,8]])
print(before)

after = before.reshape((4,2))
print(after)
#[[1 2 3 4]
#  [5 6 7 8]] 

# 
# [[1,2]
#  [3,4]
#  [5,6]
#  [7,8]]