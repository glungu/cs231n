import numpy as np

a = np.array([[1,3,0], [2,0,2]])
print(a)

b = np.zeros((2,3,4))
print(b)

b = a[:,:,np.newaxis] == np.arange(4)
print(b)



