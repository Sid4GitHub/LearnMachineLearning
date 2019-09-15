import numpy as np
import matplotlib.pyplot as plt

xy=np.loadtxt("./openclassroom.stanford.edu/ex4x.dat")
x=xy[:,0]
y=xy[:,1]
print(x)
print(y)
plt.scatter(x,y, s=20, cmap='cool')
plt.show()