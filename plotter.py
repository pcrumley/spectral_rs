import numpy as np
import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt
x= np.load('output/dat_00001/x.npy')
y= np.load('output/dat_00001/y.npy')
plt.plot(x,y, '.')
plt.show()

