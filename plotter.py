import numpy as np
import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt
x= np.load('output/trckd_prtl/x.npy')
y= np.load('output/trckd_prtl/y.npy')

plt.plot(x,y)
plt.show()

plt.plot(np.sqrt((x[:-1]-x[1:])**2+(y[1:]-y[:-1])**2), '.')
plt.show()
