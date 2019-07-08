import numpy as np
import matplotlib.pyplot as plt
x= np.load('output/trckd_prtl/x.npy')
y= np.load('output/trckd_prtl/y.npy')

plt.plot(x,y)
plt.show()
