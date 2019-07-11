import numpy as np
import matplotlib.pyplot as plt
x= np.load('output/trckd_prtl/x.npy')
y= np.load('output/trckd_prtl/y.npy')
gam= np.load('output/trckd_prtl/psa.npy')

plt.plot(x)
plt.show()

plt.plot(gam)
plt.show()

x0= np.load('output/dat_0000/gam.npy')
print(x0)
x1 = np.load('output/dat_0001/gam.npy')
print(x1)
print(x1-x0)
