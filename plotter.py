import numpy as np
import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt
#x = np.load('output/dat_00000/x.npy')
#y = np.load('output/dat_00000/y.npy')
#plt.hist(x, bins=256)
#plt.show()

outdir = "output/dat_00001/flds/"

(ex, ey, ez) = (np.load(outdir + elm + ".npy").reshape(256, 256) for elm in ['Ex', 'Ey', 'Ez'])

plt.imshow(ex)
plt.show()
plt.imshow(ey)
plt.show()
plt.imshow(ez)
plt.show()

(fx, fy, fz) = (np.load(outdir + elm + ".npy").reshape(256, 256) for elm in ['Jx', 'Jy', 'Jz'])
print(np.abs(fx).max())
plt.imshow(fx)
plt.show()
plt.imshow(fy)
plt.show()
plt.imshow(fz)
plt.show()
