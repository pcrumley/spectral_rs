import numpy as np
import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt
# x = np.load('output/dat_00004/x.npy')
# y = np.load('output/dat_00004/u.npy')
#plt.plot(x, y, '.')
#plt.show()

outdir = "output/dat_00000/flds/"

(ex, ey, ez) = (np.load(outdir + elm + ".npy").reshape(256, 256) for elm in ['Ex', 'Ey', 'Ez'])

plt.imshow(ex)
plt.show()
plt.imshow(ey)
plt.show()
plt.imshow(ez)
plt.show()

(fx, fy, fz) = (np.load(outdir + elm + ".npy").reshape(256, 256) for elm in ['Jx', 'Jy', 'Jz'])

plt.imshow(fx)
plt.show()
plt.imshow(fy)
plt.show()
plt.imshow(fz)
plt.show()
