import numpy as np
import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt
#x = np.load('output/dat_00000/x.npy')
#y = np.load('output/dat_00000/y.npy')
#plt.hist(x, bins=256)
#plt.show()

outdir = "output/dat_00027/flds/"

(ex, ey, ez) = (np.load(outdir + elm + ".npy").reshape(512, 2048) for elm in ['Bx', 'By', 'Bz'])

#plt.imshow(np.sqrt(ex**2 + ey**2 +ez**2))
plt.imshow(ez)
plt.show()
