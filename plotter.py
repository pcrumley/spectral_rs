import numpy as np
import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt
#x = np.load('output/dat_00000/x.npy')
#y = np.load('output/dat_00000/y.npy')
#plt.hist(x, bins=256)
#plt.show()
for i in range(32, 62):
    outdir = f"output/dat_{str(i).zfill(5)}/flds/"

    # (ex, ey, ez) = (np.load(outdir + elm + ".npy").reshape(512, 2048) for elm in ['Bx', 'By', 'Bz'])
    dens = np.load(outdir + "Dens" + ".npy").reshape(256, 2048)
    print(np.max(dens))
    #plt.imshow(np.sqrt(ex**2 + ey**2 +ez**2))
    plt.imshow(dens, cmap='inferno', vmin=0, vmax=18)
    plt.xlim(1000,2048-64)
    plt.axis('off')
    plt.savefig(f'out{str(i).zfill(2)}.png',dpi=200, bbox_inches='tight', pad_inches=0)

