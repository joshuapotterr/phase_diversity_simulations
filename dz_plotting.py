import matplotlib.pyplot as plt
import numpy as np
from hcipy import *
from skimage.transform import resize
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

load_heatmap=True
if load_heatmap:
            loaded_heatmap = np.load("OTF_heatmap_data.npz")
            H_loaded = loaded_heatmap["H"]
            fixed_dz_loaded= loaded_heatmap["fixed_dz_heatmap"]
            v0_loaded = loaded_heatmap["v0_heatmap"]
fixed_dz_heatmap = fixed_dz_loaded
v0_heatmap = v0_loaded
H = H_loaded
fixed_defocus_mm_custom = [215.8, 222.1, 237.6, 228.3, 234.4, 200.0, 166.0, 175.0, 197.0]
row_custom_dz = [int(np.argmin(np.abs(fixed_dz_heatmap-dzz))) for dzz in fixed_defocus_mm_custom]
dz_snapto = [float(fixed_dz_heatmap[i]) for i in row_custom_dz]



plt.figure(figsize=(11,10))
for dz_fixed, i in zip(dz_snapto,row_custom_dz):
            row = H[i, ].astype(float).copy()
            finite=np.isfinite(row)
            plt.plot(v0_heatmap[finite], row[finite], label = fr'dz={dz_fixed:.1f} mm ')
plt.legend()
plt.xlabel("Spatial frequency n0 [cycles/aperture]")
plt.ylabel("OTF sidepeak amplitude")
plt.title("Contrast vs injected frequency for heatmap defocus values")
plt.grid(True)
plt.tight_layout()
plt.show()