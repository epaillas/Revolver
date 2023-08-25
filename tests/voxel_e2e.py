import fitsio
from pathlib import Path
import numpy as np
from revolver import VoxelVoids, setup_logging
from pycorr import TwoPointCorrelationFunction
import matplotlib.pyplot as plt


def get_data_positions(filename):
    data = fitsio.read(filename)
    x = data['X']
    y = data['Y']
    z = data['Z']
    data_positions = np.c_[x, y, z]
    return data_positions

def get_voids_positions(data_positions, boxsize, cellsize, boxcenter=None,
    wrap=True, boxpad=1.0, smoothing_radius=10, return_radii=False):
    boxcenter = boxsize / 2 if boxcenter is None else boxcenter
    voxel = VoxelVoids(
        data_positions=data_positions,
        boxsize=boxsize,
        boxcenter=boxcenter,
        wrap=wrap,
        boxpad=boxpad,
        cellsize=cellsize,
    )
    voxel.set_density_contrast(smoothing_radius=10)
    voxel.find_voids()
    voids_positions, voids_radii = voxel.postprocess_voids()
    if return_radii:
        return voids_positions, voids_radii
    return voids_positions


setup_logging()

# Load some mock data. These are HOD galaxies on a 1.5 Gpc/h box
data_fn = 'mock_data.fits'
data_positions = get_data_positions(data_fn)

# Run the Voxel void finder
voids_positions, voids_radii = get_voids_positions(
    data_positions=data_positions,
    boxsize=1500,
    wrap=True,
    boxpad=1.0,
    cellsize=5.0,
    smoothing_radius=10,
    return_radii=True,
)

# Compute void-galaxy correlation function
redges = np.hstack([np.arange(0, 5, 1),
                    np.arange(7, 30, 3),
                    np.arange(31, 155, 5)])
muedges = np.linspace(-1, 1, 241)
edges = (redges, muedges)

result = TwoPointCorrelationFunction(
    mode='smu', edges=edges, data_positions1=data_positions,
    data_positions2=voids_positions, estimator='auto', boxsize=1500,
    nthreads=4, compute_sepsavg=False, position_type='pos', los='z',
)
s, multipoles = result(ells=(0, 2, 4), return_sep=True)

# Plot void-galaxy correlation function monopole
fig, ax = plt.subplots()
ax.plot(s, multipoles[0])
ax.set_xlabel('s [Mpc/h]')
ax.set_ylabel('monopole')
ax.grid()
plt.show()

# Plot distribution of void radii
fig, ax = plt.subplots()
ax.hist(voids_radii, bins=40)
ax.set_xlabel('void radius [Mpc/h]')
ax.set_ylabel('number counts')
plt.show()
