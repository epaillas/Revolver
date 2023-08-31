import fitsio
from pathlib import Path
import numpy as np
from revolver import VoxelVoids, setup_logging
from pycorr import TwoPointCorrelationFunction
from cosmoprimo.fiducial import AbacusSummit
from cosmoprimo.utils import DistanceToRedshift
from pyrecon import utils
import matplotlib.pyplot as plt
import logging


def get_data_positions(filename, distance, zmin=0.45, zmax=0.6,
    weight_type=None,):
    """Read Patchy mocks."""
    data = np.genfromtxt(filename)
    mask = (data[:, 2] > zmin) & (data[:, 2] < zmax)
    ra = data[mask, 0]
    dec = data[mask, 1]
    redshift = data[mask, 2]
    weights = np.ones(len(ra))
    if 'default' in weight_type:
        weights *= data[mask, 3]
    if 'completeness' in weight_type:
        weights *= data[mask, 4]
    dist = distance(redshift)
    positions = utils.sky_to_cartesian(dist=dist, ra=ra, dec=dec)
    return positions, weights


def get_voids_positions(data_positions, randoms_positions, cellsize,
    boxcenter=None, wrap=False, boxpad=1.0, smoothing_radius=10, return_radii=False):
    voxel = VoxelVoids(
        data_positions=data_positions,
        randoms_positions=randoms_positions,
        wrap=wrap,
        boxpad=boxpad,
        cellsize=cellsize,
        handle='/pscratch/sd/e/epaillas/tmp/survey'
    )
    voxel.set_density_contrast(smoothing_radius=smoothing_radius)
    voxel.find_voids()
    voids_positions, voids_radii = voxel.postprocess_voids()
    if return_radii:
        return voids_positions, voids_radii
    return voids_positions


setup_logging()
logger = logging.getLogger('voxel_survey')

fid_cosmo = AbacusSummit(0)
distance = fid_cosmo.comoving_radial_distance
weight_type = 'default'
zmin, zmax = 0.45, 0.6


randoms_dir = Path('/pscratch/sd/e/epaillas/ds_boss/nseries/')
randoms_fn = randoms_dir / f'Nseries_cutsky_randoms_50x_redshifts.dat'
logger.info(f'Loading randoms from {randoms_fn}')
randoms_positions, randoms_weights = get_data_positions(
    filename=randoms_fn, distance=distance, zmin=zmin, zmax=zmax, weight_type=weight_type,)

data_dir = Path(f'/pscratch/sd/e/epaillas/ds_boss/nseries/')
data_fn = data_dir / f'CutskyN1.rdzw'
logger.info(f'Loading data from {data_fn}')
data_positions, data_weights = get_data_positions(
    filename=data_fn, distance=distance, zmin=zmin, zmax=zmax, weight_type=weight_type,)


# Run the Voxel void finder
voids_positions, voids_radii = get_voids_positions(
    data_positions=data_positions,
    randoms_positions=randoms_positions,
    wrap=False,
    boxpad=1.0,
    cellsize=10.0,
    smoothing_radius=10,
    return_radii=True,
)

print(data_positions.min(), data_positions.max())
print(randoms_positions.min(), randoms_positions.max())
print(voids_positions.min(), voids_positions.max())

# Compute void-galaxy correlation function
redges = np.hstack([np.arange(0, 5, 1),
                    np.arange(7, 30, 3),
                    np.arange(31, 155, 5)])
muedges = np.linspace(-1, 1, 241)
edges = (redges, muedges)

result = TwoPointCorrelationFunction(
    mode='smu', edges=edges, data_positions1=voids_positions,
    data_positions2=data_positions, randoms_positions2=randoms_positions,
    estimator='davispeebles', nthreads=4, compute_sepsavg=False, position_type='pos',
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
