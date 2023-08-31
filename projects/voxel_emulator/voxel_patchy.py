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
import argparse


def get_data_positions(filename, distance, zmin=0.45, zmax=0.6,
    weight_type=None, is_random=False, P0=20000.0):
    """Read Patchy mocks."""
    data = np.genfromtxt(filename)
    mask = (data[:, 2] > zmin) & (data[:, 2] < zmax)
    ra = data[mask, 0]
    dec = data[mask, 1]
    redshift = data[mask, 2]
    weights = np.ones(len(ra))
    if 'FKP' in weight_type:
        if is_random:
            weights *= 1 / (1 + P0 * data[mask, 3])
        else:
            weights *= 1 / (1 + P0 * data[mask, 4])
    if 'default' in weight_type:
        if is_random:
            weights *= (data[mask, 5] * data[mask, 6])
        else:
            weights *= (data[mask, 6] * data[mask, 7])
    dist = distance(redshift)
    positions = utils.sky_to_cartesian(dist=dist, ra=ra, dec=dec)
    return positions, weights

def get_voids_positions(data_positions, randoms_positions, cellsize,
    wrap=False, boxpad=1.0, smoothing_radius=10, return_radii=False,
    handle=None, data_weights=None, randoms_weights=None):
    voxel = VoxelVoids(
        data_positions=data_positions,
        randoms_positions=randoms_positions,
        data_weights=data_weights,
        randoms_weights=randoms_weights,
        wrap=wrap,
        boxpad=boxpad,
        cellsize=cellsize,
        handle=handle,
    )
    voxel.set_density_contrast(smoothing_radius=smoothing_radius,
                               nthreads=args.nthreads)
    voxel.find_voids()
    voids_positions, voids_radii = voxel.postprocess_voids()
    if return_radii:
        return voids_positions, voids_radii
    return voids_positions


# parsing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--start_phase", type=int, default=1)
parser.add_argument("--n_phase", type=int, default=1)
parser.add_argument("--nthreads", type=int, default=1)
parser.add_argument("--save_voids", action='store_true')
parser.add_argument("--weight_type", type=str, default=None)
parser.add_argument("--zlim", type=str, nargs='*', default=None)
parser.add_argument("--region", type=str, default='NGC')
parser.add_argument("--use_gpu", action='store_true')
args = parser.parse_args()
start_phase = args.start_phase
n_phase = args.n_phase

setup_logging()
logger = logging.getLogger('voxel_patchy')

# some convenient declarations
fid_cosmo = AbacusSummit(0)
distance = fid_cosmo.comoving_radial_distance
weight_type = '' if args.weight_type is None else f'_{args.weight_type}'
smoothing_radius = 10
cellsize = 5.0
redges = np.arange(0, 151, 1)
muedges = np.linspace(-1, 1, 241)
edges = (redges, muedges)
gpu = True if args.use_gpu else False
if args.zlim is not None:
    zlims = [float(zlim) for zlim in args.zlim]
else:
    zlims = [0.45, 0.6]
zmin, zmax = zlims
flags = f'Patchy_{args.region}_zmin{zmin}_zmax{zmax}_Rs{smoothing_radius}{weight_type}'


randoms_dir = Path('/pscratch/sd/e/epaillas/ds_boss/Patchy/')
randoms_fn = randoms_dir / f'Patchy-Mocks-Randoms-DR12{args.region}-COMPSAM_V6C_x50.dat'
logger.info(f'Reading randoms: {randoms_fn}')
randoms_positions, randoms_weights = get_data_positions(
    filename=randoms_fn, zmin=zmin, zmax=zmax,
    weight_type=weight_type, distance=distance, is_random=True)

for phase in list(range(start_phase, start_phase + n_phase)):
    data_dir = Path(f'/pscratch/sd/e/epaillas/ds_boss/Patchy/')
    data_fn = data_dir / f'Patchy-Mocks-DR12{args.region}-COMPSAM_V6C_{phase:04}.dat'
    logger.info(f'Reading data: {data_fn}')
    data_positions, data_weights = get_data_positions(
        filename=data_fn, zmin=zmin, zmax=zmax,
        weight_type=weight_type, distance=distance,)
    
    # Run the Voxel void finder
    handle = f'/pscratch/sd/e/epaillas/tmp/{flags}_ph{phase:04}'

    voids_positions, voids_radii = get_voids_positions(
        handle=handle,
        data_positions=data_positions,
        randoms_positions=randoms_positions,
        data_weights=data_weights,
        randoms_weights=randoms_weights,
        wrap=False,
        boxpad=1.1,
        cellsize=cellsize,
        smoothing_radius=smoothing_radius,
        return_radii=True,
    )

    if args.save_voids:
        output_dir = f'/pscratch/sd/e/epaillas/voxel_emulator/voxel_voids/Patchy/'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_fn = Path(output_dir) / f'voxel_voids_{flags}_ph{phase:04}.npy'
        cout = {'positions': voids_positions, 'radii': voids_radii}
        np.save(output_fn, cout)

    # Compute void-galaxy correlation function
    result = TwoPointCorrelationFunction(
        mode='smu', edges=edges, data_positions1=voids_positions,
        data_positions2=data_positions, randoms_positions2=randoms_positions,
        data_weights2=data_weights, randoms_weights2=randoms_weights,
        estimator='davispeebles', nthreads=args.nthreads, compute_sepsavg=False,
        position_type='pos', gpu=gpu,
    )

    output_dir = f'/pscratch/sd/e/epaillas/voxel_emulator/voxel_multipoles/Patchy/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_fn = Path(output_dir) / f'voxel_multipoles_{flags}_ph{phase:04}.npy'
    result.save(output_fn)