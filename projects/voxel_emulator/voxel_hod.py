import fitsio
from pathlib import Path
import numpy as np
from revolver import VoxelVoids, setup_logging
from pycorr import TwoPointCorrelationFunction
from cosmoprimo.fiducial import AbacusSummit
from abacusnbody.hod.abacus_hod import AbacusHOD
import argparse
import yaml
import matplotlib.pyplot as plt


def get_rsd_positions(hod_dict):
    """Read positions and velocities from input fits
    catalogue and return real and redshift-space
    positions."""
    data = hod_dict['LRG']
    vx = data['vx']
    vy = data['vy']
    vz = data['vz']
    x = data['x'] + boxsize / 2
    y = data['y'] + boxsize / 2
    z = data['z'] + boxsize / 2
    x_rsd = x + vx / (hubble * az)
    y_rsd = y + vy / (hubble * az)
    z_rsd = z + vz / (hubble * az)
    x_rsd = x_rsd % boxsize
    y_rsd = y_rsd % boxsize
    z_rsd = z_rsd % boxsize
    return x, y, z, x_rsd, y_rsd, z_rsd

def get_distorted_positions(positions, q_perp, q_para, los='z'):
    """Given a set of comoving galaxy positions in cartesian
    coordinates, return the positions distorted by the 
    Alcock-Pacynski effect"""
    positions_ap = np.copy(positions)
    factor_x = q_para if los == 'x' else q_perp
    factor_y = q_para if los == 'y' else q_perp
    factor_z = q_para if los == 'z' else q_perp
    positions_ap[:, 0] /= factor_x
    positions_ap[:, 1] /= factor_y
    positions_ap[:, 2] /= factor_z
    return positions_ap

def get_distorted_box(boxsize, q_perp, q_para, los='z'):
    """Distort the dimensions of a cubic box with the
    Alcock-Pacynski effect"""
    factor_x = q_para if los == 'x' else q_perp
    factor_y = q_para if los == 'y' else q_perp
    factor_z = q_para if los == 'z' else q_perp
    boxsize_ap = [boxsize/factor_x, boxsize/factor_y, boxsize/factor_z]
    return boxsize_ap

def output_mock(mock_dict, newBall, fn, tracer):
    """Save HOD catalogue to disk."""
    Ncent = mock_dict[tracer]['Ncent']
    mock_dict[tracer].pop('Ncent', None)
    cen = np.zeros(len(mock_dict[tracer]['x']))
    cen[:Ncent] += 1
    mock_dict[tracer]['cen'] = cen
    table = Table(mock_dict[tracer])
    header = Header({'Ncent': Ncent, 'Gal_type': tracer, **newBall.tracers[tracer]})
    myfits = fits.BinTableHDU(data = table, header = header)
    myfits.writeto(fn, overwrite=True)

def get_hod(p, param_mapping, param_tracer, data_params, Ball, nthread):
    # read the parameters 
    for key in param_mapping.keys():
        mapping_idx = param_mapping[key]
        tracer_type = param_tracer[key]
        if key == 'sigma' and tracer_type == 'LRG':
            Ball.tracers[tracer_type][key] = 10**p[mapping_idx]
        else:
            Ball.tracers[tracer_type][key] = p[mapping_idx]
        # Ball.tracers[tracer_type][key] = p[mapping_idx]
    Ball.tracers['LRG']['ic'] = 1 # a lot of this is a placeholder for something more suited for multi-tracer
    ngal_dict = Ball.compute_ngal(Nthread = nthread)[0]
    N_lrg = ngal_dict['LRG']
    Ball.tracers['LRG']['ic'] = min(1, data_params['tracer_density_mean']['LRG']*Ball.params['Lbox']**3/N_lrg)
    mock_dict = Ball.run_hod(Ball.tracers, Ball.want_rsd, Nthread = nthread)
    return mock_dict

def setup_hod(config):
    print(f"Processing {config['sim_params']['sim_name']}")
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    data_params = config['data_params']
    fit_params = config['fit_params']    
    # create a new abacushod object and load the subsamples
    newBall = AbacusHOD(sim_params, HOD_params)
    newBall.params['Lbox'] = boxsize
    # parameters to fit
    param_mapping = {}
    param_tracer = {}
    for key in fit_params.keys():
        mapping_idx = fit_params[key][0]
        tracer_type = fit_params[key][-1]
        param_mapping[key] = mapping_idx
        param_tracer[key] = tracer_type
    return newBall, param_mapping, param_tracer, data_params


def get_data_positions(filename):
    data = fitsio.read(filename)
    x = data['X']
    y = data['Y']
    z = data['Z']
    data_positions = np.c_[x, y, z]
    return data_positions

def get_voids_positions(data_positions, boxsize, cellsize, boxcenter=None,
    wrap=True, boxpad=1.0, smoothing_radius=10, return_radii=False, nthreads=1):
    boxcenter = boxsize / 2 if boxcenter is None else boxcenter
    voxel = VoxelVoids(
        data_positions=data_positions,
        boxsize=boxsize,
        boxcenter=boxcenter,
        wrap=wrap,
        boxpad=boxpad,
        cellsize=cellsize,
    )
    voxel.set_density_contrast(smoothing_radius=10, nthreads=nthreads)
    voxel.find_voids()
    voids_positions, voids_radii = voxel.postprocess_voids()
    if return_radii:
        return voids_positions, voids_radii
    return voids_positions

if __name__ == '__main__':
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_hod", type=int, default=0)
    parser.add_argument("--n_hod", type=int, default=1)
    parser.add_argument("--start_cosmo", type=int, default=0)
    parser.add_argument("--n_cosmo", type=int, default=1)
    parser.add_argument("--start_phase", type=int, default=0)
    parser.add_argument("--n_phase", type=int, default=1)
    parser.add_argument("--nthreads", type=int, default=1)

    args = parser.parse_args()
    start_hod = args.start_hod
    n_hod = args.n_hod
    start_cosmo = args.start_cosmo
    n_cosmo = args.n_cosmo
    start_phase = args.start_phase
    n_phase = args.n_phase

    setup_logging(level='INFO')
    overwrite = True
    save_mock = False
    boxsize = 2000
    cellsize = 5.0
    smoothing_radius = 10
    redshift = 0.5
    redges = np.arange(0, 151, 1)
    muedges = np.linspace(-1, 1, 241)
    edges = (redges, muedges)

    # HOD configuration
    dataset = 'voidprior'
    config_dir = './'
    config_fn = Path(config_dir, f'hod_config_{dataset}.yaml')
    config = yaml.safe_load(open(config_fn))

    # baseline AbacusSummit cosmology as our fiducial
    fid_cosmo = AbacusSummit(0)

    for cosmo in range(start_cosmo, start_cosmo + n_cosmo):
        # cosmology of the mock as the truth
        mock_cosmo = AbacusSummit(cosmo)
        az = 1 / (1 + redshift)
        hubble = 100 * mock_cosmo.efunc(redshift)

        # calculate distortion parameters
        q_perp = mock_cosmo.comoving_angular_distance(redshift) / fid_cosmo.comoving_angular_distance(redshift)
        q_para = fid_cosmo.efunc(redshift) / mock_cosmo.efunc(redshift)
        q = q_perp**(2/3) * q_para**(1/3)
        print(f'q_perp = {q_perp:.3f}')
        print(f'q_para = {q_para:.3f}')
        print(f'q = {q:.3f}')

        hods_dir = Path(f'./hod_parameters/{dataset}/')
        hods_fn = hods_dir / f'hod_parameters_{dataset}_c{cosmo:03}.csv'
        hod_params = np.genfromtxt(hods_fn, skip_header=1, delimiter=',')

        for phase in range(start_phase, start_phase + n_phase):
            sim_fn = f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}'
            config['sim_params']['sim_name'] = sim_fn
            newBall, param_mapping, param_tracer, data_params = setup_hod(config)

            for hod in range(start_hod, start_hod + n_hod):
                print(f'c{cosmo:03} ph{phase:03} hod{hod}')

                hod_dict = get_hod(hod_params[hod], param_mapping, param_tracer,
                              data_params, newBall, args.nthreads)

                if save_mock:
                    output_dir = Path(f'/pscratch/sd/e/epaillas/voxel_emulator/HOD/{dataset}/',
                        f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/')
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    output_fn = Path(
                        output_dir,
                        f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}_hod{hod}.npy'
                    )
                    output_mock(hod_dict, newBall, output_fn, 'LRG',)

                x, y, z, x_rsd, y_rsd, z_rsd = get_rsd_positions(hod_dict)

                multipoles_los = []
                for los in ['x', 'y', 'z']:
                    xpos = x_rsd if los == 'x' else x
                    ypos = y_rsd if los == 'y' else y
                    zpos = z_rsd if los == 'z' else z

                    data_positions = np.c_[xpos, ypos, zpos]

                    data_positions_ap = get_distorted_positions(positions=data_positions, los=los,
                                                                q_perp=q_perp, q_para=q_para)
                    boxsize_ap = get_distorted_box(boxsize=boxsize, q_perp=q_perp, q_para=q_para,
                                                   los=los)

                    # Run the Voxel void finder
                    voids_positions = get_voids_positions(
                        data_positions=data_positions,
                        boxsize=boxsize,
                        wrap=True,
                        boxpad=1.0,
                        cellsize=cellsize,
                        smoothing_radius=smoothing_radius,
                        return_radii=False,
                        nthreads=args.nthreads
                    )

                    voids_positions_ap = get_distorted_positions(positions=voids_positions, los=los,
                                                                q_perp=q_perp, q_para=q_para)

                    # Compute void-galaxy correlation function
                    result = TwoPointCorrelationFunction(
                        mode='smu', edges=edges, data_positions1=data_positions_ap,
                        data_positions2=voids_positions_ap, estimator='auto', boxsize=boxsize_ap,
                        nthreads=4, compute_sepsavg=False, position_type='pos', los=los,
                        gpu=True,
                    )

                    output_dir = Path(f'/pscratch/sd/e/epaillas/voxel_emulator/voxel_multipoles/HOD/{dataset}/',
                        f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/')
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    output_fn = Path(
                        output_dir,
                        f'voxel_multipoles_Rs{smoothing_radius}_c{cosmo:03}_ph{phase:03}_hod{hod}_los{los}.npy'
                    )
                    result.save(output_fn)

                #     s, multipoles = result(ells=(0, 2, 4), return_sep=True)
                #     multipoles_los.append(multipoles)

                # multipoles_los = np.array(multipoles_los)

                # cout = {
                #     's': s,
                #     'multipoles': multipoles_los
                # }
                # output_dir = Path(f'/pscratch/sd/e/epaillas/voids_emulator/voids_multipoles/HOD/{dataset}/',
                #     f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/')
                # Path(output_dir).mkdir(parents=True, exist_ok=True)
                # output_fn = Path(
                #     output_dir,
                #     f'voids_multipoles_Rs{smoothing_radius}_c{cosmo:03}_ph{phase:03}_hod{hod}.npy'
                # )
                # np.save(output_fn, cout)

                # multipoles_mean = np.mean(multipoles_los, axis=0)

                # # Plot void-galaxy correlation function monopole
                # fig, ax = plt.subplots()
                # ax.plot(s, multipoles_mean[0])
                # ax.set_xlabel('s [Mpc/h]')
                # ax.set_ylabel('monopole')
                # ax.grid()
                # plt.show()

                # # Plot void-galaxy correlation function quadrupole
                # fig, ax = plt.subplots()
                # ax.plot(s, multipoles_mean[1])
                # ax.set_xlabel('s [Mpc/h]')
                # ax.set_ylabel('monopole')
                # ax.grid()
                # plt.show()
