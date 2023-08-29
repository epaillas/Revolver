from astropy.io import fits
import numpy as np
from pathlib import Path
import pandas as pd
from pymocker.catalogues import read_utils
import sys
import argparse



if __name__ == '__main__':
    phase = 0
    hods = list(range(0, 100))
    cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))

    # columns to read
    columns_cosmo = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld']
    columns_hod = ['logM1', 'logM_cut', 'alpha', 'logsigma', 'kappa',]

    # output columns
    columns_csv = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'nrun', 'N_ur', 'w0_fld', 'wa_fld',
        'logM1', 'logM_cut', 'alpha', 'logsigma', 'kappa',]

    for cosmo in cosmos:
        cosmo_dict = read_utils.get_abacus_params(cosmo)
        params_cosmo = [cosmo_dict[column] for column in columns_cosmo]

        df = pd.DataFrame(columns=columns_csv)

        params_dir = Path('./hod_parameters/voidprior/')
        params_fn = params_dir / f'hod_parameters_voidprior_c{cosmo:03}.csv'
        hod_params = np.genfromtxt(params_fn, skip_header=1, delimiter=',')

        for i, hod in enumerate(hods):
            logM_cut = hod_params[i, 0]
            logM1 = hod_params[i, 1]
            logsigma = hod_params[i, 2]
            alpha = hod_params[i, 3]
            kappa = hod_params[i, 4]

            params_hod = [logM1, logM_cut, alpha, logsigma, kappa,]
            params = params_cosmo + params_hod
            df.loc[i] = params

        output_dir = '/pscratch/sd/e/epaillas/sunbird/data/parameters/abacus/voidprior'
        output_fn = Path(output_dir, f'AbacusSummit_c{cosmo:03}.csv')
        df.to_csv(output_fn, sep=',', index=False)
