import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pathlib import Path
from sunbird.data.data_readers import NseriesCutsky, CMASS
from sunbird.covariance import CovarianceMatrix
from sunbird.summaries import VoxelVoids
from getdist import plots, MCSamples
from sunbird.cosmology.growth_rate import Growth
import pandas as pd
import argparse

# plt.style.use(['science.mplstyle', 'bright.mplstyle'])

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def read_dynesty_chain(filename):
    data = np.genfromtxt(filename, skip_header=1, delimiter=",")
    chain = data[:, 4:]
    weights = np.exp(data[:, 1] - data[-1, 2])
    return chain, weights

def read_hmc_chain(filename, add_fsigma8=False, redshift=0.525):
    data = pd.read_csv(filename)
    param_names = list(data.columns)
    if add_fsigma8:
        growth = Growth(
            emulate=True,
        )
        param_names.append("fsigma8")
        data['fsigma8'] = growth.get_fsigma8(
            omega_b = data['omega_b'].to_numpy(),
            omega_cdm = data['omega_cdm'].to_numpy(),
            sigma8 = data['sigma8_m'].to_numpy(),
            n_s = data['n_s'].to_numpy(),
            N_ur = np.ones_like(data['omega_b'].to_numpy()) * 2.0328,
            w0_fld = np.ones_like(data['omega_b'].to_numpy()) * -1.0,
            wa_fld = np.ones_like(data['omega_b'].to_numpy()) * 0.0,
            z=redshift,
        )
        param_names.append("H0")
        data['H0'] = growth.get_emulated_h(
            omega_b = data['omega_b'].to_numpy(),
            omega_cdm = data['omega_cdm'].to_numpy(),
            sigma8 = data['sigma8_m'].to_numpy(),
            n_s = data['n_s'].to_numpy(),
            N_ur = np.ones_like(data['omega_b'].to_numpy()) * 2.0328,
            w0_fld = np.ones_like(data['omega_b'].to_numpy()) * -1.0,
            wa_fld = np.ones_like(data['omega_b'].to_numpy()) * 0.0,
        ) * 100
        param_names.append("Omega_m")
        data['Omega_m'] = growth.Omega_m0(
            data['omega_cdm'],
            data['omega_b'],
            data['H0'] / 100,
        )
    data = data.to_numpy()
    return param_names, data, None, None

def get_MCSamples(filename, add_fsigma8=False, redshift=0.525, hmc=True,):
    priors = {
        "omega_b": [0.0207, 0.0243],
        "omega_cdm": [0.1032, 0.140],
        "sigma8_m": [0.678, 0.938],
        "n_s": [0.9012, 1.025],
        "nrun": [-0.038, 0.038],
        "N_ur": [1.188, 2.889],
        "w0_fld": [-1.22, -0.726],
        "wa_fld": [-0.628, 0.621],
        "logM1": [13.2, 14.4],
        "logM_cut": [12.4, 13.3],
        "alpha": [0.7, 1.5],
        "alpha_s": [0.7, 1.3],
        "alpha_c": [0.0, 0.5],
        "logsigma": [-3.0, 0.0],
        "kappa": [0.0, 1.5],
        "B_cen": [-0.5, 0.5],
        "B_sat": [-1.0, 1.0],
    }
    if hmc:
        names, chain, weights, loglikes = read_hmc_chain(
            filename,
            add_fsigma8=add_fsigma8,
            redshift=redshift,
        )
    else:
        names, chain, weights, loglikes = read_dynesty_chain(
            filename,
            add_fsigma8=add_fsigma8,
            redshift=redshift,
        )

    samples = MCSamples(
        samples=chain,
        weights=weights,
        labels=[labels[n] for n in names],
        names=names,
        ranges=priors,
        loglikes=loglikes
    )
    # print(samples.getTable(limit=1).tableTex())
    return samples, names

labels = {
    "omega_b": r"\omega_{\rm b}",
    "omega_cdm": r"\omega_{\rm cdm}",
    "sigma8_m": r"\sigma_8",
    "n_s": r"n_s",
    "nrun": r"\alpha_s",
    "N_ur": r"N_{\rm ur}",
    "w0_fld": r"w_0",
    "wa_fld": r"w_a",
    "logM1": "logM_1",
    "logM_cut": r"logM_{\rm cut}",
    "alpha": r"\alpha",
    "alpha_s": r"\alpha_{\rm vel, s}",
    "alpha_c": r"\alpha_{\rm vel, c}",
    "logsigma": r"\log \sigma",
    "kappa": r"\kappa",
    "B_cen": r"B_{\rm cen}",
    "B_sat": r"B_{\rm sat}",
    "fsigma8": r"f \sigma_8",
    "Omega_m": r"\Omega_{\rm m}",
    "H0": r"H_0",
}

# Nseries cosmology
Omega_m = 0.286
Omega_b = 0.047
Omega_cdm = Omega_m - Omega_b
h = 0.7
omega_cdm = Omega_cdm * h**2
omega_b = Omega_b * h**2
truth = {
    "omega_b": omega_b,
    "omega_cdm": omega_cdm,
    "sigma8_m": 0.82,
    "n_s": 0.96,
    "nrun": 0.0,
    "N_ur": 2.046,
    "w0_fld": -1.0,
    "wa_fld": 0.0,
}
redshift = 0.525
growth = Growth(emulate=True,)
pred_fsigma8 = growth.get_fsigma8(
    omega_b = np.array(truth['omega_b']).reshape(1,1),
    omega_cdm = np.array(truth['omega_cdm']).reshape(1,1),
    sigma8 = np.array(truth['sigma8_m']).reshape(1,1),
    n_s = np.array(truth['n_s']).reshape(1,1),
    N_ur = np.array(truth['N_ur']).reshape(1,1),
    w0_fld = np.array(truth['w0_fld']).reshape(1,1),
    wa_fld = np.array(truth['wa_fld']).reshape(1,1),
    z=redshift,
)
truth['fsigma8'] = pred_fsigma8[0][0]


args = argparse.ArgumentParser()
args.add_argument('--statistic', type=str, nargs="*", default=['voxel_voids'])
args.add_argument('--ell', type=int, nargs="*", default=[0])
args.add_argument('--phase', type=int, default=0)
args = args.parse_args()

fitted_params = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'logM1', 'logM_cut', 'alpha', 'logsigma', 'kappa',]

for statistic in args.statistic:
    for ell in args.ell:
        print(statistic, ell)
        root_dir = Path('/pscratch/sd/e/epaillas/sunbird/chains/voxel_voids')
        chain_handle = f'hmc_nseries_cutsky_ph0_voxel_voids_mae_patchycov_vol1_smin0.70_smax120.00_m02_base'

        chain_fn = root_dir / chain_handle / 'results.csv'
        samples, names = get_MCSamples(chain_fn, hmc=True, add_fsigma8=True)

        parameters = {param: samples.mean(param) for param in fitted_params}
        parameters['nrun'] = 0.0
        parameters['N_ur'] = 2.046
        parameters['w0_fld'] = -1.0
        parameters['wa_fld'] = 0.0
        print(parameters)

        fig, ax = plt.subplots(2, 1, figsize=(4.5, 4.5), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
        colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']

        emulator = VoxelVoids()
        s = emulator.coordinates['s']
        smin, smax = 0.7, 120.0
        s = s[(s > smin) & (s < smax)]

        slice_filters = {
            's': [smin, smax],
        }
        select_filters = {
            'multipoles': ell,
        }

        datavector = NseriesCutsky(
            dataset='voidprior',
            statistics=[statistic],
            select_filters=select_filters,
            slice_filters=slice_filters,
        ).get_observation(phase=args.phase)

        cov = CovarianceMatrix(
            loss='mae',
            dataset='voidprior',
            covariance_data_class='Patchy',
            statistics=[statistic],
            select_filters=select_filters,
            slice_filters=slice_filters,
        )


        model, error_model = emulator(
            param_dict=parameters,
            select_filters=select_filters,
            slice_filters=slice_filters,
        )
        model = model.numpy()

        cov_data = cov.get_covariance_data()
        cov_emu = cov.get_covariance_emulator()
        cov_sim = cov.get_covariance_simulation()
        error_data = np.sqrt(np.diag(cov_data))
        error_emu = np.sqrt(np.diag(cov_emu))
        error_sim = np.sqrt(np.diag(cov_sim))
        error_model = np.sqrt(error_sim**2 + error_emu**2)
        error_tot = np.sqrt(error_data**2 + error_emu**2 + error_sim**2)

        ax[0].errorbar(s, datavector, error_tot, marker='o',
                        ms=4.0, ls='', color=colors[0], elinewidth=1.0,
                        capsize=0.0, markeredgewidth=1.0,
                        markerfacecolor=lighten_color(colors[0], 0.5),
                        markeredgecolor=colors[0], label='Nseries')


        ax[0].plot(s, model, ls='-', color=colors[0])
        ax[0].fill_between(s, (model - error_model), (model + error_model), color=colors[0], alpha=0.3)

        ax[1].plot(s, (datavector - model)/error_tot, ls='-', color=colors[0])

    ax[0].axes.get_xaxis().set_visible(False)
    ax[1].fill_between([-1, 130], -1, 1, color='grey', alpha=0.2)
    ax[1].fill_between([-1, 130], -2, 2, color='grey', alpha=0.15)
    # ax[1].set_ylim(-3, 3)
    ax[1].set_xlim(-0, 120)
    ax[1].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$')
    ax[0].set_ylabel(rf'$s^2\xi_{ell}(s)\,[h^{{-2}}{{\rm Mpc}}^2]$')
    ax[1].set_ylabel(rf'$\Delta\xi_{ell}(s)/\sigma$')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    # plt.savefig(f'fig/multipoles_nseries_cutsky_{statistic}_ell{ell}.pdf')
    plt.show()