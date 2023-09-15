from getdist import plots, MCSamples
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
import inference_plot_utils as inference_plots

args = argparse.ArgumentParser()
args.add_argument(
    "--chain_dir",
    type=str,
    default="/pscratch/sd/e/epaillas/sunbird/chains/enrique",
)
args.add_argument(
    "--loss",
    type=str,
    default='learned_gaussian',
)
args = args.parse_args()

cosmologies = [0, 1, 3, 4]

# best hod for each cosmology
best_hod = {0: 80, 1:80, 3:80, 4: 80,}

chain_dir = Path(args.chain_dir)
chain_labels = [f"c{cosmo:03}" for cosmo in cosmologies]

params_toplot = [
    "omega_cdm",
    "sigma8_m",
    # "logM_cut",
    # "logM1"
]
true_params = [
    inference_plots.get_true_params(cosmo, best_hod[cosmo]) for cosmo in cosmologies
]

samples_list = []
for cosmo in cosmologies:
    hod = best_hod[cosmo]
    chain_fn = (
        chain_dir
        / f'abacus_c{cosmo}_voxel_voids_mae_patchycov_vol64_smin0.70_smax120.00_m02_base'
    )
    samples_list.append(
        inference_plots.get_MCSamples(
            chain_fn / "results.csv",
        )
    )

colors = ["#4165c0", "#e770a2", "#5ac3be", "dimgray",] #"#f79a1e"]
g = inference_plots.plot_corner(
    samples_list,
    params_toplot,
    chain_labels,
    colors=colors,
    true_params=None,
    markers=true_params,
    markers_colors=colors[::-1],
    inches_per_param=15.5 / 7,
)
# g.fig.suptitle('Voxel voids')
# plt.savefig("figures/pdf/F7_cosmo_c0_c1_c3_c4.pdf", bbox_inches="tight")
# plt.savefig("figures/png/F7_cosmo_c0_c1_c3_c4.png", bbox_inches="tight")
# plt.savefig('cosmo_inference_abacus.pdf')
plt.tight_layout()
plt.savefig('cosmo_inference_abacus.png', dpi=300, bbox_inches='tight')
plt.show()