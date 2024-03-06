from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cosmo_list = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))

# transform parameters
param_dir = '/pscratch/sd/e/epaillas/sunbird/data/parameters/abacus/voidprior'
combined_df = pd.DataFrame()
for cosmo in cosmo_list:
    param_fn = Path(param_dir, f'AbacusSummit_c{cosmo:03}.csv')
    param_df = pd.read_csv(param_fn)
    combined_df = pd.concat([combined_df, param_df])
output_dir = '/pscratch/sd/e/epaillas/voxel_emulator/cosmopower'
output_fn = Path(output_dir) / 'parameters.npy'
np.save(output_fn, combined_df.to_dict('list'))

 # transform multipoles
data_dir = '/pscratch/sd/e/epaillas/sunbird/data/clustering/abacus/voidprior/voxel_voids'
combined_data = []
for cosmo in cosmo_list:
    data_fn = Path(data_dir) / f'voxel_voids_multipoles_Rs10_c{cosmo:03}_ph000.npy'
    data = np.load(data_fn, allow_pickle=True).item()
    s = data['s']
    multipoles = data['multipoles'].mean(axis=1)[:, :2]
    combined_data.append(multipoles)
combined_data = np.concatenate(combined_data, axis=0)
combined_data = combined_data.reshape(len(combined_data), -1)
output_dir = '/pscratch/sd/e/epaillas/voxel_emulator/cosmopower'
output_fn = Path(output_dir) / 'voxel_voids.npy'
np.save(output_fn, combined_data)

# transform covariance
patchy_dir = '/pscratch/sd/e/epaillas/sunbird/data/clustering/patchy/voidprior/voxel_voids'
patchy_fn = Path(patchy_dir) / 'voxel_voids_multipoles_Rs10_fullap_NGC_default_FKP_landyszalay.npy'
data = np.load(patchy_fn, allow_pickle=True).item()
s = data['s']
multipoles = data['multipoles'][:, :2]
multipoles = multipoles.reshape(len(multipoles), -1)
cov = np.cov(multipoles, rowvar=False)
output_dir = '/pscratch/sd/e/epaillas/voxel_emulator/cosmopower'
output_fn = Path(output_dir) / 'covariance.npy'
np.save(output_fn, cov)