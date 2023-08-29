import numpy as np
from pycorr import TwoPointCorrelationFunction
from pathlib import Path
import matplotlib.pyplot as plt
import os

# hods = list(range(0, 100))
# cosmos = [0, 1, 2, 3, 4] + [13] + list(range(100, 127))  + list(range(130, 182))

# for cosmo in cosmos:
#     multipoles_hod = []
#     for hod in hods:
#         multipoles_los = []
#         for los in ['x', 'y', 'z']:
#             data_dir = f'/pscratch/sd/e/epaillas/voxel_emulator/voxel_multipoles/HOD/voidprior/AbacusSummit_base_c{cosmo:03}_ph000/z0.500'
#             data_fn = Path(data_dir) / f'voxel_multipoles_Rs10_c{cosmo:03}_ph000_hod{hod}_los{los}.npy'
#             result = TwoPointCorrelationFunction.load(data_fn)
#             result.select((0, 120))
#             result = result[::4, :]
#             s, multipoles = result(ells=(0, 2, 4), return_sep=True)
#             multipoles_los.append(multipoles)
#         multipoles_hod.append(multipoles_los)

#     multipoles_hod = np.asarray(multipoles_hod)
#     print(cosmo, np.shape(multipoles_hod))

#     output_dir = f'/pscratch/sd/e/epaillas/voxel_emulator/voxel_multipoles/HOD/voidprior/compressed/z0.500/'
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
#     output_fn = Path(output_dir) / f'voxel_multipoles_Rs10_c{cosmo:03}.npy'
#     cout = {'s': s, 'multipoles': multipoles_hod}
#     np.save(output_fn, cout)

hod = 0
cosmo = 0
phases = list(range(3000, 5000))
multipoles_phases = []
for phase in phases:
    data_dir = f'/pscratch/sd/e/epaillas/voxel_emulator/voxel_multipoles/HOD/voidprior/small/AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/'
    if not os.path.exists(data_dir):
        continue
    multipoles_los = []
    for los in ['x', 'y', 'z']:
        data_fn = Path(data_dir) / f'voxel_multipoles_Rs10_c{cosmo:03}_ph{phase:03}_hod{hod}_los{los}.npy'
        result = TwoPointCorrelationFunction.load(data_fn)
        result.select((0, 120))
        result = result[::4, :]
        s, multipoles = result(ells=(0, 2, 4), return_sep=True)
        if multipoles[0, 4] < -0.8:
            break
        multipoles_los.append(multipoles)
    else:
        multipoles_phases.append(multipoles_los)
multipoles_phases = np.array(multipoles_phases)

output_dir = f'/pscratch/sd/e/epaillas/voxel_emulator/voxel_multipoles/HOD/voidprior/small/compressed/z0.500/'
Path(output_dir).mkdir(parents=True, exist_ok=True)
output_fn = Path(output_dir) / f'voxel_multipoles_Rs10_c{cosmo:03}_hod{hod}.npy'
cout = {'s': s, 'multipoles': multipoles_phases}
np.save(output_fn, cout)

