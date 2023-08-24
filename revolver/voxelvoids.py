import subprocess
import numpy as np
from pyrecon import RealMesh
from pandas import qcut
import sys
import os


class VoxelVoids:
    def __init__(self, data_positions, boxsize=None, boxcenter=None,
        data_weights=None, randoms_positions=None, randoms_weights=None,
        cellsize=None, wrap=False, boxpad=1.5, nthreads=None):
        self.data_positions = data_positions
        self.randoms_positions = randoms_positions
        self.boxsize = boxsize
        self.boxcenter = boxcenter
        self.cellsize = cellsize
        self.boxpad = boxpad
        self.wrap = wrap
        self.nthreads = nthreads

        if data_weights is not None:
            self.data_weights = data_weights
        else:
            self.data_weights = np.ones(len(data_positions))

        if boxsize is None:
            if randoms_positions is None:
                raise ValueError(
                    'boxsize is set to None, but randoms were not provided.')
            if randoms_weights is None:
                self.randoms_weights = np.ones(len(randoms_positions))
            else:
                self.randoms_weights = randoms_weights


    def set_density_contrast(self, smoothing_radius, check=False, ran_min=0.01):
        self.data_mesh = RealMesh(boxsize=self.boxsize, cellsize=self.cellsize,
                                  boxcenter=self.boxcenter, nthreads=self.nthreads,
                                  positions=self.randoms_positions, boxpad=self.boxpad)
        self.data_mesh.assign_cic(positions=self.data_positions, wrap=self.wrap,
                                  weights=self.data_weights)
        self.data_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=True,)
        if self.boxsize is None:
            self.randoms_mesh = RealMesh(boxsize=self.boxsize, cellsize=self.cellsize,
                                         boxcenter=self.boxcenter, nthreads=self.nthreads,
                                         positions=self.randoms_positions, boxpad=self.boxpad)
            self.randoms_mesh.assign_cic(positions=self.randoms_positions, wrap=self.wrap,
                                         weights=self.randoms_weights)
            if check:
                mask_nonzero = self.randoms_mesh.value > 0.
                nnonzero = mask_nonzero.sum()
                if nnonzero < 2: raise ValueError('Very few randoms.')
            self.randoms_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=True)
            sum_data, sum_randoms = np.sum(self.data_mesh.value), np.sum(self.randoms_mesh.value)
            alpha = sum_data * 1. / sum_randoms
            self.delta_mesh = self.data_mesh - alpha * self.randoms_mesh
            threshold = ran_min * sum_randoms / len(self.randoms_positions)
            mask = self.randoms_mesh > threshold
            self.delta_mesh[mask] /= alpha * self.randoms_mesh[mask]
            self.delta_mesh[~mask] = 0.0
            del self.data_mesh
            del self.randoms_mesh
        else:
            self.delta_mesh = self.data_mesh / np.mean(self.data_mesh) - 1.
            del self.data_mesh
        return self.delta_mesh

    def find_voids(self):
        self.nbins = int(self.boxsize / self.cellsize)
        # write this to file for jozov-grid to read
        delta_mesh_flat = np.array(self.delta_mesh, dtype=np.float32)
        with open(f'delta_mesh_n{self.nbins}d.dat', 'w') as F:
            delta_mesh_flat.tofile(F, format='%f')

        # now call jozov-grid
        bin_path  = os.path.join(os.path.dirname(__file__), 'c', 'jozov-grid')
        cmd = [bin_path, "v", f"delta_mesh_n{self.nbins}d.dat",
               './tmp', str(self.nbins)]
        subprocess.call(cmd)


    def postprocess_voids(self):

        print("Post-processing voids")
        self.mask_cut = np.zeros(self.nbins**3, dtype='int')  # we don't mask any voxels in a box
        self.min_dens_cut = 1.0

        raw_dir = './'
        rawdata = np.loadtxt("tmp.txt", skiprows=2)
        nvox = self.nbins ** 3

        # load zone membership data
        zonefile = "tmp.zone"
        with open(zonefile, 'r') as F:
            hierarchy = F.readlines()
        hierarchy = np.asarray(hierarchy, dtype=str)

        # remove voids that: a) don't meet minimum density cut, b) are edge voids, or c) lie in a masked voxel
        # select = np.zeros(rawdata.shape[0], dtype='int')
        # fastmodules.voxelvoid_cuts(select, self.mask_cut, rawdata, self.min_dens_cut)
        # select = np.asarray(select, dtype=bool)
        # rawdata = rawdata[select]
        # densratio = densratio[select]
        # hierarchy = hierarchy[select]

        # void minimum density centre locations
        xpos, ypos, zpos = self.voxel_position(rawdata[:, 2])

        # if not self.is_box:  # convert void centre coordinates from box Cartesian to sky positions
        #     xpos += self.xmin
        #     ypos += self.ymin
        #     zpos += self.zmin
        #     dist = np.sqrt(xpos**2 + ypos**2 + zpos**2)
        #     redshift = self.cosmo.get_redshift(dist)
        #     ra = np.degrees(np.arctan2(ypos, xpos))
        #     dec = 90 - np.degrees(np.arccos(zpos / dist))
        #     ra[ra < 0] += 360
        #     xpos = ra
        #     ypos = dec
        #     zpos = redshift
        #     # and an additional cut on any voids with min. dens. centre outside specified redshift range
        #     select_z = np.logical_and(zpos > self.z_min, zpos < self.z_max)
        #     rawdata = rawdata[select_z]
        #     densratio = densratio[select_z]
        #     hierarchy = hierarchy[select_z]
        #     xpos = xpos[select_z]
        #     ypos = ypos[select_z]
        #     zpos = zpos[select_z]

        # void effective radii
        vols = (rawdata[:, 5] * self.cellsize ** 3.)
        rads = (3. * vols / (4. * np.pi)) ** (1. / 3)
        # void minimum densities (as delta)
        mindens = rawdata[:, 3] - 1.

        os.remove('tmp.void')
        os.remove('tmp.txt')
        os.remove('tmp.zone')
        os.remove(f'delta_mesh_n{self.nbins}d.dat')
        return np.c_[xpos, ypos, zpos], rads
        # void average densities and barycentres
        # avgdens = np.zeros(len(rawdata))
        # barycentres = np.zeros((len(rawdata), 3))
        # for i in range(len(rawdata)):
        #     member_voxels = np.fromstring(hierarchy[i], dtype=int, sep=' ')[1:]
        #     member_dens = np.zeros(len(member_voxels), dtype='float64')
        #     fastmodules.get_member_densities(member_dens, member_voxels, self.rhoflat)
        #     # member_dens = self.rhoflat[member_voxels]
        #     avgdens[i] = np.mean(member_dens) - 1.
        #     if self.use_barycentres:
        #         member_x, member_y, member_z = self.voxel_position(member_voxels)
        #         barycentres[i, 0] = np.average(member_x, weights=1. / member_dens)
        #         barycentres[i, 1] = np.average(member_y, weights=1. / member_dens)
        #         barycentres[i, 2] = np.average(member_z, weights=1. / member_dens)
        # if self.use_barycentres and not self.is_box:
        #     barycentres[:, 0] += self.xmin
        #     barycentres[:, 1] += self.ymin
        #     barycentres[:, 2] += self.zmin
        #     dist = np.linalg.norm(barycentres, axis=1)
        #     redshift = self.cosmo.get_redshift(dist)
        #     ra = np.degrees(np.arctan2(barycentres[:, 1], barycentres[:, 0]))
        #     dec = 90 - np.degrees(np.arccos(barycentres[:, 2] / dist))
        #     ra[ra < 0] += 360
        #     barycentres[:, 0] = ra
        #     barycentres[:, 1] = dec
        #     barycentres[:, 2] = redshift

        # # record void lambda value, even though usefulness of this has only really been shown for ZOBOV voids so far
        # void_lambda = avgdens * (rads ** 1.2)

        # # create output array
        # output = np.zeros((len(rawdata), 9))
        # output[:, 0] = rawdata[:, 0]
        # output[:, 1] = xpos
        # output[:, 2] = ypos
        # output[:, 3] = zpos
        # output[:, 4] = rads
        # output[:, 5] = mindens
        # output[:, 6] = avgdens
        # output[:, 7] = void_lambda
        # output[:, 8] = densratio

        # print('Total %d voids pass all cuts' % len(output))
        # sys.stdout.flush()

        # # sort in increasing order of minimum density
        # sort_order = np.argsort(output[:, 5])
        # output = output[sort_order]
        # if self.use_barycentres:
        #     barycentres = barycentres[sort_order]
        # # save to file
        # catalogue_file = self.output_folder + self.void_prefix + '_cat.txt'
        # header = '%d voxels, %d voids\n' % (nvox, len(output))
        # if self.is_box:
        #     header += 'VoidID XYZ[3](Mpc/h) R_eff(Mpc/h) delta_min delta_avg lambda_v DensRatio'
        # else:
        #     header += 'VoidID RA Dec z R_eff(Mpc/h) delta_min delta_avg lambda_v DensRatio'
        # np.savetxt(catalogue_file, output, fmt='%d %0.4f %0.4f %0.4f %0.4f %0.6f %0.6f %0.6f %0.6f', header=header)

        # if self.use_barycentres:
        #     if not os.access(self.output_folder + "barycentres/", os.F_OK):
        #         os.makedirs(self.output_folder + "barycentres/")
        #     catalogue_file = self.output_folder + 'barycentres/' + self.void_prefix + '_baryC_cat.txt'
        #     output[:, 1:4] = barycentres
        #     np.savetxt(catalogue_file, output, fmt='%d %0.4f %0.4f %0.4f %0.4f %0.6f %0.6f %0.6f %0.6f',
        #                header=header)


    def voxel_position(self, voxel):
        xind = np.array(voxel / (self.nbins ** 2), dtype=int)
        yind = np.array((voxel - xind * self.nbins ** 2) / self.nbins, dtype=int)
        zind = np.array(voxel % self.nbins, dtype=int)
        xpos = xind * self.boxsize / self.nbins
        ypos = yind * self.boxsize / self.nbins
        zpos = zind * self.boxsize / self.nbins
        return xpos, ypos, zpos
