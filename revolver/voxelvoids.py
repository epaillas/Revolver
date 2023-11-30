import subprocess
import numpy as np
from pyrecon import RealMesh
import revolver.fastmodules as fastmodules
import logging
import time
import sys
import os


class VoxelVoids:
    def __init__(self, data_positions, boxsize=None, boxcenter=None,
        data_weights=None, randoms_positions=None, randoms_weights=None,
        cellsize=None, wrap=False, boxpad=1.5, handle=None):
        self.data_positions = data_positions
        self.randoms_positions = randoms_positions
        self.boxsize = boxsize
        self.boxcenter = boxcenter
        self.cellsize = cellsize
        self.boxpad = boxpad
        self.wrap = wrap

        self.logger = logging.getLogger('VoxelVoids')

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
        else:
            self.boxsize = np.array([boxsize] * 3) if np.isscalar(boxsize) else boxsize

        self.handle = 'tmp' if handle is None else handle


    def set_density_contrast(self, smoothing_radius, check=False, ran_min=0.1, nthreads=1):
        self.logger.info('Setting density contrast')
        self.time = time.time()
        if self.boxsize is None:
            # we do a first iteration to figure out the boxsize
            self.randoms_mesh = RealMesh(cellsize=self.cellsize, boxcenter=self.boxcenter, nthreads=nthreads,
                                         positions=self.randoms_positions, boxpad=self.boxpad)
            max_boxsize = np.max(self.randoms_mesh.boxsize)
            # now build the mesh with the fixed boxsize
            self.randoms_mesh = RealMesh(boxsize=max_boxsize, cellsize=self.cellsize,
                                         boxcenter=self.randoms_mesh.boxcenter, nthreads=nthreads,)
            self.randoms_mesh.assign_cic(positions=self.randoms_positions, wrap=self.wrap,
                                         weights=self.randoms_weights)
            self.data_mesh = RealMesh(boxsize=max_boxsize, cellsize=self.cellsize,
                                      boxcenter=self.randoms_mesh.boxcenter, nthreads=nthreads,)
            self.data_mesh.assign_cic(positions=self.data_positions, wrap=self.wrap,
                                    weights=self.data_weights)
            if check:
                mask_nonzero = self.randoms_mesh.value > 0.
                nnonzero = mask_nonzero.sum()
                if nnonzero < 2: raise ValueError('Very few randoms.')
            self.randoms_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=True)
            self.data_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=True,)
            sum_data, sum_randoms = np.sum(self.data_mesh.value), np.sum(self.randoms_mesh.value)
            alpha = sum_data * 1. / sum_randoms
            self.delta_mesh = self.data_mesh - alpha * self.randoms_mesh
            self.ran_min = ran_min
            self.ran_min *= sum_randoms / len(self.randoms_positions)
            mask = self.randoms_mesh > self.ran_min
            self.delta_mesh[mask] /= alpha * self.randoms_mesh[mask]
            self.delta_mesh[~mask] = 0.0
            del self.data_mesh
            # del self.randoms_mesh
        else:
            self.data_mesh = RealMesh(boxsize=self.boxsize, cellsize=self.cellsize,
                                    boxcenter=self.boxcenter, nthreads=nthreads,
                                    positions=self.randoms_positions, boxpad=self.boxpad)
            self.data_mesh.assign_cic(positions=self.data_positions, wrap=self.wrap,
                                    weights=self.data_weights)
            self.data_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=True,)
            self.delta_mesh = self.data_mesh / np.mean(self.data_mesh) - 1.
            del self.data_mesh
        return self.delta_mesh

    def find_voids(self):
        self.logger.info("Finding voids")
        self.nmesh = self.delta_mesh.nmesh
        delta_mesh_flat = np.array(self.delta_mesh, dtype=np.float32)
        with open(f'{self.handle}_delta_mesh_n{self.nmesh[0]}{self.nmesh[1]}{self.nmesh[2]}d.dat', 'w') as F:
            delta_mesh_flat.tofile(F, format='%f')
        bin_path  = os.path.join(os.path.dirname(__file__), 'c', 'jozov-grid.exe')
        cmd = [bin_path, "v", f"{self.handle}_delta_mesh_n{self.nmesh[0]}{self.nmesh[1]}{self.nmesh[2]}d.dat",
               self.handle, str(self.nmesh[0]),str(self.nmesh[1]),str(self.nmesh[2])]
        subprocess.call(cmd)

    def postprocess_voids(self):
        self.logger.info("Post-processing voids")
        mask_cut = np.zeros(self.nmesh[0]*self.nmesh[1]*self.nmesh[2], dtype='int')
        if self.boxsize is None:
            # identify "empty" cells for later cuts on void catalogue
            mask_cut = np.zeros(self.nmesh[0]*self.nmesh[1]*self.nmesh[2], dtype='int')
            fastmodules.survey_mask(mask_cut, self.randoms_mesh.value, self.ran_min)
        self.mask_cut = mask_cut
        self.min_dens_cut = 1.0

        rawdata = np.loadtxt(f"{self.handle}.txt", skiprows=2)

        # remove voids that: a) don't meet minimum density cut, b) are edge voids, or c) lie in a masked voxel
        select = np.zeros(rawdata.shape[0], dtype='int')
        fastmodules.voxelvoid_cuts(select, self.mask_cut, rawdata, self.min_dens_cut)
        select = np.asarray(select, dtype=bool)
        rawdata = rawdata[select]

        # void minimum density centre locations
        self.logger.info('Calculating void positions')
        xpos, ypos, zpos = self.voxel_position(rawdata[:, 2])

        # void effective radii
        self.logger.info('Calculating void radii')
        vols = (rawdata[:, 5] * self.cellsize ** 3.)
        rads = (3. * vols / (4. * np.pi)) ** (1. / 3)

        os.remove(f'{self.handle}.void')
        os.remove(f'{self.handle}.txt')
        os.remove(f'{self.handle}.zone')
        os.remove(f'{self.handle}_delta_mesh_n{self.nmesh[0]}{self.nmesh[1]}{self.nmesh[2]}d.dat')

        self.logger.info(f"Found a total of {len(rawdata)} voids in {time.time() - self.time:.2f} s.")
        return np.c_[xpos, ypos, zpos], rads

    # def voxel_position(self, voxel):
    #     xind = np.array(voxel / (self.nmesh ** 2), dtype=int)
    #     yind = np.array((voxel - xind * self.nmesh ** 2) / self.nmesh, dtype=int)
    #     zind = np.array(voxel % self.nmesh, dtype=int)
    #     if self.boxsize is None:
    #         xpos = xind * self.delta_mesh.boxsize[0] / self.nmesh
    #         ypos = yind * self.delta_mesh.boxsize[0] / self.nmesh
    #         zpos = zind * self.delta_mesh.boxsize[0] / self.nmesh

    #         xpos += self.delta_mesh.boxcenter[0] - self.delta_mesh.boxsize[0] / 2.
    #         ypos += self.delta_mesh.boxcenter[1] - self.delta_mesh.boxsize[1] / 2.
    #         zpos += self.delta_mesh.boxcenter[2] - self.delta_mesh.boxsize[2] / 2.
    #     else:
    #         xpos = xind * self.boxsize / self.nmesh
    #         ypos = yind * self.boxsize / self.nmesh
    #         zpos = zind * self.boxsize / self.nmesh
    #     return xpos, ypos, zpos

    def voxel_position(self, voxel):
        voxel = voxel.astype('i')
        all_vox = np.arange(0,self.nmesh[0]*self.nmesh[1]*self.nmesh[2],dtype=int)
        vind = np.zeros((np.copy(all_vox).shape[0]),dtype=int) 
        xpos = np.zeros(vind.shape[0],dtype=float)
        ypos = np.zeros(vind.shape[0],dtype=float)
        zpos = np.zeros(vind.shape[0],dtype=float)
        all_vox = np.arange(0,self.nmesh[0]*self.nmesh[1]*self.nmesh[2],dtype = int)
        xi = np.zeros(self.nmesh[0]*self.nmesh[1]*self.nmesh[2])
        yi = np.zeros(self.nmesh[1]*self.nmesh[2])
        zi = np.arange(self.nmesh[2])
        if self.boxsize is None:
            for i in range(self.nmesh[1]):
                yi[i*(self.nmesh[2]):(i+1)*(self.nmesh[2])] =i
            for i in range(self.nmesh[0]):
                xi[i*(self.nmesh[1]*self.nmesh[2]):(i+1)*(self.nmesh[1]*self.nmesh[2])] = i
            xpos = xi*self.delta_mesh.boxsize[0]/self.nmesh[0]
            ypos = np.tile(yi,self.nmesh[0])*self.delta_mesh.boxsize[1]/self.nmesh[1]
            zpos = np.tile(zi,self.nmesh[1]*self.nmesh[0])*self.delta_mesh.boxsize[2]/self.nmesh[2]
            xpos += self.delta_mesh.boxcenter[0] - self.delta_mesh.boxsize[0] / 2.
            ypos += self.delta_mesh.boxcenter[1] - self.delta_mesh.boxsize[1] / 2.           
            zpos += self.delta_mesh.boxcenter[2] - self.delta_mesh.boxsize[2] / 2.
            return xpos[voxel],ypos[voxel],zpos[voxel]
        else:
            for i in range(self.nmesh[1]):
                yi[i*(self.nmesh[2]):(i+1)*(self.nmesh[2])] =i
            for i in range(self.nmesh[0]):
                xi[i*(self.nmesh[1]*self.nmesh[2]):(i+1)*(self.nmesh[1]*self.nmesh[2])] = i
            xpos = xi*self.boxsize[0]/self.nmesh[0]
            ypos = np.tile(yi,self.nmesh[0])*self.boxsize[1]/self.nmesh[1]
            zpos = np.tile(zi,self.nmesh[1]*self.nmesh[0])*self.boxsize[2]/self.nmesh[2]
            return xpos[voxel],ypos[voxel],zpos[voxel]