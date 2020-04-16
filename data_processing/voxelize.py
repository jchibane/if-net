import trimesh
import numpy as np
import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import traceback
import voxels
import argparse


def voxelize(in_path, res):
    try:

        filename = os.path.join(in_path, 'voxelization_{}.npy'.format(res))

        if os.path.exists(filename):
            return

        mesh = trimesh.load(in_path + '/isosurf_scaled.off', process=False)
        occupancies = voxels.VoxelGrid.from_mesh(mesh, res, loc=[0, 0, 0], scale=1).data
        occupancies = np.reshape(occupancies, -1)

        if not occupancies.any():
            raise ValueError('No empty voxel grids allowed.')

        occupancies = np.packbits(occupancies)
        np.save(filename, occupancies)

    except Exception as err:
        path = os.path.normpath(in_path)
        print('Error with {}: {}'.format(path, traceback.format_exc()))
    print('finished {}'.format(in_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run voxalization'
    )
    parser.add_argument('-res', type=int)

    args = parser.parse_args()

    ROOT = 'shapenet/data'

    p = Pool(mp.cpu_count())
    p.map(partial(voxelize, res=args.res), glob.glob( ROOT + '/*/*/'))