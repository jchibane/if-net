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
        in_path = os.path.splitext(in_path)[0]
        filename = in_path + '_voxelization_{}.npy'.format(res)

        if os.path.exists(filename):
            print('Voxelization file exists. Done.')
            return

        mesh = trimesh.load(in_path + '_scaled.off', process=False)
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
    parser.add_argument('-data', type=str)

    args = parser.parse_args()

    if args.data == "train":
        ROOT = '../SHARP_data/track1/train_partial'
    elif args.data == "test":
        ROOT = '../SHARP_data/track1/test_partial'
    elif args.data == "test-codalab-partial":
        ROOT = '../SHARP_data/track1/test-codalab-partial'
    elif args.data == "train_gt":
        ROOT = '../SHARP_data/track1/train'
    elif args.data == "test_gt":
        ROOT = '../SHARP_data/track1/test'

    p = Pool(mp.cpu_count())
    p.map(partial(voxelize, res=args.res), glob.glob( ROOT + '/*/*..npz'))