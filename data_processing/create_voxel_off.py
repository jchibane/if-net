from voxels import VoxelGrid
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import glob
import os
import argparse



def create_voxel_off(tmp_path):
    path, unpackbits, res, min, max = tmp_path

    voxel_path = path + '_voxelization_{}.npy'.format( res)
    off_path = path + '_voxelization_{}.off'.format( res)
    if os.path.exists(off_path):
        print('VPC File exists. Done.')
        return


    if unpackbits:
        occ = np.unpackbits(np.load(voxel_path))
        voxels = np.reshape(occ, (res,)*3)
    else:
        voxels = np.reshape(np.load(voxel_path)['occupancies'], (res,)*3)

    loc = ((min+max)/2, )*3
    scale = max - min

    VoxelGrid(voxels, loc, scale).to_mesh().export(off_path)
    print('Finished: {}'.format(path))







if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run voxalization to off'
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

    unpackbits = True
    res = args.res
    min = -0.5
    max = 0.5

    npz_paths = glob.glob(ROOT + '/*/*..npz')
    new_paths = []
    for npz_path in npz_paths:
        path = os.path.splitext(npz_path)[0]
        new_paths.append((path, unpackbits, res, min, max))


    p = Pool(mp.cpu_count())
    p.map(create_voxel_off, new_paths)