import trimesh
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import glob
import os
import argparse

def create_voxel_off(tmp_path):
    path, pc_path, args = tmp_path

    pc_path = pc_path
    fname= os.path.splitext(pc_path)[0]
    off_path = fname + '.off'

    pc = np.load(pc_path)['point_cloud']


    trimesh.Trimesh(vertices = pc , faces = []).export(off_path)
    print('Finished: {}'.format(path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create off visualization from point cloud.'
    )

    parser.add_argument('-res', type=int)
    parser.add_argument('-num_points', type=int)
    parser.add_argument('-data', type=str)

    args = parser.parse_args()

    if args.data == "train":
        ROOT = '../SHARP_data/track1/train_partial'
    elif args.data == "test":
        ROOT = '../SHARP_data/track1/test_partial'
    elif args.data == "test-codalab-partial":
        ROOT = '../SHARP_data/track1/test-codalab-partial'

    pc_paths = glob.glob(ROOT + '/*/*voxelized_point_cloud_*.npz')
    new_paths = []
    for pc_path in pc_paths:
        path = os.path.split(pc_path)[0]
        new_paths.append((path, pc_path, args))

    p = Pool(mp.cpu_count())
    p.map(create_voxel_off, new_paths)