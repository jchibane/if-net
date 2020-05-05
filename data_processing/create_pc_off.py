import trimesh
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import glob
import os
import argparse

def create_voxel_off(path):

    pc_path = path + '/voxelized_point_cloud_{}res_{}points.npz'.format(args.res, args.num_points)
    off_path = path + '/voxelized_point_cloud_{}res_{}points.off'.format(args.res, args.num_points)

    pc = np.load(pc_path)['point_cloud']


    trimesh.Trimesh(vertices = pc , faces = []).export(off_path)
    print('Finished: {}'.format(path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create off visualization from point cloud.'
    )

    parser.add_argument('-res', type=int)
    parser.add_argument('-num_points', type=int)

    args = parser.parse_args()

    ROOT = 'shapenet/data'

    p = Pool(mp.cpu_count())
    p.map(create_voxel_off, glob.glob(ROOT + '/*/*/'))