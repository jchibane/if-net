import implicit_waterproofing as iw
from scipy.spatial import cKDTree as KDTree
import numpy as np
import trimesh
from glob import glob
import os
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import random
import traceback


def voxelized_pointcloud_sampling(tmp_path):
    path, off_path, args, grid_points, kdtree, bb_max, bb_min = tmp_path
    try:
        fname= os.path.splitext(off_path)[0]
        out_file = fname + '_voxelized_point_cloud_{}res_{}points.npz'.format(args.res, args.num_points)

        if os.path.exists(out_file):
            print('VPC File exists. Done.')
            return
        off_path = off_path


        mesh = trimesh.load(off_path)
        point_cloud = mesh.sample(args.num_points)


        occupancies = np.zeros(len(grid_points), dtype=np.int8)

        _, idx = kdtree.query(point_cloud)
        occupancies[idx] = 1

        compressed_occupancies = np.packbits(occupancies)


        np.savez(out_file, point_cloud=point_cloud, compressed_occupancies = compressed_occupancies, bb_min = bb_min, bb_max = bb_max, res = args.res)
        print('Finished {}'.format(path))

    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run point cloud sampling'
    )

    parser.add_argument('-res', default = "128", type=int)
    parser.add_argument('-num_points', default = "300", type=int)
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


    bb_min = -0.5
    bb_max = 0.5



    grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, args.res)
    kdtree = KDTree(grid_points)

    p = Pool(mp.cpu_count())
    off_paths = glob(ROOT + '/*/*scaled.off')
    new_paths = []
    for off_path in off_paths:
        path = os.path.split(off_path)[0]
        new_paths.append((path, off_path, args, grid_points, kdtree, bb_max, bb_min))


    # enabeling to run te script multiple times in parallel: shuffling the data
    random.shuffle(off_paths)
    p.map(voxelized_pointcloud_sampling, new_paths)