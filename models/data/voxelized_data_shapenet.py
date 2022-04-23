from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import pickle
import imp
import trimesh
import torch



class VoxelizedDataset(Dataset):


    def __init__(self, mode, res = 32,  voxelized_pointcloud = False, pointcloud_samples = 3000, data_path = '/cluster/project/infk/courses/252-0579-00L/group20/SHARP_data/', split_file = '/cluster/project/infk/courses/252-0579-00L/group20/SHARP_data/track1/split.npz',
                 batch_size = 64, num_sample_points = 1024, num_workers = 12, sample_distribution = [1], sample_sigmas = [0.015], **kwargs):

        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.path = data_path
        self.split = np.load(split_file)

        self.data = self.split[mode]
        self.res = res

        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.voxelized_pointcloud = voxelized_pointcloud
        self.pointcloud_samples = pointcloud_samples

        # compute number of samples per sampling method
        self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.path + self.data[idx]
        # last = path.split(os.sep)[-1] # scan name
        # second_last = path.split(os.sep)[-2] # determine if it is partial data
        # substring = "partial"
        # if substring in second_last:
        #     path = path + last + ""
        #     path = os.path.join(path, )


        if not self.voxelized_pointcloud:
            occupancies = np.load(path[:-7] + '_voxelization_{}.npy'.format(self.res))
            occupancies = np.unpackbits(occupancies)
            input = np.reshape(occupancies, (self.res,)*3)
        else:
            voxel_path = path + '_voxelized_point_cloud_{}res_{}points.npz'.format(self.res, self.pointcloud_samples)
            occupancies = np.unpackbits(np.load(voxel_path)['compressed_occupancies'])
            input = np.reshape(occupancies, (self.res,)*3)

        points = []
        coords = []
        occupancies = []

        for i, num in enumerate(self.num_samples):
            boundary_samples_path = path + '_boundary_{}_samples.npz'.format(self.sample_sigmas[i])
            boundary_samples_npz = np.load(boundary_samples_path)
            boundary_sample_points = boundary_samples_npz['points']
            boundary_sample_coords = boundary_samples_npz['grid_coords']
            boundary_sample_occupancies = boundary_samples_npz['occupancies']
            subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
            points.extend(boundary_sample_points[subsample_indices])
            coords.extend(boundary_sample_coords[subsample_indices])
            occupancies.extend(boundary_sample_occupancies[subsample_indices])

        assert len(points) == self.num_sample_points
        assert len(occupancies) == self.num_sample_points
        assert len(coords) == self.num_sample_points

        return {'grid_coords':np.array(coords, dtype=np.float32),'occupancies': np.array(occupancies, dtype=np.float32),'points':np.array(points, dtype=np.float32), 'inputs': np.array(input, dtype=np.float32), 'path' : path}

    def get_loader(self, shuffle =True):

        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
