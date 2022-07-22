from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import pickle
import imp
import trimesh
import torch
from random import shuffle


class VoxelizedDataset(Dataset):


    def __init__(self,
                mode, 
                res = 32,
                voxelized_pointcloud = False,
                pointcloud_samples = 3000,
                data_path = 'shapenet/data/',
                split_file = 'shapenet/split.npz',
                batch_size = 64,
                num_sample_points = 1024,
                num_workers = 12,
                sample_distribution = [1],
                sample_sigmas = [0.015],
                use_sdf = False,
                category = None,
                matching_model = False,
                noisy = False,
                std_noise = None,
                 **kwargs):

        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.path = data_path
        self.split = np.load(split_file) if category is None else np.load(split_file, allow_pickle=True)[category].tolist()
        self.mode = mode
        self.data = self.split[mode]
        if mode == 'val':
            shuffle(self.data)
            print('Warning: Validation split have been Shuffled')
        self.res = res

        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.voxelized_pointcloud = voxelized_pointcloud
        self.pointcloud_samples = pointcloud_samples
        self.use_sdf = use_sdf
        self.path2samples = '_sdf' if self.use_sdf else '' #sdf of sampled points are saved as f'/boundary_{sigma}_samples_sdf.npz'

        # compute number of samples per sampling method
        self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)
        self.matching_model = matching_model
        self.noisy = noisy
        self.std_noise = std_noise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.path + self.data[idx]

        if not self.voxelized_pointcloud:
            occupancies = np.load(path + '/voxelization_{}.npy'.format(self.res))
            occupancies = np.unpackbits(occupancies)
            input = np.reshape(occupancies, (self.res,)*3)
        else:
            name = '/voxelized_point_cloud_{}res_{}points.npz' if not self.noisy else '/noisy_voxelized_point_cloud_{}res_{}_std_{}points.npz'
            voxel_path = path + name.format(self.res, self.pointcloud_samples, self.std_noise)
            occupancies = np.unpackbits(np.load(voxel_path)['compressed_occupancies'])
            input = np.reshape(occupancies, (self.res,)*3)

        points = []
        coords = []
        occupancies = []

        for i, num in enumerate(self.num_samples):
            boundary_samples_path = path + '/boundary_{}_samples{}.npz'.format(self.sample_sigmas[i], self.path2samples)
            boundary_samples_npz = np.load(boundary_samples_path)
            boundary_sample_points = boundary_samples_npz['points']
            boundary_sample_coords = boundary_samples_npz['grid_coords']
            boundary_sample_occupancies = boundary_samples_npz['occupancies'] # occupancies or sdfs
            subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
            if self.mode == 'val':
                subsample_indices = np.linspace(0, len(boundary_sample_points)-1, num, dtype=int)
            points.extend(boundary_sample_points[subsample_indices])
            #p_coords = np.concatenate((self.get_level_set(path),np.array(coords, dtype=np.float32))) if self.matching_model else np.array(coords, dtype=np.float32)
            coords.extend(boundary_sample_coords[subsample_indices])
            occupancies.extend(boundary_sample_occupancies[subsample_indices])

        assert len(points) == self.num_sample_points
        assert len(occupancies) == self.num_sample_points
        assert len(coords) == self.num_sample_points
        p_coords = np.concatenate((self.get_level_set(path),np.array(coords, dtype=np.float32))) if self.matching_model else np.array(coords, dtype=np.float32)
        return {'grid_coords':p_coords,'occupancies': np.array(occupancies, dtype=np.float32),'points':np.array(points, dtype=np.float32), 'inputs': np.array(input, dtype=np.float32), 'path' : path}

    def get_loader(self, subset = None, shuffle =True):
        ds = torch.utils.data.Subset(self, indices= subset) if subset is not None else self
        return torch.utils.data.DataLoader(
                ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
    def get_level_set( self, path):
        name = '/voxelized_point_cloud_{}res_{}points.npz' if not self.noisy else '/noisy_voxelized_point_cloud_{}res_{}_std_{}points.npz'
        file = path + name.format(self.res, self.pointcloud_samples, self.std_noise)

        pc = np.load(file)
        pc = pc.f.point_cloud
        p = pc.copy()
        p[:, 0], p[:, 2] = pc[:, 2], pc[:, 0]

        return 2*p