import data_processing.implicit_waterproofing as iw
import mcubes
import trimesh
import torch
import os
from glob import glob
import numpy as np

class Generator(object):
    def __init__(self, model, threshold, exp_name, checkpoint = None, device = torch.device("cuda"), resolution = 16, batch_points = 1000000):
        self.model = model.to(device)
        self.model.eval()
        self.threshold = threshold
        self.device = device
        self.resolution = resolution
        self.resolution = resolution
        self.checkpoint_path = os.path.dirname(__file__) + '/../experiments/{}/checkpoints/'.format( exp_name)
        self.load_checkpoint(checkpoint)
        self.batch_points = batch_points

        self.min = -0.5
        self.max = 0.5


        grid_points = iw.create_grid_points_from_bounds(self.min, self.max, self.resolution)
        grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()

        a = self.max + self.min
        b = self.max - self.min

        grid_coords = 2 * grid_points - a
        grid_coords = grid_coords / b

        grid_coords = torch.from_numpy(grid_coords).to(self.device, dtype=torch.float)
        grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to(self.device)
        self.grid_points_split = torch.split(grid_coords, self.batch_points, dim=1)


    def generate_mesh(self, data):


        inputs = data['inputs'].to(self.device)


        logits_list = []
        for points in self.grid_points_split:
            with torch.no_grad():
                logits = self.model(points,inputs)
            logits_list.append(logits.squeeze(0).detach().cpu())

        logits = torch.cat(logits_list, dim=0)

        return logits.numpy()
        logits = np.reshape(logits.numpy(), (self.resolution,)*3)

        #padding to be able to retrieve object close to bounding box bondary
        logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        vertices, triangles = mcubes.marching_cubes(
            logits, threshold)

        #remove translation due to padding
        vertices -= 1

        #rescale to original scale
        step = (self.max - self.min) / (self.resolution - 1)
        vertices = np.multiply(vertices, step)
        vertices += [self.min, self.min, self.min]

        mesh = trimesh.Trimesh(vertices, triangles)
        return mesh

    def mesh_from_logits(self, logits):
        logits = np.reshape(logits, (self.resolution,) * 3)

        # padding to ba able to retrieve object close to bounding box bondary
        logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        vertices, triangles = mcubes.marching_cubes(
            logits, threshold)

        # remove translation due to padding
        vertices -= 1

        # rescale to original scale
        step = (self.max - self.min) / (self.resolution - 1)
        vertices = np.multiply(vertices, step)
        vertices += [self.min, self.min, self.min]

        return trimesh.Trimesh(vertices, triangles)

    def load_checkpoint(self, checkpoint):
        if checkpoint is None:
            checkpoints = glob(self.checkpoint_path+'/*')
            if len(checkpoints) == 0:
                print('No checkpoints found at {}'.format(self.checkpoint_path))

            checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=int)
            checkpoints = np.sort(checkpoints)
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])
        else:
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoint)
        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])