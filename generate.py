import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
import numpy as np
import argparse
from models.generation import Generator
from generation_iterator import gen_iterator

parser = argparse.ArgumentParser(
    description='Run generation'
)


parser.add_argument('-pointcloud', dest='pointcloud', action='store_true')
parser.add_argument('-voxels', dest='pointcloud', action='store_false')
parser.set_defaults(pointcloud=False)
parser.add_argument('-pc_samples' , default=3000, type=int)
parser.add_argument('-dist','--sample_distribution', default=[0.5,0.5], nargs='+', type=float)
parser.add_argument('-std_dev','--sample_sigmas',default=[], nargs='+', type=float)
parser.add_argument('-res' , default=32, type=int)
parser.add_argument('-decoder_hidden_dim' , default=256, type=int)
parser.add_argument('-mode' , default='test', type=str)
parser.add_argument('-retrieval_res' , default=256, type=int)
parser.add_argument('-checkpoint', type=int)
parser.add_argument('-batch_points', default=1000000, type=int)
parser.add_argument('-m','--model' , default='LocNet', type=str)

args = parser.parse_args()

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]


if args.model ==  'ShapeNet32Vox':
    net = model.ShapeNet32Vox()

if args.model ==  'ShapeNet128Vox':
    net = model.ShapeNet128Vox()

if args.model == 'ShapeNetPoints':
    net = model.ShapeNetPoints()

if args.model == 'SVR':
    net = model.SVR()


dataset = voxelized_data.VoxelizedDataset(args.mode, voxelized_pointcloud= args.pointcloud , pointcloud_samples= args.pc_samples, res=args.res, sample_distribution=args.sample_distribution,
                                          sample_sigmas=args.sample_sigmas ,num_sample_points=100, batch_size=1, num_workers=0)


exp_name = 'i{}_dist-{}sigmas-{}v{}_m{}'.format(  'PC' + str(args.pc_samples) if args.pointcloud else 'Voxels',
                                    ''.join(str(e)+'_' for e in args.sample_distribution),
                                       ''.join(str(e) +'_'for e in args.sample_sigmas),
                                                                args.res,args.model)


gen = Generator(net,0.5, exp_name, checkpoint=args.checkpoint ,resolution=args.retrieval_res, batch_points=args.batch_points)

out_path = 'experiments/{}/evaluation_{}_@{}/'.format(exp_name,args.checkpoint, args.retrieval_res)


gen_iterator(out_path, dataset, gen)
