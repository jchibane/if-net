import glob
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import os
import shutil
import numpy
import numpy as np

# python data_processing/filter_corrupted.py -file 'voxelization_32.npy'

def filter(tmp_path):
    path, delete, file = tmp_path
    path = os.path.splitext(path)[0]
    if not os.path.exists('{}_{}'.format(path,file)):
        print('Remove: {}'.format(path, file))
        if delete:
            shutil.rmtree('{}'.format(path))


parser = argparse.ArgumentParser(
    description='Filter shapenet objects if preprocessing failed.'
)
parser.add_argument('-file', type=str)
parser.add_argument('-delete', action='store_true')
parser.add_argument('-data', type=str)
parser.set_defaults(delete=False)

file = parser.parse_args().file
delete = parser.parse_args().delete
data = parser.parse_args().data

if data == "train":
    ROOT = '../SHARP_data/track1/train_partial'
elif data == "test":
    ROOT = '../SHARP_data/track1/test_partial'
elif data == "test-codalab-partial":
    ROOT = '../SHARP_data/track1/test-codalab-partial'
elif data == "train_gt":
    ROOT = '../SHARP_data/track1/train'
elif data == "test_gt":
    ROOT = '../SHARP_data/track1/test'

paths = glob.glob(ROOT + '/*/*..npz')
new_paths = []
for i , path in enumerate(paths):
    new_paths.append((path, delete, file))
#p = Pool(mp.cpu_count())
#p.map(filter, new_paths)
for new_path in new_paths:
    filter(new_path)

def update_split():

    split = np.load('shapenet/split.npz')
    split_dict = {}
    for set in ['train','test','val']:
        filterd_set = split[set].copy()
        for path in split[set]:
            if not os.path.exists('shapenet/data/' + path):
                print('Filtered: ' + path)
                filterd_set = np.delete(filterd_set, np.where(filterd_set == path))
        split_dict[set] = filterd_set

    np.savez('shapenet/split.npz', **split_dict)


if delete:
    update_split()