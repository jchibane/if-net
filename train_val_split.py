import numpy as np
import os
from glob import glob
import random

if __name__ == '__main__':
    ROOT1 = '../SHARP_data/track1/train_partial'
    train_files = glob(ROOT1 + '/*/*scaled.off')
    train_paths = []
    for file in train_files:
        train_paths.append(os.path.splitext(file)[0])
    train_folders = np.array(train_paths, dtype = np.str_)
    
    random.shuffle(train_folders)
    train = train_folders[:int(train_folders.shape[0]*0.8)]
    val = train_folders[int(train_folders.shape[0]*0.8):]

    ROOT2 = '../SHARP_data/track1/test_partial'
    test_files = glob(ROOT2 + '/*/*scaled.off')
    test_paths = []
    for file in test_files:
        test_paths.append(os.path.splitext(file)[0])
    test = np.array(test_paths, dtype = np.str_)
    
    np.savez("/cluster/project/infk/courses/252-0579-00L/group20/SHARP_data/track1/split.npz", train = train, val = val, test = test)

    

