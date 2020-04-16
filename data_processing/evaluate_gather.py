from glob import glob
import pickle as pkl
import numpy as np
import argparse
import traceback
import os
import pandas as pd

repair = False

if __name__ == '__main__' and not repair:
    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )



    parser.add_argument('-generation_path', type=str)
    parser.add_argument('-voxel_input', dest='voxel_input', action='store_true')
    parser.add_argument('-pc_input', dest='voxel_input', action='store_false')
    parser.set_defaults(voxel_input=True)
    parser.add_argument('-res',type=int)
    parser.add_argument('-points',type=int)

    args = parser.parse_args()


    if args.voxel_input:
        input_name = "eval_voxelization_{}".format(args.res)
    else:
        input_name = "eval_pointcloud_{}".format(args.points)

    generation_paths = glob(args.generation_path + "/*/*/")

    gt_data_path = 'shapenet/data/'





    eval_all = {
        'path' : [],
        'reconst_completeness': [],
        'reconst_accuracy': [],
        'reconst_normals completeness': [],
        'reconst_normals accuracy': [],
        'reconst_normals': [],
        'reconst_completeness2': [],
        'reconst_accuracy2': [],
        'reconst_chamfer_l2': [],
        'reconst_iou' : [],
        'input_completeness': [],
        'input_accuracy': [],
        'input_normals completeness': [],
        'input_normals accuracy': [],
        'input_normals': [],
        'input_completeness2': [],
        'input_accuracy2': [],
        'input_chamfer_l2': [],
        'input_iou': []
    }

    eval_all_avg = {
    }


    for path in generation_paths:
        try:
            norm_path = os.path.normpath(path)
            folder = norm_path.split(os.sep)[-2]
            file_name = norm_path.split(os.sep)[-1]


            eval_reconst = pkl.load(open(path + '/eval.pkl','rb'))
            eval_input = pkl.load(open(gt_data_path + '/{}/{}/{}.pkl'.format(folder, file_name, input_name),'rb'))

            eval_all['path'].append(path)

            for key in eval_reconst:
                if key == 'chamfer':
                    eval_all['reconst_chamfer_l2'].append( 0.5 * (eval_reconst['accuracy2'] + eval_reconst['completeness2']))
                else:
                    eval_all['reconst_' + key].append(eval_reconst[key])

            for key in eval_input:
                eval_all['input_' + key].append(eval_input[key])

        except Exception as err:
            # logger.exception('Path: >>>{}<<<'.format(data['path'][0]))
            print('Error with {}: {}'.format(path, traceback.format_exc()))

    pkl.dump(eval_all, open(args.generation_path + '/../evaluation_results.pkl', 'wb'))

    for key in eval_all:
        if not key == 'path':
            data = np.array(eval_all[key])
            data = data[~np.isnan(data)]
            eval_all_avg[key+'_mean'] = np.mean(data)
            eval_all_avg[key + '_median'] = np.median(data)

    pkl.dump(eval_all_avg, open(args.generation_path + '/../evaluation_results_avg.pkl', 'wb'))

    eval_df = pd.DataFrame(eval_all_avg ,index=[0])
    eval_df.to_csv( args.generation_path + '/../evaluation_results.csv')

def repair_nans(path):

    pkl_file = pkl.load(open(path))

    for key in pkl_file:

        arr = np.array(pkl_file[key])
        arr = arr[~np.isnan(arr)]
        pkl_file[key] = arr

    eval_avg = {}

    for key in pkl_file:
        eval_avg[key] = pkl_file[key].sum() / len(pkl_file[key])

    pkl.dump(pkl_file , open(os.path.dirname(path) + '/eval_repaired.pkl', 'wb'))
    pkl.dump(eval_avg , open(os.path.dirname(path) + '/eval_avg_repaired.pkl', 'wb'))

