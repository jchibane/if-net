import trimesh
import numpy as np
import implicit_waterproofing as iw
import glob
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import os
import traceback
from mesh_to_sdf import mesh_to_voxels,sample_sdf_near_surface, mesh_to_sdf as mesh2sdf
import sys
ROOT = 'shapenet/data'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_DEVICE_ID'] = '0'
def boundary_sampling_sdf(path):
    try:

        if os.path.exists(path +'/boundary_{}_samples_sdf.npz'.format(args.sigma)):
            return
        
        off_path = path + '/isosurf_scaled.off'
        out_file = path +'/boundary_{}_samples_sdf.npz'.format(args.sigma)
        print()
        mesh = trimesh.load(off_path)
        points = mesh.sample(sample_num)

        boundary_points = points + args.sigma * np.random.randn(sample_num, 3)
        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]

        grid_coords = 2 * grid_coords
        #occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]
        sdf = mesh2sdf(mesh, boundary_points, surface_point_method='scan', 
                            sign_method='normal', bounding_radius=None, 
                            scan_count=100, scan_resolution=400, 
                            sample_point_count=10000000, normal_sample_count=11)
        np.savez(out_file, points=boundary_points, occupancies = sdf, grid_coords= grid_coords)
        print('Finished {}'.format(path))
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run boundary sampling SDF'
    )
    parser.add_argument('-sigma', type=float)

    args = parser.parse_args()
    sample_num = 100000

    files = glob.glob( ROOT + '/*/*/')
    print('Files to sample from:', len(files))
    import time; time.sleep(2)
    #sys.exit()
    p = Pool(mp.cpu_count())
    p.map(boundary_sampling_sdf, files)
