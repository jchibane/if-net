import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
import trimesh

INPUT_PATH = 'shapenet/data'

def to_off(path):

    if os.path.exists(path + '/isosurf.off'):
        return

    input_file  = path + '/isosurf.obj'
    output_file = path + '/isosurf.off'

    cmd = 'meshlabserver -i {} -o {}'.format(input_file,output_file)
    # if you run this script on a server: comment out above line and uncomment the next line
    # cmd = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {} -o {}'.format(input_file,output_file)
    os.system(cmd)



p = Pool(mp.cpu_count())
p.map(to_off, glob.glob( INPUT_PATH + '/*/*'))

def scale(path):

    if os.path.exists(path + '/isosurf_scaled.off'):
        return

    try:
        mesh = trimesh.load(path + '/isosurf.off', process=False)
        total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
        centers = (mesh.bounds[1] + mesh.bounds[0]) /2

        mesh.apply_translation(-centers)
        mesh.apply_scale(1/total_size)
        mesh.export(path + '/isosurf_scaled.off')
    except:
        print('Error with {}'.format(path))
    print('Finished {}'.format(path))

p = Pool(mp.cpu_count())
p.map(scale, glob.glob( INPUT_PATH + '/*/*'))