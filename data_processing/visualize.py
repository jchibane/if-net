from sharp_challenge1.data import *
import trimesh 
import os.path
input_path = r"E:\03_Uni\02_ETH\02_22_sum\3dVision\01_work\Sharp_data\Track1-3DBodyTex.v2\test-codalab-partial\170410-001-a-r9iu-494a-low-res-result\170410-001-a-r9iu-494a-low-res-result-partial.npz"
current_mesh = load_mesh(input_path)
fn = os.path.splitext(input_path)[0]
save_obj(fn+".obj", current_mesh, save_texture=True)
mesh = trimesh.load(fn+".obj")
mesh.show()

# a = np.load(r"E:\03_Uni\02_ETH\02_22_sum\3dVision\01_work\IFNet\SHARP_data\track1\test_partial\171005-005-casual-run-dydd-88dd-low-res-result\points.npz")
# print(a['occupancies'])
