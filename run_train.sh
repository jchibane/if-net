python data_processing/convert_to_scaled_off.py -data train
python data_processing/voxelized_pointcloud_sampling.py -res 128 -num_points 300 -data train
python data_processing/boundary_sampling.py -sigma 0.1 -data train
python data_processing/boundary_sampling.py -sigma 0.01 -data train
python data_processing/create_pc_off.py -res 128 -num_points 300 -data train
