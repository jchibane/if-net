python data_processing/convert_to_scaled_off.py -data test
python data_processing/voxelize.py -res 32 -data test
python data_processing/voxelized_pointcloud_sampling.py -res 128 -num_points 300 -data test
python data_processing/boundary_sampling.py -sigma 0.1 -data test
python data_processing/boundary_sampling.py -sigma 0.01 -data test
python data_processing/filter_corrupted.py -file 'voxelization_32.npy' -delete -data test
python data_processing/create_voxel_off.py -res 32 -data test
python data_processing/create_pc_off.py -res 128 -num_points 300 -data test

