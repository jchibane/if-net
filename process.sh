ls shapenet/*.tar.gz |xargs -n1 -i tar -xf {} -C shapenet/data/
python data_processing/convert_to_scaled_off.py
python data_processing/voxelize.py -res 32
python data_processing/voxelized_pointcloud_sampling.py -res 128 -num_points 300
python data_processing/boundary_sampling.py -sigma 0.1
python data_processing/boundary_sampling.py -sigma 0.01
python data_processing/filter_corrupted.py -file 'voxelization_32.npy' -delete
python data_processing/create_voxel_off.py -res 32