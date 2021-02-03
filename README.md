
# Implict Feature Networks
> Implicit Functions in Feature Space for Shape Reconstruction and Completion <br />
> [Julian Chibane](http://virtualhumans.mpi-inf.mpg.de/people/Chibane.html), [Thiemo Alldieck](http://virtualhumans.mpi-inf.mpg.de/people/alldieck.html), [Gerard Pons-Moll](http://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html)

![Teaser](teaser.gif)

[Paper](https://virtualhumans.mpi-inf.mpg.de/papers/chibane20ifnet/chibane20ifnet.pdf) - 
[Supplementaty](https://virtualhumans.mpi-inf.mpg.de/papers/chibane20ifnet/chibane20ifnet_supp.pdf) -
[Project Website](https://virtualhumans.mpi-inf.mpg.de/ifnets/) -
[Arxiv](https://arxiv.org/abs/2003.01456) -
[Video](https://youtu.be/cko07jINRZg) -
Published in CVPR 2020.


#### Citation

If you find our code or paper usful for your project, please consider citing:

    @inproceedings{chibane20ifnet,
        title = {Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion},
        author = {Chibane, Julian and Alldieck, Thiemo and Pons-Moll, Gerard},
        booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {jun},
        organization = {{IEEE}},
        year = {2020},
    }

## Install

A linux system with cuda 9.0 is required for the project.

The `if-net_env.yml` file contains all necessary python dependencies for the project.
To conveniently install them automatically with [anaconda](https://www.anaconda.com/) you can use:
```
conda env create -f if-net_env.yml
conda activate if-net
```

Please clone the repository and navigate into it in your terminal, its location is assumed for all subsequent commands.

> This project uses libraries for [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks) by [Mescheder et. al. CVPR'19] 
> and the ShapeNet data preprocessed for [DISN](https://github.com/Xharlie/DISN) by [Xu et. al. NeurIPS'19], please also cite them if you use our code.

Install the needed libraries with:
```
cd data_processing/libmesh/
python setup.py build_ext --inplace
cd ../libvoxelize/
python setup.py build_ext --inplace
cd ../..
```

## Data Preparation
The full prepared data will take up 800 GB in total. Download the [ShapeNet](https://www.shapenet.org/) data preprocessed by [Xu et. al. NeurIPS'19] from [here](https://drive.google.com/drive/folders/1QGhDW335L7ra31uw5U-0V7hB-viA0JXr)
into the `shapenet` folder.

Now extract the files into `shapenet\data` with:

```
ls shapenet/*.tar.gz |xargs -n1 -i tar -xf {} -C shapenet/data/
```

Next, the inputs and training point samples for IF-Nets are created. The following three commands can be run in parallel on multiple machines to significantly increase speed.
First, the data is converted to the .off-format and scaled using
```
python data_processing/convert_to_scaled_off.py
```
The input data for Voxel Super-Resolution of voxels is created with
```
python data_processing/voxelize.py -res 32
```
using `-res 32` for 32<sup>3</sup> and `-res 128` for 128<sup>3</sup> resolution.

The input data for Point Cloud Completion is created with
```
python data_processing/voxelized_pointcloud_sampling.py -res 128 -num_points 300
```
using `-num_points 300` for point clouds with 300 points and `-num_points 3000` for 3000 points.

Training input points and the corresponding ground truth occupancy values are generated with
```
python data_processing/boundary_sampling.py -sigma 0.1
python data_processing/boundary_sampling.py -sigma 0.01
```
where `-sigma` specifies the standard deviation of the normally distributed displacements added onto surface samples.

In order to remove meshes that could not be preprocessed (should not be more than around 15 meshes) you should run
```
python data_processing/filter_corrupted.py -file 'voxelization_32.npy' -delete
```
The input data can be visualized by converting them to .off-format using
```
python data_processing/create_voxel_off.py -res 32
```
for voxel input and 
```
python data_processing/create_pc_off.py -res 128 -num_points 300
```
where `-res` and `-num_points` matches the values from the previous steps.

## Training
The training of IF-Nets is started running
```
python train.py -std_dev 0.1 0.01 -res 32 -m ShapeNet32Vox -batch_size 6
```
where `-std_dev` indicates the sigmas to use, `-res` the input resolution (32<sup>3</sup> or 128<sup>3</sup>), `-m` the IF-Net model setup
+ ShapeNet32Vox for 32<sup>3</sup> voxel Super-Resolution experiment
+ ShapeNet128Vox for 128<sup>3</sup> voxel Super-Resolution experiment
+ ShapeNetPoints for Point Cloud Completion experiments
+ SVR for 3D Single-View Reconstruction

and `-batch_size` the number of different meshes inputted in a batch, each with 50.000 point samples (=6 for small GPU's). 
If you want to train with point cloud input please add `-pointcloud` and `-pc_samples` followed by the number of point samples used, e.g. `-pc_samples 3000`.
Consider using the highest possible `batch_size` in order to speed up training.

In the `experiments/` folder you can find an experiment folder containing the model checkpoints, the checkpoint of validation minimum, and a folder containing a tensorboard summary, which can be started at with
```
tensorboard --logdir experiments/YOUR_EXPERIMENT/summary/ --host 0.0.0.0
```
## Generation
The command
```
python generate.py -std_dev 0.1 0.01 -res 32 -m ShapeNet32Vox -checkpoint 10 -batch_points 400000
```
generates the reconstructions of the, during training unseen, test examples from ShapeNet into  the folder 
```experiments/YOUR_EXPERIMENT/evaluation_CHECKPOINT_@256/generation```.
With `-checkpoint` you can choose the IF-Net model checkpoint. Use the model with minimum validation error for this, 
`-batch_points` indicates the number of points that fit into GPU memory at once (400k for small GPU's). Please also add all parameters set during training. 
> The generation script can be run on multiple machines in parallel in order to increase generation speed significantly. Also, consider using the maximal batch size possible for your GPU.
## Evaluation
Please run

```
python data_processing/evaluate.py -reconst -generation_path experiments/iVoxels_dist-0.5_0.5_sigmas-0.1_0.01_v32_mShapeNet32Vox/evaluation_10_@256/generation/
```
to evaluate each reconstruction, where `-generation_path` is the path to the reconstructed objects generated in the previous step.
> The above evaluation script can be run on multiple machines in parallel in order to increase generation speed significantly.

Then run
```
python data_processing/evaluate.py -voxels -res 32
```
 to evaluate the quality of the input. For voxel girds use '-voxels' with '-res' to specify the input resolution and for point clouds use '-pc' with '-points' to specify the number of points.

The quantitative evaluation of all reconstructions and inputs are gathered and put into `experiment/YOUR_EXPERIMENT/evaluation_CHECKPOINT_@256` using

```
python data_processing/evaluate_gather.py -voxel_input -res 32 -generation_path experiments/iVoxels_dist-0.5_0.5_sigmas-0.1_0.01_v32_mShapeNet32Vox/evaluation_10_@256/generation/
```
where you should use `-voxel_input` for Voxel Super-Resolution experiments, with `-res` specifying the input resolution or `-pc_input` for Point Cloud Completion, with `-points` specifying the number of points used.

## Pretrained Models

Pretrained models can be found [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/rdBogFjm3LSxYGy).

## Contact

For questions and comments regarding the code please contact [Julian Chibane](http://virtualhumans.mpi-inf.mpg.de/people/Chibane.html) via mail. (See Paper)

## License
Copyright (c) 2020 Julian Chibane, Max-Planck-Gesellschaft

Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.
For commercial inquiries, please see above contact information.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the `Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion` paper in documents and papers that report on research using this Software.
