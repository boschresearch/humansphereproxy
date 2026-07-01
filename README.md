# Human Sphere Proxy

This is the companion code for the paper
"Self-Intersection-Aware 3D Human Motion Generation Using an Efficient Human Sphere Proxy" by Pascal Herrmann et al.,
published at BMVC 2025. The paper can be found [here](https://arxiv.org/abs/2605.26744). The code allows the users to
train a human sphere proxy and calculate the novel self-intersection loss reported in the paper.
The results from the paper can be reproduced by integrating the self-intersection loss
into the human motion generation methods [MDM](https://github.com/GuyTevet/motion-diffusion-model) and [MoMask](https://github.com/EricGuo5513/momask-codes).
Please cite the above paper when reporting, reproducing or extending the results.

```
@inproceedings{
herrmann_2025_BMVC,
title={Self-Intersection-Aware 3D Human Motion Generation Using an Efficient Human Sphere Proxy},
author={Herrmann, Pascal and Bieshaar, Maarten and Mack, Dennis and Herzog, Paul Robert and Gall, Juergen},
booktitle={BMVC},
year={2025},
volume={36},
address = {Sheffield, UK}
}
```

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication "Self-Intersection-Aware 3D Human Motion Generation Using an Efficient Human Sphere Proxy".
It will neither be maintained nor monitored in any way.

## Setup
Create conda environment

```
yes | conda create -n sp python=3.11
conda activate sp
./setup_environment.sh
```

Build HumanML3D_to_SMPL_layer (optional; Used to sample SMPL meshes)
```
cd sphere_proxy/humanml3d_to_smpl_layer
pip install .
```

Build custom cuda kernels

```
cd ../extensions/mesh2sdf2_cuda
make
```

Build sphere proxy package
```
cd ../../..
pip install .
```

Obtain the gender-neutral SMPL model. Details are in sphere_proxy/body_models/README.md

### Create training data

#### Sample SMPL meshes

```
python -m data_generation.sample_smpl --num_samples 10 --out_dir ./dataset/smpl_samples/ --sample_pose --max_rot 30 --num_minibatches 1
```

Arguments:
--num_samples N : Set the number of meshes to sample
--out_dir ./path/ : Set the path where the meshes are saved (ideally in dataset folder)
--sample_pose : Flag to sample random poses. If not set, only use default pose
--max_rot deg : If poses are samples, set the maximal rotation angle (in degrees)
--num_minibatches m : If num_samples is big, the script uses a lot of memory. Setting m divides the sampling into batches which reduces the memory load.

#### Sample SDF values for those SMPL meshes
```
python -m data_generation.sample_sdfs_smpl ./dataset/smpl_samples/triangles/0/ ./dataset/sdf_smpl_samples/ --resume 0
```

Arguments:
mesh_npy_path : Path to the triangle dir of the previous step
output_path : Path to where the sdf samples should be saved (ideally in dataset folder)
--resume i : If the script was terminated, resume from index i

### Generate split files used for training and evaluation
```
python -m data_generation.create_split_files --split 0.8 0.15 0.05 --data_pth ./dataset/sdf_smpl_samples/
```

Arguments:
--split: [train, test, val], set the percentages for the split
--data_pth: Path to where the sdf samples are saved


## Train the sphere proxy

First, check the config folder and adjust the settings as necessary. Especially, check the data path.

### Train the joint regressor

```
python -m train.train_joint_regressor config/config_training.yaml
```

### Train the sphere regressor

```
python -m train.train_sphere_regressor config/config_training.yaml
```

## Obtain boneweight matrices and collision matrices

```
python -m train.calculate_boneweights save/<reg-dir>/config_training.yaml 
```

Arguments:
config_pth: Set the path to the config in the save dir of the trained regressors

```
python -m train.calculate_collision_ids --humanml_dir path_to_humanml3d --reg_dir save/<reg-dir>
```

Arguments:
--humanml_dir: Path to the HumanML3D dataset
--reg_dir: Path to the save dir of the sphere proxy for which collisions should be calculated

- Functionality
    - Train a sphere proxy
        - Generate data
            - unposed
                - Train the sphere regressor
                - Train the joint regressor
                - Determine the boneweight matrix
            - posed
                - Train the sphere regressor
        - Determine collision reduction matrix

    - Use the sphere proxy (given trained checkpoints)
        - Pose sphere proxy
        - Calculate selfintersections
        - Calculate selfintersection metric




## Folder structure
human_sphere_proxy
|__README.md
|__setup.py
|__sphere_proxy
    |__ config                      <!--- Config files for training the sphere proxy --->
    |__ data_generation
        |__ sample_smpl.py          <!--- Sample SMPL and save joint locations, poses, shape parameters, and triangles --->
        |__ sample_sdfs_smpl.py     <!--- Given the samples SMPL data, sample SDF values for these meshes --->
        |__ create_split_files.py   <!--- Given generate data, assign each sample a train, test, and val set --->
    |__ data_loader
        |__ shape_joint_dataset.py  <!--- PyTorch dataset to handle the resulting dataset --->
    |__ dataset                     <!--- Empty; Will save results from datageneration --->
    |__ eval
        |__ metrics.py              <!--- Contains metrics to evaluate selfintersections --->
    |__ extensions                  <!--- Custom CUDA kernels --->
    |__ humanml3d_to_smpl_layer     <!--- Get SMPL joint rotations from HumanML3D feature vector--->
    |__ models
        |__ joint_regressor.py      <!--- Given SMPL shape, predict SMPL joint locations --->
        |__ sphere_regressor.py     <!--- Given SMPL shape, predict sphere locations --->
        |__ losses.py               <!--- Losses to train the sphere proxy --->
        |__ sphere_proxy.py         <!--- Putting everything together --->
    |__ train
        |__ train_joint_regressor.py   <!--- Train the joint regressor--->
        |__ train_sphere_regressor.py   <!--- Train the sphere regressor--->


## Relevant repositories
- DualSDF

## License

Human Sphere Proxy is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in Human Sphere Proxy, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).
