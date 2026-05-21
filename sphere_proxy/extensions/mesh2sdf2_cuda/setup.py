## This source code is DualSDF
##   (https://github.com/zekunhao1995/DualSDF)
## Copyright (c) 2020 Zekun Hao
## This source code is licensed under the MIT license found in the
## 3rd-party-licenses.txt file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++11', '-ffast-math']

include_dirs = ["/home/hpr2hi/miniconda3/envs/sp/lib/python3.11/site-packages/nvidia/cuda_runtime/include/", "/home/hpr2hi/miniconda3/envs/sp/targets/x86_64-linux/include/"]
library_dirs = ["/home/hpr2hi/miniconda3/envs/sp/x86_64-conda-linux-gnu/"]

nvcc_args = [
    #'-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    #'-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_70,code=compute_70'
]

setup(
    name='mesh2sdf',
    ext_modules=[
        CUDAExtension('mesh2sdf', [
            'mesh2sdf_kernel.cu'
        ],
        include_dirs = include_dirs,
        library_dirs = library_dirs,
        extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args},
        extra_link_args=['-L/usr/lib/x86_64-linux-gnu/'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
    
