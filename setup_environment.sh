yes | conda install conda-forge::gxx==10.4
yes | conda install nvidia/label/cuda-11.8.0::cuda
yes | conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
yes | conda install anaconda::numpy
pip install smplx
yes | conda install conda-forge::trimesh
yes | conda install conda-forge::matplotlib
yes | conda install anaconda::pandas
yes | conda install conda-forge::tqdm
yes | conda install conda-forge::tensorboard
yes | conda install scipy

