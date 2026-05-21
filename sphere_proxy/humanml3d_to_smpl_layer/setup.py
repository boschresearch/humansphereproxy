import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='HumanML3D_To_SMPL_Layer',
      version='1.0',
      description='PyTorch layer which converts HumanML3D features to SMPL parameters',
      long_description=long_description,
      author='Pascal Herrmann',
      author_email='pascal.herrmann@de.bosch.com',
      packages=['humanml3d_to_smpl', 'humanml3d_to_smpl.utils'])