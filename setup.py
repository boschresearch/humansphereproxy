import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='Human Sphere Proxy',
      version='1.0',
      description='Approximate meshes using a collection of spheres',
      long_description=long_description,
      author='Pascal Herrmann',
      author_email='pascal.herrmann@de.bosch.com',
      packages=[
          'sphere_proxy',
          'sphere_proxy.data_generation',
          'sphere_proxy.data_loader',
          'sphere_proxy.eval',
          'sphere_proxy.extensions',
          'sphere_proxy.models',
          'sphere_proxy.models',
          'sphere_proxy.train',
          'sphere_proxy.util',
          'sphere_proxy.visualize'
          ])