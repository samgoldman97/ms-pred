from setuptools import setup
from setuptools import find_packages
import numpy as np

setup(
    include_dirs=[np.get_include()],
    packages=find_packages(where="src"),
    package_dir={'': "src"},
    include_package_data=True,
    version='0.0.1',
)

#setup(name='ms_pred',
#      packages=find_packages(where="src"),
#      package_dir={'': "src"}
#      )
#