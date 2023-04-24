import os

from setuptools import setup


def get_version_info():
    """Read __version__ and DEV_CLASSIFIER from version.py, using exec, not import."""
    fn_version = os.path.join("ms_pred", "_version.py")
    if os.path.isfile(fn_version):
        myglobals = {}
        with open(fn_version, "r") as f:
            exec(f.read(), myglobals)  # pylint: disable=exec-used
        return myglobals["__version__"], myglobals["DEV_CLASSIFIER"]
    return "0.0.0.post0", "Development Status :: 2 - Pre-Alpha"


def get_readme():
    """Load README.rst for display on PyPI."""
    with open("README.md") as fhandle:
        return fhandle.read()


VERSION, DEV_CLASSIFIER = get_version_info()

setup(
    name="ms-pred",
    version=VERSION,
    description="The ms-pred Package",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    url="http://github.com/theochem/procrustes",
    license="MIT",
    author="Samuel Goldman",
    author_email="samlg@mit.edu",
    package_dir={"ms_pred": "src/ms_pred"},
    packages=[
        "ms_pred",
        # "ms_pred.test",
    ],
    include_package_data=True,
    classifiers=[
        DEV_CLASSIFIER,
        "Environment :: Console",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Intended Audience :: Science/Research",
    ],
    # todo: update this
    # install_requires=["numpy>=1.19.5", "scipy>=1.5.0", "pytest>=5.4.3", "sphinx>=2.3.0"],
)
