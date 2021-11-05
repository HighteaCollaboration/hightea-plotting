from setuptools import setup, find_namespace_packages
from src.hightea.plotting import __version__

setup(
    name = "hightea-plotting",
    version = __version__,
    author = "Hightea Collaboration",
    author_email = "hightea@hep.phy.cam.ac.uk",
    url = "https://github.com/HighteaCollaboration/hightea-plotting",
    description = "Plotting routines for hightea project",
    license = "MIT",
    packages=find_namespace_packages(
        where='src',
        # exclude=[
        #     'hightea.plotting.stripper'
        #     ]
        ),
    package_dir={
        '': 'src',
    },
    install_requires = [
        "pathlib",
        "matplotlib",
        "numpy",
        "pandas",
        "more-itertools",
    ],
    python_requires = ">=3.6",
)
