try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pyAMOSA",
    version="1.2.3",
    description="Python implementation of the Archived Multi-Objective Simulated Annealing optimization heuristic",
    long_description="Python implementation of the Archived Multi-Objective Simulated Annealing optimization heuristic. Take a look at https://github.com/SalvatoreBarone/pyAMOSA.",
    url="https://github.com/SalvatoreBarone/pyAMOSA",
    author="Salvatore Barone",
    author_email="salvatore.barone@unina.it",
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3.9"
    ],
    keywords="Verilator Wrapper Verilog",
    packages=["pyamosa"],
    include_package_data=True,
    install_requires=["numpy", "matplotlib", "click", "tqdm"],
    # setup_requires=["pytest-runner"],
    # tests_require=["pytest"],
    # entry_points={
    #     # If we ever want to add an executable script, this is where it goes
    # },
    project_urls={
        "Bug Reports": "https://github.com/SalvatoreBarone/pyAMOSA/issues",
        "Source": "https://github.com/SalvatoreBarone/pyAMOSA",
    },
)

