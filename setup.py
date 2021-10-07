#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='DIPS-Plus',
    version='0.0.8',
    description='The Enhanced Database of Interacting Protein Structures for Interface Prediction',
    author='Alex Morehead',
    author_email='acmwhb@umsystem.edu',
    url='https://github.com/BioinfoMachineLearning/DIPS-Plus',
    install_requires=[
        'setuptools==56.2.0',
        'dill==0.3.3',
        'tqdm==4.49.0',
        'Sphinx==4.0.1',
        'easy-parallel-py3==0.1.6.4',
        'atom3-py3==0.1.9.8',
        'click==7.0.0',
        # mpi4py==3.0.3  # On Andes, do 'source venv/bin/activate', 'module load gcc/10.3.0', and 'pip install mpi4py --no-cache-dir --no-binary :all:'
    ],
    packages=find_packages(),
)
