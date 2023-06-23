#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='DIPS-Plus',
    version='1.2.0',
    description='The Enhanced Database of Interacting Protein Structures for Interface Prediction',
    author='Alex Morehead',
    author_email='acmwhb@umsystem.edu',
    url='https://github.com/BioinfoMachineLearning/DIPS-Plus',
    install_requires=[
        'setuptools==65.5.1',
        'dill==0.3.3',
        'tqdm==4.49.0',
        'Sphinx==4.0.1',
        'easy-parallel-py3==0.1.6.4',
        'click==7.0.0',
    ],
    packages=find_packages(),
)
