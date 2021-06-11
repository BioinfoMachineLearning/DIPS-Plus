#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='DIPS-Plus',
    version='0.0.2',
    description='The Enhanced Database of Interacting Protein Structures for Interface Prediction',
    author='Alex Morehead',
    author_email='alex.morehead@gmail.com',
    url='https://github.com/amorehead/DIPS-Plus',
    install_requires=['setuptools', 'dill', 'tqdm', 'easy-parallel-py3', 'atom3-py3', 'click'],
    packages=find_packages(),
)
