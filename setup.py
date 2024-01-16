# setup
from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as req:
        return req.read().splitlines()

setup(
    name='dynamic_bias',
    version='dev',
    packages=find_packages(),
    install_requires=read_requirements(),
    extras_require={
        'torch': ['torch'],
        'tensorflow': ['tensorflow'],
        'all': ['torch', 'tensorflow']
    }
)