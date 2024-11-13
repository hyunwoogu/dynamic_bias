"""Setup script"""
from setuptools import setup, find_packages
setup(
    name='dynamic_bias',
    author='Hyunwoo Gu',
    author_email='hwgu@stanford.edu',
    license='MIT',
    version='0.0.1',
    packages=find_packages(),
    long_description=open('README.md').read(),
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Operating System :: MacOS',
    ],
)