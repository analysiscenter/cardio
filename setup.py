"""
CardIO is a library that works with electrocardiograms.
Documentation - https://analysiscenter.github.io/cardio/
"""

from setuptools import setup, find_packages
from os import path
import re
#from pypandoc import convert_file


here = path.abspath(path.dirname(__file__))

with open('dataset/__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)


#long_description = convert_file(path.join(here, 'README.md'), 'rst')



setup(
    name='cardio',
    packages=find_packages(exclude=['tutorials']),
    version=version,
    url='https://github.com/analysiscenter/cardio',
    license='Apache License 2.0',
    author='Data Analysis Center',
    author_email='rhudor@gmail.com',
    description='A framework for deep research of electrocardiograms',
    #long_description=long_description,
    zip_safe=False,
    platforms='any',
    install_requires=[
        'numpy>=1.10',
        'wfdb>=1.2.2.'
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.4'],
        'tensorflow-gpu': ['tensorflow-gpu>=1.13'],
        'keras': ['keras>=2.0.0']
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering'
    ],
)