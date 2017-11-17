"""
CardIO is a library that works with electrocardiograms.
Documentation - https://analysiscenter.github.io/cardio/
"""

from setuptools import setup, find_packages
import re

with open('cardio/__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)


with open('docs/index.rst', 'r') as f:
    long_description = f.read()


setup(
    name='cardio',
    packages=find_packages(exclude=['tutorials']),
    version=version,
    url='https://github.com/analysiscenter/cardio',
    license='Apache License 2.0',
    author='Roman Kh at al',
    author_email='rhudor@gmail.com',
    description='A framework for deep research of electrocardiograms',
    long_description=long_description,
    zip_safe=False,
    platforms='any',
    install_requires=[
        'numpy>=1.10',
        'wfdb>=1.2.2.',
        'dill>=0.2.7.1',
        'pywt>=0.5.2',
        'sklearn>=0.19.0'
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.4'],
        'tensorflow-gpu': ['tensorflow-gpu>=1.4'],
        'keras': ['keras>=2.0.0'],
        'hmmlearn': ['hmmlearn>=0.2.0']
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