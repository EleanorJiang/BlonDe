#!/usr/bin/env python3
# MIT License
# Copyright (c) 2022 Yuchen Eleanor Jiang.
# You may not use this file except in compliance with the License.

"""
A setuptools based setup module.

See:
- https://packaging.python.org/en/latest/distributing.html
- https://github.com/pypa/sampleproject

To install:

1. Setup pypi by creating ~/.pypirc

        [distutils]
        index-servers =
          pypi
          pypitest

        [pypi]
        username=
        password=

        [pypitest]
        username=
        password=

2. Create the dist

   python3 setup.py sdist bdist_wheel

3. Push

   twine upload dist/*
"""

import os
import re
from io import open
from setuptools import find_packages, setup

ROOT = os.path.dirname(__file__)

def get_version():
    """
    Reads the version from blonde's __init__.py file.
    We can't import the module because required modules may not
    yet be installed.
    """
    VERSION_RE = re.compile(r'''__version__ = ['"]([0-9.]+)['"]''')
    init = open(os.path.join(ROOT, 'blonde', '__init__.py')).read()
    return VERSION_RE.search(init).group(1)


def get_long_description():
    with open('README.md') as f:
        long_description = f.read()

    with open('CHANGELOG.md') as f:
        release_notes = f.read()

    # Plug release notes into the long description
    long_description = long_description.replace(
        '# Release Notes\n\nPlease see [CHANGELOG.md](CHANGELOG.md) for release logs.',
        release_notes)

    return long_description


setup(
    name="blonde",
    version=get_version(),
    author="Yuchen Eleanor Jiang",
    author_email="eleanorjiang630@gmail.com",
    description="PyTorch implementation of BlonDe score",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    keywords='document metric machine translation',
    license='MIT',
    url="https://github.com/EleanorJiang/blonde",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=['spacy',
                      'numpy',
                      # 'pandas>=1.0.1',
                      # 'requests',
                      # 'tqdm>=4.31.1',
                      'packaging>=20.9',
                      ],
    entry_points={
        'console_scripts': [
            "blonde=blonde_cli.score:main"
        ]
    },
    include_package_data=True,
    python_requires='>=3.6',
    tests_require=['pytest'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],

)
