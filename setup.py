from io import open
from setuptools import find_packages, setup

setup(
    name="blonde",
    version='0.1.2',
    author="Yuchen Eleanor Jiang",
    author_email="eleanorjiang630@gmail.com",
    description="PyTorch implementation of BlonDe score",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='document metric machine translation',
    license='MIT',
    url="https://github.com/EleanorJiang/blonde",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=['spacy',
                      'pandas>=1.0.1',
                      'numpy',
                      'requests',
                      'tqdm>=4.31.1',
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
