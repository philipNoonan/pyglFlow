#!/usr/bin/env python

"""The setup script."""

import os
from setuptools import setup, find_packages

requirements = [
    'glfw',
    'PyOpenGL',
    'opencv-python',
    'imgui',
]

setup(
    author='PN',
    author_email='philip.noonan@kcl.ac.uk',
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'pyglflow=pyglflow.pyglFlow:main',
        ],
    },
    install_requires=requirements,
    include_package_data=True,
    name='pyglflow',
    packages=find_packages(include=['pyglflow', 'pyglflow.*']),
    setup_requires=[],
    test_suite='tests',
    tests_require=[],
    url='https://github.com/philipNoonan/pyglFlow',
    version='0.1.0',
    zip_safe=False,
)
