#!/usr/bin/env python
"""
Setup script for Graphem package.
Note: This setup.py is provided for backward compatibility.
For modern Python packaging, pyproject.toml is the recommended approach.
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="graphem",
    version="0.1.0",
    description="A graph embedding library based on JAX for efficient k-nearest neighbors",
    author="Igor Rivin",
    packages=find_packages(),
    install_requires=required,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
