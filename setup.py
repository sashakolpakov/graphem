#!/usr/bin/env python
"""
Setup script for Graphem-JAX package.
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read the long description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Install requirements
required = ["jax>=0.3.0",
            "jaxlib>=0.3.0",
            "numpy>=1.21.0",
            "matplotlib>=3.5.0",
            "networkx>=2.6.0",
            "pandas>=1.3.0",
            "plotly>=5.5.0",
            "scipy>=1.7.0",
            "ndlib>=5.1.0",
            "loguru>=0.6.0",
            "kaleido>=0.2.1",
            "line_profiler>=4.0.0",
            "snakeviz>=2.2.0",
            "tensorboard>=2.10.0",
            "tqdm>=4.66.0",
            "pyinstrument>=5.0.0",
            "tabulate>=0.9.0"]

setup(
    name="graphem-jax",
    version="0.0.2",
    description="A graph embedding library based on JAX",
    author="Alexander Kolpakov, Igor Rivin",
    packages=find_packages(),
    install_requires=required,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
