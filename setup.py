#!/usr/bin/env python
"""
Setup script for Graphem-JAX package.
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read the long description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Install requirements
required = [
    "jax>=0.3.0",
    "jaxlib>=0.3.0",
    "numpy>=1.21.0",
    "matplotlib>=3.5.0",
    "networkx>=2.6.0",
    "pandas>=1.3.0",
    "plotly>=5.5.0",
    "scipy>=1.7.0",
    "ndlib>=5.1.0",
    "loguru>=0.6.0",
    "requests>=2.25.0",
    "line_profiler>=4.0.0",
    "snakeviz>=2.2.0",
    "tensorboard>=2.10.0",
    "tqdm>=4.66.0",
    "pyinstrument>=5.0.0",
    "tabulate>=0.9.0"
]

docs_required = [
    "sphinx>=4.0.0",
    "sphinx_rtd_theme>=1.0.0",
    "sphinx-autodoc-typehints>=1.12.0"
]

setup(
    name="graphem-jax",
    version="0.2.0",
    description="A graph embedding library based on JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alexander Kolpakov (UATX), Igor Rivin (Temple University)",
    packages=find_packages(),
    install_requires=required,
    extras_require={
        "docs": docs_required,
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
