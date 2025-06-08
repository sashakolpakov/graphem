Installation Guide
==================

GraphEm supports Python 3.8+ and is available through multiple installation methods.

PyPI Installation (Recommended)
-------------------------------

The easiest way to install GraphEm is via pip:

.. code-block:: bash

    pip install graphem-jax

This will install GraphEm with all required dependencies.

Development Installation
------------------------

For development or to get the latest features:

.. code-block:: bash

    git clone https://github.com/sashakolpakov/graphem.git
    cd graphem
    pip install -e .

GPU/TPU Support
---------------

GraphEm leverages JAX for acceleration. For GPU and TPU installation instructions, please refer to the `official JAX installation guide <https://github.com/google/jax#installation>`_.

Dependencies
------------

GraphEm automatically installs all required dependencies:

* JAX & JAXLib (≥0.3.0) - Core computation backend
* NumPy (≥1.21.0) - Array operations
* NetworkX (≥2.6.0) - Graph algorithms
* Plotly (≥5.5.0) - Interactive visualization
* SciPy (≥1.7.0) - Scientific computing
* NDlib (≥5.1.0) - Network diffusion models
* Pandas (≥1.3.0) - Data structures
* And others for logging, profiling, and utilities

Documentation Dependencies
--------------------------

To build documentation locally:

.. code-block:: bash

    pip install "graphem-jax[docs]"
    cd docs
    make html

System Requirements
-------------------

**Minimum Requirements:**

* Python 3.8+
* 4GB RAM
* Modern CPU with AVX support

**Recommended for Large Graphs:**

* Python 3.9+
* 16GB+ RAM
* NVIDIA GPU with CUDA support
* SSD storage for large datasets

Verification
------------

Test your installation:

.. code-block:: python

    import graphem as ge
    
    # Generate a small test graph
    edges = ge.erdos_renyi_graph(n=100, p=0.1)
    embedder = ge.GraphEmbedder(edges, n_vertices=100)
    embedder.run_layout(num_iterations=10)
    
    print("GraphEm installation successful!")

Troubleshooting
---------------

**JAX Installation Issues**

If you encounter JAX installation problems:

1. Ensure you have a compatible Python version (3.8-3.11)
2. Update pip: ``pip install --upgrade pip``
3. Try installing JAX separately first: ``pip install jax jaxlib``

**Memory Issues**

For large graphs, consider:

1. Reducing ``batch_size`` and ``sample_size`` parameters
2. Using smaller embedding dimensions
3. Processing graphs in chunks

**Import Errors**

If you see import errors:

1. Reinstall GraphEm: ``pip uninstall graphem-jax && pip install graphem-jax``
2. Check that all dependencies are compatible versions
3. Try installing in a fresh virtual environment

Getting Help
------------

If you encounter installation issues:

1. Check our `GitHub Issues <https://github.com/sashakolpakov/graphem/issues>`_
2. Create a new issue with your system details and error messages