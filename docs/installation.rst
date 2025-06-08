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

Optional Dependencies
---------------------

For additional visualization and profiling features:

.. code-block:: bash

    # Enhanced visualization
    pip install kaleido
    
    # Profiling tools
    pip install line_profiler snakeviz pyinstrument

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

If you see import errors for optional dependencies:

1. Install missing packages: ``pip install networkx matplotlib plotly``
2. Check that all dependencies are compatible versions

Getting Help
------------

If you encounter installation issues:

1. Check our `GitHub Issues <https://github.com/sashakolpakov/graphem/issues>`_
2. Review the troubleshooting section
3. Create a new issue with your system details and error messages