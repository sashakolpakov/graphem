Contributing to GraphEm
======================

We welcome contributions to GraphEm! This guide will help you get started with development, testing, and submitting high-quality contributions.

Development Environment Setup
-----------------------------

Prerequisites
~~~~~~~~~~~~~

Before contributing, ensure you have:

* Python 3.8+ (Python 3.9+ recommended)
* Git
* A virtual environment manager (venv, conda, or similar)

Initial Setup
~~~~~~~~~~~~~

1. **Fork and clone the repository:**

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/graphem.git
   cd graphem
   git remote add upstream https://github.com/sashakolpakov/graphem.git

2. **Create and activate a virtual environment:**

.. code-block:: bash

   python -m venv graphem-dev
   source graphem-dev/bin/activate  # On Windows: graphem-dev\Scripts\activate

3. **Install in development mode:**

.. code-block:: bash

   pip install --upgrade pip
   pip install -e ".[docs]"  # Install documentation


Testing Your Installation
~~~~~~~~~~~~~~~~~~~~~~~~~

Verify your setup works correctly:

.. code-block:: bash

   # Run basic tests
   python -c "import graphem as ge; print('Import successful')"
   
   # Run example scripts
   python examples/graph_generator_test.py
   python examples/real_world_datasets_test.py
   
   # Run benchmarks (takes a few minutes)
   python run_benchmarks.py

Code Style and Quality
----------------------

Coding Standards
~~~~~~~~~~~~~~~~

GraphEm follows these coding standards:

* **PEP 8** for Python style
* **Type hints** for all public functions
* **NumPy-style docstrings** for documentation
* **JAX best practices** for numerical computation
* **Consistent naming**: snake_case for functions, PascalCase for classes


Documentation Guidelines
------------------------

Writing Documentation
~~~~~~~~~~~~~~~~~~~~~

All public functions must have comprehensive docstrings:

.. code-block:: python

   def complex_function(param1: int, param2: str, optional_param: float = 1.0) -> Tuple[np.ndarray, Dict[str, Any]]:
       """
       One-line summary of the function.
       
       Longer description explaining what the function does, when to use it,
       and any important details about the implementation or algorithm.
       
       Parameters
       ----------
       param1 : int
           Description of the first parameter.
       param2 : str
           Description of the second parameter.
       optional_param : float, optional
           Description of optional parameter, by default 1.0.
           
       Returns
       -------
       Tuple[np.ndarray, Dict[str, Any]]
           Description of return values:
           - First element: array description
           - Second element: dictionary description
           
       Raises
       ------
       ValueError
           When parameter validation fails.
       RuntimeError
           When computation fails.
           
       Examples
       --------
       >>> result_array, result_dict = complex_function(10, "test")
       >>> print(result_array.shape)
       (10, 2)
       
       Notes
       -----
       Any additional notes about the algorithm, complexity, or usage.
       
       References
       ----------
       .. [1] Author, "Paper Title", Journal, Year.
       """

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd docs
   make clean  # Clean previous builds
   make html   # Build HTML documentation
   
   # View locally
   open _build/html/index.html  # macOS
   xdg-open _build/html/index.html  # Linux

Adding Examples
~~~~~~~~~~~~~~~

When adding examples:

1. **Create complete, runnable scripts** in ``examples/``
2. **Include in documentation** with explanation
3. **Test that examples work** in CI


PR Guidelines
~~~~~~~~~~~~~

* **Clear title** describing the change
* **Detailed description** explaining:
  - What the change does
  - Why it's needed
  - How it works
  - Any breaking changes
* **Link issues** if applicable
* **Include tests** for new functionality
* **Update documentation** for user-facing changes

Example PR template:

.. code-block:: markdown

   ## Summary
   Brief description of what this PR does.

   ## Changes
   - Add new graph generator for hierarchical networks
   - Update documentation with examples
   - Add comprehensive tests

   ## Testing
   - [ ] Benchmarks run successfully
   - [ ] Documentation builds without errors

   ## Breaking Changes
   None / List any breaking changes