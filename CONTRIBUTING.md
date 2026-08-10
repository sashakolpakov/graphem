# Contributing to GraphEm

Contributions to the reference JAX package, the documentation, and the
reproduction tooling are welcome.

## Development setup

```bash
git clone https://github.com/YOUR_USERNAME/graphem.git
cd graphem
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,docs]"
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Required checks

Run the core Python and Sphinx gates before opening a pull request:

```bash
python -m pytest tests -v
pylint --output-format=colorized --rcfile=.github/workflows/.pylintrc $(git ls-files '*.py' ':!:setup.py')
python -c "import graphem, graphem.benchmark, graphem.datasets, graphem.generators, graphem.influence, graphem.visualization"
python build_docs.py
```

The real-import smoke runs before the documentation build. Sphinx then treats
broken references and every other warning as errors.

## Change guidelines

- Keep public APIs typed and documented with NumPy-style docstrings.
- Add focused tests for behavior changes and failure paths.
- Update every affected Markdown and Sphinx page with the code change.
- Use explicit random seeds in examples, but do not claim numerical byte
  determinism unless a contract actually requires it.
- Label radial ranking and influence selection as heuristics and compare them
  against appropriate baselines.
- Preserve content-addressed benchmark lineage when replacing a reproduction
  cell; do not overwrite earlier evidence.

## Pull-request checklist

- [ ] Tests pass locally.
- [ ] Pylint passes.
- [ ] Sphinx builds with warnings treated as errors.
- [ ] Examples and links reference the current API and repository.
- [ ] User-facing behavior and limitations are documented.
- [ ] Reproduction artifacts, if any, have independent hashes and audit records.

The extended Sphinx guide is in [docs/contributing.rst](docs/contributing.rst).
Use [GitHub Issues](https://github.com/sashakolpakov/graphem/issues) for bugs and
design proposals.
