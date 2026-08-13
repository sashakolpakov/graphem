"""Regression checks for the tracked, executable example notebook."""

import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "examples" / "graphem_jax_notebook.ipynb"


def _load_notebook():
    """Load the notebook through the standard-library JSON parser."""
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_example_notebook_is_valid_output_free_python():
    """Keep committed cells valid, uniquely identified, and output-free."""
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert notebook["metadata"]["kernelspec"]["language"] == "python"

    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for ordinal, cell in enumerate(code_cells):
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"notebook-cell-{ordinal}", "exec")


def test_example_notebook_uses_the_current_public_api():
    """Reject the obsolete imports and class names removed by the audit."""
    notebook = _load_notebook()
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    for stale_fragment in (
        "graphem_rapids",
        "GraphEmbedderPyTorch",
        "gr.generate_",
        "%cd",
        "!git clone",
        "!pip install",
        "fraction_infected",
    ):
        assert stale_fragment not in source

    for current_fragment in (
        "ge.GraphEmbedder(adjacency=adjacency",
        "embedder.run_layout(num_iterations=LAYOUT_ITERATIONS)",
        "ge.generate_er(**GRAPH_PARAMETERS)",
        "np.lexsort((node_ids, -radii))",
        "spearmanr(radii, values).statistic",
        "list_available_datasets()",
    ):
        assert current_fragment in source


def test_example_notebook_records_every_layout_parameter():
    """Require the exploratory run record to retain all layout controls."""
    notebook = _load_notebook()
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    for parameter in (
        "n_components",
        "L_min",
        "k_attr",
        "k_inter",
        "n_neighbors",
        "sample_size",
        "batch_size",
        "seed",
        "num_iterations",
    ):
        assert f'"{parameter}"' in source


def test_example_notebook_executes_top_to_bottom():
    """Exercise every code cell in order with the installed CPU dependencies."""
    notebook = _load_notebook()
    namespace = {"__name__": "__graphem_example_notebook__"}

    for ordinal, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(  # pylint: disable=exec-used
                compile(source, f"notebook-cell-{ordinal}", "exec"), namespace
            )
