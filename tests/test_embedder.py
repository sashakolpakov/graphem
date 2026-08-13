"""Unit tests for graph embedder."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import networkx as nx
import scipy.sparse as sp
from scipy.stats import spearmanr
from graphem.embedder import GraphEmbedder
from graphem.generators import (
    compute_vertex_degrees,
    generate_er,
    generate_random_regular,
)


class _SilentLogger:
    """Minimal logger used by midpoint-only tests."""

    @staticmethod
    def info(_message):
        """Discard an informational message."""


def _midpoint_only_embedder(sample_size, batch_size):
    """Build the part of an embedder needed for midpoint-neighbor tests."""
    embedder = object.__new__(GraphEmbedder)
    embedder.sample_size = sample_size
    embedder.batch_size = batch_size
    embedder.logger = _SilentLogger()
    return embedder


class TestEmbedder:
    """Test graph embedder functionality."""

    def test_embedder_initialization(self):
        """Test embedder initialization."""
        adj = generate_random_regular(n=50, d=4, seed=42)

        embedder = GraphEmbedder(
            adjacency=adj,
            n_components=2,
            sample_size=256,
            verbose=False
        )

        assert embedder.n == 50
        assert embedder.n_components == 2
        assert embedder.positions.shape == (50, 2)
        assert embedder.positions is not None

    def test_embedder_dimensions(self):
        """Test embedder with different dimensions."""
        adj = generate_random_regular(n=40, d=4, seed=42)

        for dim in [2, 3, 4]:
            embedder = GraphEmbedder(
                adjacency=adj,
                n_components=dim,
                sample_size=200,
                verbose=False
            )

            assert embedder.n_components == dim
            assert embedder.positions.shape == (40, dim)

    def test_layout_execution(self):
        """Test layout algorithm execution."""
        adj = generate_random_regular(n=40, d=4, seed=42)

        embedder = GraphEmbedder(
            adjacency=adj,
            n_components=2,
            sample_size=128,
            n_neighbors=10,
            verbose=False
        )

        initial_positions = embedder.get_positions().copy()
        final_positions = embedder.run_layout(num_iterations=3)

        assert not np.array_equal(initial_positions, final_positions)
        assert final_positions.shape == (40, 2)
        assert np.all(np.isfinite(final_positions))

    def test_disconnected_graph(self):
        """Test embedder with disconnected graph."""
        # Create two disconnected triangles
        n = 6
        adj = sp.csr_matrix((n, n), dtype=int)
        # Triangle 1: vertices 0, 1, 2
        adj[0, 1] = adj[1, 0] = 1
        adj[1, 2] = adj[2, 1] = 1
        adj[2, 0] = adj[0, 2] = 1
        # Triangle 2: vertices 3, 4, 5
        adj[3, 4] = adj[4, 3] = 1
        adj[4, 5] = adj[5, 4] = 1
        adj[5, 3] = adj[3, 5] = 1

        embedder = GraphEmbedder(
            adjacency=adj,
            n_components=2,
            sample_size=6,
            verbose=False
        )

        embedder.run_layout(num_iterations=2)
        assert embedder.positions.shape == (6, 2)

    def test_layout_stability(self):
        """Test that layout runs are numerically stable."""
        adj = generate_random_regular(n=30, d=4, seed=42)

        embedder = GraphEmbedder(
            adjacency=adj,
            n_components=2,
            sample_size=64,
            verbose=False
        )

        for _ in range(3):
            embedder.run_layout(num_iterations=2)

            assert np.all(np.isfinite(embedder.positions))

            max_coord = np.max(np.abs(embedder.positions))
            assert max_coord < 1000  # Reasonable bound

    def test_large_graphs(self):
        """Test embedder with large graphs."""
        adj = generate_er(n=200, p=0.02, seed=42)

        embedder = GraphEmbedder(
            adjacency=adj,
            n_components=2,
            sample_size=512,
            batch_size=1024,
            verbose=False
        )

        assert embedder.positions.shape == (200, 2)
        assert np.all(np.isfinite(embedder.positions))

    def test_get_positions_method(self):
        """Test get_positions() returns numpy array."""
        adj = generate_random_regular(n=20, d=3, seed=42)

        embedder = GraphEmbedder(
            adjacency=adj,
            n_components=2,
            verbose=False
        )

        positions = embedder.get_positions()
        assert isinstance(positions, np.ndarray)
        assert positions.shape == (20, 2)
        assert np.all(np.isfinite(positions))

    def test_adjacency_validation(self):
        """Test adjacency matrix validation."""
        # Test with non-square matrix
        with pytest.raises(ValueError, match="must be square"):
            adj = sp.csr_matrix((5, 3), dtype=int)
            GraphEmbedder(adjacency=adj, n_components=2, verbose=False)

        # Test with empty matrix
        with pytest.raises(ValueError, match="cannot be empty"):
            adj = sp.csr_matrix((0, 0), dtype=int)
            GraphEmbedder(adjacency=adj, n_components=2, verbose=False)

    def test_seed_parameter(self):
        """Test seed parameter for reproducibility."""
        adj = generate_random_regular(n=30, d=4, seed=42)

        # Create two embedders with same seed
        embedder1 = GraphEmbedder(
            adjacency=adj,
            n_components=2,
            seed=123,
            verbose=False
        )

        embedder2 = GraphEmbedder(
            adjacency=adj,
            n_components=2,
            seed=123,
            verbose=False
        )

        pos1 = embedder1.get_positions()
        pos2 = embedder2.get_positions()

        # Positions should have same structure (eigenvectors can have sign flip)
        # Test that positions are the same up to sign and permutation
        assert pos1.shape == pos2.shape

        # Check that the positions are close (accounting for potential sign flip)
        assert np.allclose(pos1, pos2) or np.allclose(pos1, -pos2) or \
               np.allclose(pos1[:, 0], pos2[:, 0]) or np.allclose(pos1[:, 1], pos2[:, 1])

    def test_logger_instance_parameter(self):
        """Test custom logger instance parameter."""
        import logging

        custom_logger = logging.getLogger("test_logger")
        adj = generate_random_regular(n=20, d=3, seed=42)

        embedder = GraphEmbedder(
            adjacency=adj,
            n_components=2,
            logger_instance=custom_logger,
            verbose=False
        )

        assert embedder.logger == custom_logger

    def test_midpoint_knn_returns_global_ids_for_sampled_queries(self):
        """Midpoint queries return exact neighbors in the global edge namespace."""
        midpoints = jnp.asarray(
            [[0.0], [0.7], [2.1], [4.8], [8.4], [13.1], [19.7], [28.0], [38.2]],
            dtype=jnp.float32,
        )
        embedder = _midpoint_only_embedder(sample_size=3, batch_size=2)

        actual, sampled_edge_ids = embedder.locate_knn_midpoints(midpoints, 2)
        actual = np.asarray(actual)
        sampled_edge_ids = np.asarray(sampled_edge_ids)

        global_edge_ids = np.arange(midpoints.shape[0])
        host_midpoints = np.asarray(midpoints)
        expected = []
        for sampled_edge_id in sampled_edge_ids:
            delta = host_midpoints - host_midpoints[sampled_edge_id]
            distances = np.sum(delta * delta, axis=1)
            order = np.lexsort((global_edge_ids, distances))
            expected.append(order[order != sampled_edge_id][:2])

        assert actual.shape == (3, 2)
        np.testing.assert_array_equal(actual, np.asarray(expected))
        assert np.all((sampled_edge_ids >= 0) & (sampled_edge_ids < 9))
        assert np.all((actual >= 0) & (actual < 9))

    def test_midpoint_knn_removes_self_by_identity_with_duplicate_midpoints(
        self, monkeypatch
    ):
        """A tied midpoint cannot leave the sampled edge as its own neighbor."""
        monkeypatch.setattr(
            jax.random,
            "choice",
            lambda *_args, **_kwargs: jnp.asarray([1], dtype=jnp.int32),
        )
        midpoints = jnp.asarray(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [3.0, 0.0]],
            dtype=jnp.float32,
        )
        embedder = _midpoint_only_embedder(sample_size=1, batch_size=1)

        actual, sampled_edge_ids = embedder.locate_knn_midpoints(midpoints, 2)

        np.testing.assert_array_equal(sampled_edge_ids, np.asarray([1]))
        np.testing.assert_array_equal(actual, np.asarray([[0, 2]]))
        assert not np.any(np.asarray(actual) == np.asarray(sampled_edge_ids)[:, None])

    def test_midpoint_knn_limits_runtime_sample_size_to_edge_count(self):
        """A midpoint array smaller than the configured sample stays in bounds."""
        midpoints = jnp.asarray([[0.0], [1.0], [3.0]], dtype=jnp.float32)
        embedder = _midpoint_only_embedder(sample_size=99, batch_size=8)

        actual, sampled_edge_ids = embedder.locate_knn_midpoints(midpoints, 9)

        assert actual.shape == (3, 2)
        assert sampled_edge_ids.shape == (3,)
        assert np.all((actual >= 0) & (actual < 3))
        assert not np.any(np.asarray(actual) == np.asarray(sampled_edge_ids)[:, None])

    def test_corrected_midpoint_knn_preserves_seeded_er_quality(self):
        """The repaired intersection term retains the radial degree signal."""
        adjacency = generate_er(n=160, p=0.07, seed=0)
        embedder = GraphEmbedder(
            adjacency=adjacency,
            n_components=3,
            sample_size=256,
            n_neighbors=14,
            batch_size=128,
            verbose=False,
        )

        positions = embedder.run_layout(num_iterations=5)
        radii = np.linalg.norm(positions, axis=1)
        degrees = compute_vertex_degrees(adjacency)
        pagerank = np.fromiter(
            nx.pagerank(nx.from_scipy_sparse_array(adjacency)).values(),
            dtype=np.float64,
        )
        degree_correlation = spearmanr(radii, degrees).statistic
        pagerank_correlation = spearmanr(radii, pagerank).statistic
        radial_top = set(np.argsort(radii)[-20:])
        degree_top = set(np.argsort(degrees)[-20:])
        pagerank_top = set(np.argsort(pagerank)[-20:])

        assert degree_correlation >= 0.65
        assert pagerank_correlation >= 0.65
        assert len(radial_top & degree_top) >= 10
        assert len(radial_top & pagerank_top) >= 8
