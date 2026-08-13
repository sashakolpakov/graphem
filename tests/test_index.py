"""Regression tests for the tiled JAX nearest-neighbor index."""

import numpy as np
import jax.numpy as jnp
import pytest

from graphem.index import HPIndex


def test_knn_tiled_breaks_distance_ties_by_global_database_id():
    """Equal distances have a stable order independent of database tiles."""
    database = jnp.asarray([[-1.0], [1.0], [0.0], [2.0]], dtype=jnp.float32)

    actual_indices, actual_distances = HPIndex.knn_tiled(
        database,
        jnp.asarray([[0.0]], dtype=jnp.float32),
        k=4,
        x_tile_size=2,
        y_batch_size=1,
    )

    np.testing.assert_array_equal(actual_indices, np.asarray([[2, 0, 1, 3]]))
    np.testing.assert_array_equal(actual_distances, np.asarray([[0.0, 1.0, 1.0, 4.0]]))


@pytest.mark.parametrize(
    ("database_size", "query_size", "tile_size", "batch_size", "k"),
    [
        (2, 1, 2, 1, 2),
        (5, 4, 2, 3, 4),
        (8, 3, 3, 2, 3),
        (11, 7, 4, 3, 5),
    ],
)
def test_knn_tiled_matches_random_brute_force(
    database_size, query_size, tile_size, batch_size, k
):
    """Tiling and query batching preserve the exact brute-force result."""
    generator = np.random.default_rng(database_size * 100 + query_size)
    database = generator.normal(size=(database_size, 3)).astype(np.float32)
    query_ids = generator.choice(database_size, size=query_size, replace=False)
    queries = database[query_ids]

    actual_indices, actual_distances = HPIndex.knn_tiled(
        jnp.asarray(database),
        jnp.asarray(queries),
        k=k,
        x_tile_size=tile_size,
        y_batch_size=batch_size,
    )

    global_ids = np.arange(database_size)
    expected_indices = []
    expected_distances = []
    for query in queries:
        distances = np.sum((database - query) ** 2, axis=1)
        order = np.lexsort((global_ids, distances))[:k]
        expected_indices.append(order)
        expected_distances.append(distances[order])

    np.testing.assert_array_equal(actual_indices, np.asarray(expected_indices))
    np.testing.assert_allclose(
        actual_distances,
        np.asarray(expected_distances),
        rtol=1e-5,
        atol=1e-6,
    )


def test_knn_tiled_matches_brute_force_with_partial_database_and_query_tiles():
    """Partial tiles retain exact indices, distances, and query rows."""
    database = jnp.asarray(
        [[0.0], [0.6], [1.9], [4.1], [7.4], [11.8], [17.3], [24.9]],
        dtype=jnp.float32,
    )
    query_ids = np.asarray([1, 7, 4])
    queries = database[query_ids]

    actual_indices, actual_distances = HPIndex.knn_tiled(
        database,
        queries,
        k=3,
        x_tile_size=3,
        y_batch_size=2,
    )

    host_database = np.asarray(database)
    host_queries = np.asarray(queries)
    global_ids = np.arange(database.shape[0])
    expected_indices = []
    expected_distances = []
    for query in host_queries:
        distances = np.sum((host_database - query) ** 2, axis=1)
        order = np.lexsort((global_ids, distances))[:3]
        expected_indices.append(order)
        expected_distances.append(distances[order])

    np.testing.assert_array_equal(actual_indices, np.asarray(expected_indices))
    np.testing.assert_allclose(
        actual_distances,
        np.asarray(expected_distances),
        rtol=5e-7,
        atol=1e-6,
    )
