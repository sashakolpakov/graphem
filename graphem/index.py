"""
A JAX-based implementation for efficient k-nearest neighbors.
"""

from functools import partial
import jax
import jax.numpy as jnp


def _compute_batch_distances(y_batch, x):
    """
    Compute the squared distances between a batch of query points and all
    database points.

    Args:
        y_batch: (batch_size, d) array of query points
        x: (n, d) array of database points

    Returns:
        (batch_size, n) array of squared distances
    """
    # Get squared norms for broadcasting
    x_norms = jnp.sum(jnp.square(x), axis=1)
    y_norms = jnp.sum(jnp.square(y_batch), axis=1)
    
    # Compute distances using ||x-y||^2 = ||x||^2 + ||y||^2 - 2x·y
    # y_norms[:, None] broadcasts to (batch_size, 1)
    # x_norms[None, :] broadcasts to (1, n)
    # The resulting distances will be of shape (batch_size, n)
    
    # Matrix multiplication to get dot products
    dot_products = jnp.dot(y_batch, x.T)
    
    # Combine using the distance formula
    distances = y_norms[:, None] + x_norms[None, :] - 2 * dot_products
    
    # Ensure no negative distances due to numerical errors
    distances = jnp.maximum(distances, 0.0)
    
    return distances


class HPIndex:
    """
    A kernelized kNN index that uses batching / tiling to efficiently handle
    large datasets with limited memory usage.
    """

    def __init__(self):
        pass

    @staticmethod
    def knn_tiled(x, y, k=5, x_tile_size=8192, y_batch_size=1024):
        """
        Advanced implementation that tiles both database and query points.
        This wrapper handles the dynamic aspects before calling the JIT-compiled
        function.

        Args:
            x: (n, d) array of database points
            y: (m, d) array of query points
            k: number of nearest neighbors
            x_tile_size: size of database tiles
            y_batch_size: size of query batches

        Returns:
            (m, k) array of indices of nearest neighbors
        """
        n_x, _ = x.shape
        n_y, _ = y.shape

        # Ensure batch sizes aren't larger than the data dimensions
        x_tile_size = min(x_tile_size, n_x)
        y_batch_size = min(y_batch_size, n_y)

        # Calculate batching parameters
        num_y_batches = n_y // y_batch_size
        y_remainder = n_y % y_batch_size
        num_x_tiles = (n_x + x_tile_size - 1) // x_tile_size

        # Call the JIT-compiled implementation with concrete values
        return HPIndex._knn_tiled_jit(
            x, y, k, x_tile_size, y_batch_size,
            num_y_batches, y_remainder, num_x_tiles, n_x
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
    def _knn_tiled_jit(x, y, k, x_tile_size, y_batch_size,
                     num_y_batches, y_remainder, num_x_tiles, n_x):
        """
        JIT-compiled implementation of tiled KNN with concrete batch parameters.
        """
        n_y, d_y = y.shape
        _, d_x = x.shape

        # Initialize results
        all_indices = jnp.zeros((n_y, k), dtype=jnp.int32)
        all_distances = jnp.ones((n_y, k)) * jnp.finfo(jnp.float32).max

        # Define the scan function for processing y batches
        def process_y_batch(carry, y_batch_idx):
            curr_indices, curr_distances = carry

            # Get current batch of query points
            y_start = y_batch_idx * y_batch_size
            y_batch = jax.lax.dynamic_slice(y, (y_start, 0), (y_batch_size, d_y))

            # Initialize batch results
            batch_indices = jnp.zeros((y_batch_size, k), dtype=jnp.int32)
            batch_distances = jnp.ones((y_batch_size, k)) * jnp.finfo(jnp.float32).max

            # Define the scan function for processing x tiles within a y batch
            def process_x_tile(carry, x_tile_idx):
                batch_idx, batch_dist = carry

                # Get current tile of database points - use fixed size slices
                x_start = x_tile_idx * x_tile_size

                # Use a fixed size for the slice and then mask invalid values
                x_tile = jax.lax.dynamic_slice(
                    x, (x_start, 0), (x_tile_size, d_x)
                )

                # Calculate how many elements are actually valid
                # (This is now done without dynamic shapes)
                x_tile_actual_size = jnp.minimum(x_tile_size, n_x - x_start)

                # Compute distances between y_batch and x_tile
                tile_distances = _compute_batch_distances(y_batch, x_tile)

                # Mask out invalid indices (those beyond the actual data)
                valid_mask = jnp.arange(x_tile_size) < x_tile_actual_size
                tile_distances = jnp.where(
                    valid_mask[jnp.newaxis, :],
                    tile_distances,
                    jnp.ones_like(tile_distances) * jnp.finfo(jnp.float32).max
                )

                # Adjust indices to account for tile offset
                # Make sure indices are within bounds
                tile_indices = jnp.minimum(
                    jnp.arange(x_tile_size) + x_start,
                    n_x - 1  # Ensure indices don't go beyond n_x
                )
                tile_indices = jnp.broadcast_to(tile_indices, tile_distances.shape)

                # Merge current tile results with previous results
                combined_distances = jnp.concatenate([batch_dist, tile_distances], axis=1)
                combined_indices = jnp.concatenate([batch_idx, tile_indices], axis=1)

                # Sort and get top k
                top_k_idx = jnp.argsort(combined_distances)[:, :k]

                # Gather top k distances and indices
                new_batch_dist = jnp.take_along_axis(combined_distances, top_k_idx, axis=1)
                new_batch_idx = jnp.take_along_axis(combined_indices, top_k_idx, axis=1)

                return (new_batch_idx, new_batch_dist), None

            # Process all x tiles for this y batch
            (batch_indices, batch_distances), _ = jax.lax.scan(
                process_x_tile,
                (batch_indices, batch_distances),
                jnp.arange(num_x_tiles)
            )

            # Update overall results for this batch
            curr_indices = jax.lax.dynamic_update_slice(
                curr_indices, batch_indices, (y_start, 0)
            )
            curr_distances = jax.lax.dynamic_update_slice(
                curr_distances, batch_distances, (y_start, 0)
            )

            return (curr_indices, curr_distances), None

        # Process all full y batches
        (all_indices, all_distances), _ = jax.lax.scan(
            process_y_batch,
            (all_indices, all_distances),
            jnp.arange(num_y_batches)
        )

        # Handle y remainder with similar changes if needed
        if y_remainder > 0:
            y_start = num_y_batches * y_batch_size

            # Get and pad remainder batch
            remainder_y = jax.lax.dynamic_slice(y, (y_start, 0), (y_remainder, d_y))
            padded_y = jnp.pad(remainder_y, ((0, y_batch_size - y_remainder), (0, 0)))

            # Initialize remainder results
            remainder_indices = jnp.zeros((y_batch_size, k), dtype=jnp.int32)
            remainder_distances = jnp.ones((y_batch_size, k)) * jnp.finfo(jnp.float32).max

            # Process x tiles for the remainder batch
            def process_x_tile_remainder(carry, x_tile_idx):
                batch_idx, batch_dist = carry

                # Get current tile of database points - use fixed size slices
                x_start = x_tile_idx * x_tile_size

                # Use fixed size for the slice
                x_tile = jax.lax.dynamic_slice(
                    x, (x_start, 0), (x_tile_size, d_x)
                )

                # Calculate actual valid size
                x_tile_actual_size = jnp.minimum(x_tile_size, n_x - x_start)

                # Compute distances between padded_y and x_tile
                tile_distances = _compute_batch_distances(padded_y, x_tile)

                # Mask out invalid indices (both for y padding and x overflow)
                x_valid_mask = jnp.arange(x_tile_size) < x_tile_actual_size
                tile_distances = jnp.where(
                    x_valid_mask[jnp.newaxis, :],
                    tile_distances,
                    jnp.ones_like(tile_distances) * jnp.finfo(jnp.float32).max
                )

                # Adjust indices to account for tile offset
                tile_indices = jnp.minimum(
                    jnp.arange(x_tile_size) + x_start,
                    n_x - 1  # Ensure indices don't go beyond n_x
                )
                tile_indices = jnp.broadcast_to(tile_indices, tile_distances.shape)

                # Merge current tile results with previous results
                combined_distances = jnp.concatenate([batch_dist, tile_distances], axis=1)
                combined_indices = jnp.concatenate([batch_idx, tile_indices], axis=1)

                # Sort and get top k
                top_k_idx = jnp.argsort(combined_distances)[:, :k]

                # Gather top k distances and indices
                new_batch_dist = jnp.take_along_axis(combined_distances, top_k_idx, axis=1)
                new_batch_idx = jnp.take_along_axis(combined_indices, top_k_idx, axis=1)

                return (new_batch_idx, new_batch_dist), None

            # Process all x tiles for the remainder batch
            (remainder_indices, remainder_distances), _ = jax.lax.scan(
                process_x_tile_remainder,
                (remainder_indices, remainder_distances),
                jnp.arange(num_x_tiles)
            )

            # Only keep results for valid y remainder points
            valid_remainder_indices = remainder_indices[:y_remainder]
            
            # Update the full results with the remainder results
            all_indices = jax.lax.dynamic_update_slice(
                all_indices, valid_remainder_indices, (y_start, 0)
            )
            all_distances = jax.lax.dynamic_update_slice(
                all_distances, remainder_distances[:y_remainder], (y_start, 0)
            )

        return all_indices
