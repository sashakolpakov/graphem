"""
Graph Embedding functionality for Graphem.
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from loguru import logger

from graphem.index import HPIndex


class GraphEmbedder:
    """
    A class for embedding graphs using the Laplacian embedding.

    Attributes:
            edges: np.ndarray of shape (num_edges, 2)
                Array of edge pairs (i, j) with i < j.
            n_vertices: int
                Number of vertices in the graph.
            dimension: int
                Dimension of the embedding.
            L_min: float
                Minimum length of the spring.
            k_attr: float
                Attraction force constant.
            k_inter: float
                Repulsion force constant for intersections.
            knn_k: int
                Number of nearest neighbors to consider.
            sample_size: int
                Number of samples for kNN search.
            batch_size: int
                Batch size for kNN search.
            my_logger: loguru.logger
                Logger object to use for logging.
    """

    def __init__(self, edges, n_vertices, dimension=2, L_min=1.0, k_attr=0.2, k_inter=0.5, 
                 knn_k=10, sample_size=256, batch_size=1024, my_logger=None, verbose=True):
        """
        Initialize the GraphEmbedder.
        """
        self.edges = jnp.array(edges)
        self.n = n_vertices
        self.dimension = dimension
        self.L_min = L_min
        self.k_attr = k_attr
        self.k_inter = k_inter
        self.knn_k = knn_k
        self.sample_size = sample_size
        self.batch_size = batch_size
        if my_logger is None:
            logger.remove()
            sink = sys.stdout if verbose else open(os.devnull, 'w', encoding='utf-8')
            logger.add(sink, level="INFO")
            self.logger = logger
            """ System logger """
        else:
            self.logger = my_logger
            self.logger.info("Logger initialized")
        self.positions = self._laplacian_embedding()

    def _laplacian_embedding(self):
        """
        Compute the Laplacian embedding of the graph.
        """
        # Create adjacency matrix
        adjacency = np.zeros((self.n, self.n))
        for i, j in self.edges:
            adjacency[i, j] = 1
            adjacency[j, i] = 1

        # Compute degree matrix and Laplacian
        degree = np.diag(np.sum(adjacency, axis=1))
        laplacian = degree - adjacency

        # Compute eigenvectors and eigenvalues of Laplacian
        eigval, eigvec = np.linalg.eigh(laplacian)

        # Use first non-zero eigenvectors as embedding
        # (skip 0th eigenvector which corresponds to the constant vector)
        dim = min(self.dimension, self.n - 1)
        
        # Create the initial embedding
        embedding = eigvec[:, 1:dim+1]
        
        # Scale by 10 to get a reasonable initial size
        embedding = embedding * 10

        return jnp.array(embedding)

    def locate_knn_midpoints(self, midpoints, k):
        """
        Locate k nearest neighbors for each midpoint.
        """
        # Use our efficient KNN search
        knn_idx = HPIndex.knn_tiled(
            midpoints, midpoints, k=k+1, 
            x_tile_size=min(midpoints.shape[0], 8192), 
            y_batch_size=min(midpoints.shape[0], 1024)
        )
        
        # Remove self-neighbor (i.e., the first index which is the point itself)
        return knn_idx[:, 1:]

    @staticmethod
    def compute_spring_forces(positions, edges, L_min, k_attr):
        """
        Compute the spring forces between vertices.
        """
        # Initialize displacement array
        displacement = jnp.zeros_like(positions)
        
        # Loop over edges and compute spring forces
        def compute_edge_force(i, disp_acc):
            # Extract edge endpoints
            edge = edges[i]
            u, v = edge[0], edge[1]
            
            # Compute current length
            edge_vec = positions[v] - positions[u]
            length = jnp.linalg.norm(edge_vec)
            
            # Compute force direction (unit vector along the edge)
            direction = edge_vec / (length + 1e-8)  # Avoid division by zero
            
            # Compute spring force magnitude (proportional to length)
            force_mag = k_attr * (length - L_min)
            
            # Apply force to both endpoints in opposite directions
            force = force_mag * direction
            
            # Update displacements
            disp_acc = disp_acc.at[u].add(force)
            disp_acc = disp_acc.at[v].add(-force)
            
            return disp_acc
        
        # Compute forces for all edges
        displacement = jax.lax.fori_loop(
            0, edges.shape[0], compute_edge_force, displacement
        )
        
        return displacement

    @staticmethod
    def compute_intersection_forces_with_knn_index(positions, edges, knn_idx, sampled_indices, k_inter):
        """
        Compute the intersection forces between vertices and their nearest neighbors.
        """
        # Initialize displacement array
        displacement = jnp.zeros_like(positions)
        
        # Compute midpoints of all edges
        midpoints = (positions[edges[:, 0]] + positions[edges[:, 1]]) / 2
        
        # Sample uniformly at random from midpoints
        sampled_midpoints = midpoints[sampled_indices]
        
        # Loop over each sampled midpoint and its nearest neighbors
        def process_midpoint(i, disp_acc):
            midpoint_idx = sampled_indices[i]
            mid = midpoints[midpoint_idx]
            edge = edges[midpoint_idx]
            u, v = edge[0], edge[1]
            
            # loop over the k nearest neighbors
            def process_neighbor(j, inner_disp):
                # Get the neighbor edge midpoint
                neighbor_idx = knn_idx[i, j]
                neighbor_edge = edges[neighbor_idx]
                neighbor_mid = midpoints[neighbor_idx]
                
                # Avoid self-edge and immediate neighbors
                not_adjacent = (neighbor_edge[0] != u) & (neighbor_edge[0] != v) & \
                               (neighbor_edge[1] != u) & (neighbor_edge[1] != v)
                
                def apply_forces(inner_disp_apply):
                    # Compute minimum distance from midpoint to the edge
                    # First compute vectors for the edge
                    p1 = positions[neighbor_edge[0]]
                    p2 = positions[neighbor_edge[1]]
                    
                    # Repulsion force
                    force_u = GraphEmbedder.repulse(positions[u], neighbor_mid)
                    force_v = GraphEmbedder.repulse(positions[v], neighbor_mid)
                    
                    # Apply repulsion to the vertices of the original edge
                    inner_disp_apply = inner_disp_apply.at[u].add(k_inter * force_u)
                    inner_disp_apply = inner_disp_apply.at[v].add(k_inter * force_v)
                    
                    # Also apply repulsion to the vertices of the neighboring edge
                    force_p1 = GraphEmbedder.repulse(p1, mid)
                    force_p2 = GraphEmbedder.repulse(p2, mid)
                    
                    inner_disp_apply = inner_disp_apply.at[neighbor_edge[0]].add(k_inter * force_p1)
                    inner_disp_apply = inner_disp_apply.at[neighbor_edge[1]].add(k_inter * force_p2)
                    
                    return inner_disp_apply
                
                # Apply forces only if edges are not adjacent
                inner_disp = jax.lax.cond(
                    not_adjacent,
                    apply_forces,
                    lambda x: x,
                    inner_disp
                )
                
                return inner_disp
            
            # Process all neighbors of this midpoint
            disp_acc = jax.lax.fori_loop(
                0, knn_idx.shape[1], process_neighbor, disp_acc
            )
            
            return disp_acc
        
        # Process all sampled midpoints
        displacement = jax.lax.fori_loop(
            0, sampled_indices.shape[0], process_midpoint, displacement
        )
        
        return displacement

    @staticmethod
    def orient(a, b, c):
        """Compute the orientation of three points (positive if counter-clockwise)."""
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    @staticmethod
    def repulse(vertex, mid):
        """Compute repulsion force vector."""
        diff = vertex - mid
        distance = jnp.linalg.norm(diff)
        # Avoid division by very small numbers
        direction = diff / (distance + 1e-8)
        # Inverse square law for repulsion
        magnitude = 1.0 / (distance * distance + 1e-8)
        return magnitude * direction

    def update_positions(self):
        """
        Update the positions of the vertices based on the spring forces and intersection forces.
        """
        # Compute spring forces
        spring_forces = self.compute_spring_forces(
            self.positions, self.edges, self.L_min, self.k_attr
        )
        
        # Sample midpoints for intersection detection
        self.logger.info(f"Sampling {self.sample_size} midpoints for intersection detection")
        n_edges = self.edges.shape[0]
        sampled_indices = jnp.array(np.random.choice(
            n_edges, size=min(self.sample_size, n_edges), replace=False
        ))
        
        # Compute midpoints of edges
        midpoints = (self.positions[self.edges[:, 0]] + self.positions[self.edges[:, 1]]) / 2
        
        # Find k nearest neighbors for each midpoint
        knn_idx = self.locate_knn_midpoints(
            midpoints[sampled_indices], k=self.knn_k
        )
        
        # Compute intersection forces
        intersection_forces = self.compute_intersection_forces_with_knn_index(
            self.positions, self.edges, knn_idx, sampled_indices, self.k_inter
        )
        
        # Combine forces and update positions
        total_forces = spring_forces + intersection_forces
        
        # Apply a constant factor to the forces to control the speed of movement
        self.positions = self.positions + 0.1 * total_forces
        
        # If the dimension is 3, keep it bounded to the specified plane
        if self.dimension == 3:
            self.positions = self.positions.at[:, 2].set(
                jnp.clip(self.positions[:, 2], -5.0, 5.0)
            )

    def run_layout(self, num_iterations=100):
        """
        Run the layout for a given number of iterations.
        """
        self.logger.info(f"Running layout for {num_iterations} iterations")
        for i in range(num_iterations):
            self.update_positions()
            if (i + 1) % 10 == 0:
                self.logger.info(f"Iteration {i+1}/{num_iterations}")

    def display_layout(self, edge_width=1, node_size=3, node_colors=None):
        """
        Display the graph embedding using Plotly.
        """
        if self.dimension == 2:
            self._display_layout_2d(edge_width, node_size, node_colors)
        elif self.dimension == 3:
            self._display_layout_3d(edge_width, node_size, node_colors)
        else:
            raise ValueError(f"Cannot display {self.dimension}D layout")

    def _display_layout_2d(self, edge_width=1, node_size=3, node_colors=None):
        """
        Display a 2D graph embedding using Plotly.

        Parameters
        ----------
        edges : array-like of shape (num_edges, 2)
            An array of edge pairs, where each pair (i, j) represents an edge between vertices i and j.
        pos : array-like of shape (num_vertices, 2)
            A 2D array containing the (x, y) coordinates of each vertex.

        Returns
        -------
        None
            Displays a Plotly 2D figure with vertices plotted as red markers and edges as gray lines.
        """
        # Convert positions to numpy for easier handling
        positions = np.array(self.positions)

        # Create an empty figure
        fig = go.Figure()

        # Add edges as scatter line objects
        for i, j in self.edges:
            fig.add_trace(go.Scatter(
                x=[positions[i, 0], positions[j, 0]],
                y=[positions[i, 1], positions[j, 1]],
                mode='lines',
                line=dict(width=edge_width, color='gray'),
                hoverinfo='none',
                showlegend=False
            ))

        # Add vertices as scatter points
        x, y = positions[:, 0], positions[:, 1]
        
        # Default color to red if not specified
        if node_colors is None:
            node_colors = 'red'
            
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(
                size=node_size,
                color=node_colors,
                line=dict(width=0)
            ),
            hovertext=[f"Vertex {i}" for i in range(len(x))],
            hoverinfo='text',
            showlegend=False
        ))

        # Update layout
        fig.update_layout(
            title="2D Graph Embedding",
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            plot_bgcolor='white'
        )

        # Display the figure
        fig.show()

    def _display_layout_3d(self, edge_width=1, node_size=3, node_colors=None):
        """
        Display a 3D graph embedding using Plotly.

        Parameters
        ----------
        edges : array-like of shape (num_edges, 2)
            An array of edge pairs, where each pair (i, j) represents an edge between vertices i and j.
        pos : array-like of shape (num_vertices, 3)
            A 3D array containing the (x, y, z) coordinates of each vertex.

        Returns
        -------
        None
            Displays a Plotly 3D figure with vertices plotted as red markers and edges as gray lines.
        """
        # Convert positions to numpy for easier handling
        positions = np.array(self.positions)

        # Create an empty figure
        fig = go.Figure()

        # Add edges as scatter line objects
        for i, j in self.edges:
            fig.add_trace(go.Scatter3d(
                x=[positions[i, 0], positions[j, 0]],
                y=[positions[i, 1], positions[j, 1]],
                z=[positions[i, 2], positions[j, 2]],
                mode='lines',
                line=dict(width=edge_width, color='gray'),
                hoverinfo='none',
                showlegend=False
            ))

        # Add vertices as scatter points
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        
        # Default color to red if not specified
        if node_colors is None:
            node_colors = 'red'
            
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=node_size,
                color=node_colors,
                line=dict(width=0)
            ),
            hovertext=[f"Vertex {i}" for i in range(len(x))],
            hoverinfo='text',
            showlegend=False
        ))

        # Update layout
        fig.update_layout(
            title="3D Graph Embedding",
            scene=dict(
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False
                ),
                zaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False
                )
            )
        )

        # Display the figure
        fig.show()
