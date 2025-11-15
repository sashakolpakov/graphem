Tutorials
=========

This section provides in-depth tutorials covering various aspects of GraphEm.

Graph Embedding Deep Dive
--------------------------

Understanding GraphEm's Embedding Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GraphEm combines spectral methods with force-directed layout for high-quality embeddings:

.. code-block:: python

    import graphem as ge
    import numpy as np
    import matplotlib.pyplot as plt

    # Create a graph with known structure (returns sparse adjacency matrix)
    adjacency = ge.generate_sbm(n_per_block=50, num_blocks=3, p_in=0.15, p_out=0.02, seed=42)

    # Create embedder with detailed parameters
    embedder = ge.GraphEmbedder(
        adjacency=adjacency,
        n_components=2,
        L_min=5.0,        # Shorter edges for tighter layout
        k_attr=0.8,       # Strong attraction within communities
        k_inter=0.05,     # Weak repulsion between communities
        n_neighbors=20    # More neighbors for better structure
    )

    # Monitor convergence
    positions_history = []
    for i in range(0, 100, 10):
        embedder.run_layout(num_iterations=10)
        positions_history.append(embedder.get_positions())

    # Final visualization
    embedder.display_layout()

Parameter Tuning Guide
~~~~~~~~~~~~~~~~~~~~~~

Understanding how parameters affect the embedding:

.. code-block:: python

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Test different parameter combinations
    params_to_test = [
        {'L_min': 1.0, 'k_attr': 0.2, 'k_inter': 0.1, 'title': 'Loose Layout'},
        {'L_min': 5.0, 'k_attr': 0.5, 'k_inter': 0.2, 'title': 'Balanced Layout'},
        {'L_min': 10.0, 'k_attr': 0.8, 'k_inter': 0.5, 'title': 'Tight Layout'}
    ]

    fig = make_subplots(rows=1, cols=3, 
                        subplot_titles=[p['title'] for p in params_to_test],
                        specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]])

    for i, params in enumerate(params_to_test, 1):
        embedder = ge.GraphEmbedder(
            adjacency=adjacency, n_components=2,
            L_min=params['L_min'], k_attr=params['k_attr'], k_inter=params['k_inter']
        )
        embedder.run_layout(num_iterations=50)

        pos = embedder.get_positions()
        fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode='markers',
                                name=params['title']), row=1, col=i)

    fig.show()

Working with Large Networks
---------------------------

Handling Networks with 10k+ Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large networks, optimize performance and memory usage:

.. code-block:: python

    # Generate a large scale-free network (returns sparse adjacency matrix)
    adjacency = ge.generate_ba(n=10000, m=5, seed=42)

    # Optimized embedder for large graphs
    large_embedder = ge.GraphEmbedder(
        adjacency=adjacency,
        n_components=2,       # 2D is faster than 3D
        L_min=2.0,
        k_attr=0.3,
        k_inter=0.1,
        n_neighbors=8,        # Fewer neighbors for speed
        sample_size=512,      # Automatically limited to len(edges)
        batch_size=4096,      # Automatically limited to n_vertices
        verbose=True          # Monitor progress
    )

    # Progressive refinement
    print("Initial layout...")
    large_embedder.run_layout(num_iterations=20)

    print("Refinement...")
    large_embedder.k_attr = 0.5  # Increase attraction for refinement
    large_embedder.run_layout(num_iterations=30)

    # Sample visualization (full graph would be too dense)
    pos = large_embedder.get_positions()
    n_vertices = adjacency.shape[0]
    sample_nodes = np.random.choice(n_vertices, 1000, replace=False)
    
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Scatter(
        x=pos[sample_nodes, 0], 
        y=pos[sample_nodes, 1],
        mode='markers',
        marker=dict(size=2),
        title="Sample of 1000 nodes from 10k node network"
    ))
    fig.show()

Memory-Efficient Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For extremely large networks, use chunked processing:

.. code-block:: python

    def embed_large_network_chunked(adjacency, chunk_size=5000):
        """Embed very large networks in chunks."""

        n_vertices = adjacency.shape[0]

        if n_vertices <= chunk_size:
            # Small enough to process normally
            embedder = ge.GraphEmbedder(adjacency=adjacency)
            embedder.run_layout(num_iterations=50)
            return embedder.get_positions()

        # For very large networks, use progressive approach
        print(f"Processing {n_vertices} nodes in chunks of {chunk_size}")

        # Start with a subgraph - sample nodes and extract subgraph
        import networkx as nx
        G = nx.from_scipy_sparse_array(adjacency)
        node_subset = np.random.choice(list(G.nodes()), chunk_size, replace=False)
        G_subset = G.subgraph(node_subset).copy()
        G_subset = nx.convert_node_labels_to_integers(G_subset)

        # Get adjacency matrix of subset
        subset_adjacency = nx.adjacency_matrix(G_subset, dtype=int)

        # Embed subset
        embedder = ge.GraphEmbedder(adjacency=subset_adjacency)
        embedder.run_layout(num_iterations=100)

        # This is a simplified example - full implementation would
        # gradually add nodes and refine positions
        return embedder.get_positions()

Influence Maximization Applications
-----------------------------------

Viral Marketing Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulate information spread in social networks:

.. code-block:: python

    import networkx as nx

    # Create a social network-like graph (returns sparse adjacency matrix)
    adjacency = ge.generate_ws(n=1000, k=8, p=0.1, seed=42)  # Small-world
    G = nx.from_scipy_sparse_array(adjacency)

    # Compare different seed selection strategies
    strategies = {
        'Random': np.random.choice(1000, 20, replace=False).tolist(),
        'High Degree': sorted(G.nodes(), key=G.degree, reverse=True)[:20],
        'GraphEm': None,  # Will compute below
        'Greedy': ge.greedy_seed_selection(G, k=20, p=0.05)
    }

    # Compute GraphEm strategy
    embedder = ge.GraphEmbedder(adjacency=adjacency, n_components=2)
    strategies['GraphEm'] = ge.graphem_seed_selection(embedder, k=20)
    
    # Simulate influence spread for each strategy
    results = {}
    for name, seeds in strategies.items():
        influence, _ = ge.ndlib_estimated_influence(
            G, seeds, p=0.05, iterations_count=500
        )
        results[name] = influence
        print(f"{name:12}: {influence:4d} nodes ({influence/1000:.1%})")
    
    # Visualize the best strategy
    best_strategy = max(results.keys(), key=lambda k: results[k])
    best_seeds = strategies[best_strategy]
    
    if best_strategy == 'GraphEm':
        # We already have the embedding
        pos = embedder.get_positions()
    else:
        # Create embedding for visualization
        embedder = ge.GraphEmbedder(adjacency=adjacency, n_components=2)
        embedder.run_layout(num_iterations=50)
        pos = embedder.get_positions()
    
    # Create visualization highlighting seed nodes
    import plotly.graph_objects as go
    
    # Regular nodes
    fig = go.Figure(data=go.Scatter(
        x=pos[:, 0], y=pos[:, 1],
        mode='markers',
        marker=dict(size=3, color='lightblue'),
        name='Regular nodes'
    ))
    
    # Seed nodes
    seed_pos = pos[best_seeds]
    fig.add_trace(go.Scatter(
        x=seed_pos[:, 0], y=seed_pos[:, 1],
        mode='markers',
        marker=dict(size=8, color='red'),
        name=f'Seeds ({best_strategy})'
    ))
    
    fig.update_layout(title=f"Best Strategy: {best_strategy} ({results[best_strategy]} influenced)")
    fig.show()

Network Robustness Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze how network structure affects influence spread:

.. code-block:: python

    def analyze_network_robustness(generator, params, attack_strategies):
        """Analyze robustness under different attack strategies."""

        # Generate base network (returns sparse adjacency matrix)
        adjacency = generator(**params)
        G = nx.from_scipy_sparse_array(adjacency)
        
        results = {}
        
        for strategy_name, attack_function in attack_strategies.items():
            # Remove nodes according to strategy
            nodes_to_remove = attack_function(G, int(0.1 * G.number_of_nodes()))  # Remove 10%
            G_attacked = G.copy()
            G_attacked.remove_nodes_from(nodes_to_remove)

            # Recompute largest connected component
            largest_cc = max(nx.connected_components(G_attacked), key=len)
            G_cc = G_attacked.subgraph(largest_cc).copy()
            G_cc = nx.convert_node_labels_to_integers(G_cc)

            # Test influence spread in remaining network
            if len(G_cc) > 50:  # Only if significant network remains
                # Get adjacency matrix of remaining network
                cc_adjacency = nx.adjacency_matrix(G_cc, dtype=int)

                embedder = ge.GraphEmbedder(adjacency=cc_adjacency, n_components=2)
                seeds = ge.graphem_seed_selection(embedder, k=min(10, len(G_cc)//10))

                influence, _ = ge.ndlib_estimated_influence(G_cc, seeds, p=0.1)
                results[strategy_name] = {
                    'remaining_nodes': len(G_cc),
                    'influence': influence,
                    'influence_fraction': influence / len(G_cc)
                }
            else:
                results[strategy_name] = {
                    'remaining_nodes': len(G_cc),
                    'influence': 0,
                    'influence_fraction': 0.0
                }
        
        return results

    # Define attack strategies
    attack_strategies = {
        'Random': lambda G, k: np.random.choice(list(G.nodes()), k, replace=False),
        'High Degree': lambda G, k: sorted(G.nodes(), key=G.degree, reverse=True)[:k],
        'High Betweenness': lambda G, k: sorted(G.nodes(), 
                                               key=lambda n: nx.betweenness_centrality(G)[n], 
                                               reverse=True)[:k]
    }
    
    # Test on different network types
    network_types = [
        ('Scale-Free', ge.generate_ba, {'n': 500, 'm': 3, 'seed': 42}),
        ('Small-World', ge.generate_ws, {'n': 500, 'k': 6, 'p': 0.1, 'seed': 42}),
        ('Random', ge.generate_er, {'n': 500, 'p': 0.012, 'seed': 42})
    ]
    
    for net_name, generator, params in network_types:
        print(f"\n{net_name} Network:")
        results = analyze_network_robustness(generator, params, attack_strategies)
        
        for attack, data in results.items():
            print(f"  {attack:15}: {data['remaining_nodes']:3d} nodes, "
                  f"{data['influence_fraction']:.1%} influenced")

Centrality Analysis
-------------------

Comparing Embedding-Based and Traditional Centralities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from graphem.benchmark import benchmark_correlations
    from graphem.visualization import report_full_correlation_matrix
    import pandas as pd

    # Generate different network types for comparison
    networks = [
        ('Erdős–Rényi', ge.generate_er, {'n': 300, 'p': 0.02, 'seed': 42}),
        ('Scale-Free', ge.generate_ba, {'n': 300, 'm': 2, 'seed': 42}),
        ('Small-World', ge.generate_ws, {'n': 300, 'k': 4, 'p': 0.1, 'seed': 42}),
        ('Community', ge.generate_sbm, {'n_per_block': 100, 'num_blocks': 3, 'p_in': 0.1, 'p_out': 0.01, 'seed': 42})
    ]

    correlation_results = {}

    for net_name, generator, params in networks:
        print(f"Analyzing {net_name} network...")
        
        # Run benchmark
        results = benchmark_correlations(
            graph_generator=generator,
            graph_params=params,
            n_components=2,
            num_iterations=50
        )
        
        # Compute correlation matrix
        correlation_matrix = report_full_correlation_matrix(
            results['radii'],
            results['degree'],
            results['betweenness'],
            results['eigenvector'],
            results['pagerank'],
            results['closeness'],
            results['node_load']
        )
        
        correlation_results[net_name] = correlation_matrix

    # Compare radial centrality correlations across network types
    radial_correlations = pd.DataFrame({
        net_name: corr_matrix.loc['Radius']
        for net_name, corr_matrix in correlation_results.items()
    })
    
    print("\nRadial Centrality Correlations Across Network Types:")
    print(radial_correlations.round(3))

Custom Graph Generators
-----------------------

Creating Domain-Specific Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def generate_hierarchical_network(levels=3, branching=3, intra_level_prob=0.1):
        """Generate a hierarchical network structure, returns sparse adjacency matrix."""
        import networkx as nx
        import scipy.sparse as sp

        nodes_per_level = [branching ** i for i in range(levels)]
        total_nodes = sum(nodes_per_level)

        # Create NetworkX graph for easier construction
        G = nx.Graph()
        G.add_nodes_from(range(total_nodes))
        level_starts = [0]

        # Create hierarchical connections
        for level in range(levels - 1):
            level_start = level_starts[level]
            level_size = nodes_per_level[level]

            # Connect each node in current level to nodes in next level
            for i in range(level_size):
                current_node = level_start + i
                # Each node connects to 'branching' nodes in next level
                start_next = level_starts[level] + level_size + i * branching
                for j in range(branching):
                    if start_next + j < total_nodes:
                        G.add_edge(current_node, start_next + j)

            level_starts.append(level_starts[-1] + level_size)

        # Add intra-level connections
        for level in range(levels):
            level_start = level_starts[level]
            level_size = nodes_per_level[level]

            for i in range(level_size):
                for j in range(i + 1, level_size):
                    if np.random.random() < intra_level_prob:
                        G.add_edge(level_start + i, level_start + j)

        return nx.adjacency_matrix(G, dtype=int)

    # Test the custom generator
    hier_adjacency = generate_hierarchical_network(levels=4, branching=2, intra_level_prob=0.2)

    # Embed and visualize
    embedder = ge.GraphEmbedder(
        adjacency=hier_adjacency,
        n_components=2,
        L_min=3.0,
        k_attr=0.7,
        k_inter=0.1
    )
    embedder.run_layout(num_iterations=80)
    embedder.display_layout()

Performance Optimization
------------------------

GPU Acceleration Tips
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import jax
    
    # Check available devices
    print("Available devices:", jax.devices())

    # For consistent GPU usage across runs
    def setup_gpu_embedding(adjacency, device_id=0):
        """Setup embedder with specific GPU device."""

        # Force specific device if multiple GPUs available
        if len(jax.devices('gpu')) > 1:
            device = jax.devices('gpu')[device_id]
            with jax.default_device(device):
                embedder = ge.GraphEmbedder(
                    adjacency=adjacency,
                    batch_size=8192,      # Automatically limited to n_vertices
                    sample_size=1024      # Automatically limited to len(edges)
                )
                return embedder
        else:
            return ge.GraphEmbedder(adjacency=adjacency)

    # Example with large graph
    adjacency = ge.generate_ba(n=20000, m=4, seed=42)
    embedder = setup_gpu_embedding(adjacency)
    
    # Time the embedding
    import time
    start_time = time.time()
    embedder.run_layout(num_iterations=50)
    end_time = time.time()
    
    print(f"Embedding 20k nodes took {end_time - start_time:.2f} seconds")

Profiling and Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Profile memory usage and computation time
    def profile_embedding(adjacency, iterations=50):
        """Profile embedding performance."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Memory before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Time embedding
        start_time = time.time()
        embedder = ge.GraphEmbedder(adjacency=adjacency)
        embedder.run_layout(num_iterations=iterations)
        end_time = time.time()

        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        return {
            'time': end_time - start_time,
            'memory_used': mem_after - mem_before,
            'final_memory': mem_after
        }

    # Test different graph sizes
    sizes = [500, 1000, 2000, 5000]
    for n in sizes:
        adjacency = ge.generate_ba(n=n, m=3, seed=42)
        stats = profile_embedding(adjacency, iterations=30)
        print(f"n={n:4d}: {stats['time']:5.2f}s, "
              f"{stats['memory_used']:6.1f}MB used, "
              f"{stats['final_memory']:6.1f}MB total")

Next Steps
----------

* Explore the :doc:`examples` for complete working applications
* Check the :doc:`api_reference` for detailed function documentation  
* See the :doc:`contributing` guide to help improve GraphEm
* Run the benchmarks with ``python run_benchmarks.py`` to reproduce research results