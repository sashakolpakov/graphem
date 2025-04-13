#!/usr/bin/env python
"""
Simple script to check if all the generator and visualization function imports work correctly.
This will verify that our alias fixes in __init__.py resolved the import issues.
"""

print("Checking if all imports work correctly...")

try:
    # Try importing using all possible function names
    from graphem import (
        # Original names
        erdos_renyi_graph,
        generate_ba,
        generate_ws,
        generate_random_regular,
        generate_sbm,
        generate_scale_free,
        generate_geometric,
        generate_caveman,
        generate_relaxed_caveman,
        
        # Aliased names (previously causing pylint errors)
        generate_erdos_renyi_graph,
        generate_barabasi_albert_graph,
        generate_watts_strogatz_graph,
        generate_random_regular_graph,
        generate_sbm_graph,
        generate_scale_free_graph,
        generate_geometric_graph,
        generate_caveman_graph,
        generate_relaxed_caveman_graph,
        
        # Visualization functions
        report_corr,
        report_full_correlation_matrix,
        plot_radial_vs_centrality,
        display_benchmark_results,
        visualize_graph
    )
    print("✅ SUCCESS: All imports work correctly!")
    print("All function aliases are properly defined and importable.")

except ImportError as e:
    print(f"❌ ERROR: Not all imports work correctly: {e}")
    print("Some function aliases may still be missing or incorrectly defined.")
