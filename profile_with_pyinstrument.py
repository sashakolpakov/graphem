#!/usr/bin/env python
"""
Profile the GraphEm benchmark with PyInstrument.
This provides better visualization than cProfile for complex codebases.
"""

import os
import sys
import argparse
from pyinstrument import Profiler

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graphem.embedder import GraphEmbedder
from graphem.generators import (
    erdos_renyi_graph,
    generate_random_regular,
    generate_sbm
)
from graphem.benchmark import (
    run_benchmark, 
    benchmark_correlations,
    run_influence_benchmark
)


def profile_generator_benchmark(subsample=None):
    """Profile a generator benchmark with random regular graph."""
    print(f"Profiling generator benchmark with {'subsampling' if subsample else 'no subsampling'}...")
    
    # Use random regular graph as test case
    params = {
        'n': 500,
        'd': 3,
        'seed': 42
    }
    
    # Start profiling
    profiler = Profiler()
    profiler.start()
    
    # Run the benchmark
    result = run_benchmark(
        generate_random_regular, 
        params, 
        dim=3, 
        num_iterations=20  # Reduce iterations for profiling
    )
    
    # Stop profiling
    profiler.stop()
    
    # Output results
    print(f"Benchmark completed in {result['total_time']:.2f} seconds")
    
    # Save HTML report
    html_output = f"profile_generator{'_' + str(subsample) if subsample else ''}.html"
    with open(html_output, "w") as f:
        f.write(profiler.output_html())
    
    print(f"Profiling results saved to {html_output}")
    
    # Also print to console
    print("\nConsole profiling summary:")
    print(profiler.output_text(unicode=True, color=True))


def profile_influence_benchmark(subsample=None):
    """Profile an influence maximization benchmark."""
    print(f"Profiling influence benchmark with {'subsampling' if subsample else 'no subsampling'}...")
    
    # Use SBM graph as test case (good community structure)
    params = {
        'n_per_block': 50,  # Smaller for profiling
        'num_blocks': 4,
        'p_in': 0.15,
        'p_out': 0.01,
        'seed': 42
    }
    
    # Start profiling
    profiler = Profiler()
    profiler.start()
    
    # Run the benchmark
    result = run_influence_benchmark(
        generate_sbm, 
        params, 
        k=5,  # Number of seed nodes
        p=0.1,  # Propagation probability
        iterations=20,  # Reduced for profiling
        dim=3, 
        num_layout_iterations=10  # Reduced for profiling
    )
    
    # Stop profiling
    profiler.stop()
    
    # Output results
    print(f"GraphEm influence: {result['graphem_influence']:.2f}")
    print(f"Greedy influence: {result['greedy_influence']:.2f}")
    
    # Save HTML report
    html_output = f"profile_influence{'_' + str(subsample) if subsample else ''}.html"
    with open(html_output, "w") as f:
        f.write(profiler.output_html())
    
    print(f"Profiling results saved to {html_output}")
    
    # Also print to console
    print("\nConsole profiling summary:")
    print(profiler.output_text(unicode=True, color=True))


def main():
    """Parse arguments and run profiler."""
    parser = argparse.ArgumentParser(description="Profile GraphEm with PyInstrument")
    
    parser.add_argument(
        "--mode", "-m",
        choices=["generator", "influence", "both"],
        default="both",
        help="What to profile: generator, influence, or both"
    )
    
    parser.add_argument(
        "--subsample", "-s",
        type=int,
        default=None,
        help="Subsample vertices (default: no subsampling)"
    )
    
    args = parser.parse_args()
    
    if args.mode in ["generator", "both"]:
        profile_generator_benchmark(args.subsample)
    
    if args.mode in ["influence", "both"]:
        profile_influence_benchmark(args.subsample)


if __name__ == "__main__":
    main()
