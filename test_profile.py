"""
Simple script to test profiling and SnakeViz visualization.
"""
import time
import numpy as np
import cProfile
import pstats

def factorial(n):
    """Calculate factorial recursively."""
    if n <= 1:
        return 1
    return n * factorial(n-1)

def heavy_computation():
    """Do some computation that will show up in profiling."""
    result = 0
    for i in range(10000):
        result += factorial(10)
        # Some numpy operations
        a = np.random.rand(100, 100)
        b = np.random.rand(100, 100)
        c = np.dot(a, b)
    return result

def medium_computation():
    """Medium-weight computation."""
    for i in range(5):
        np.random.rand(500, 500)
    time.sleep(0.5)
    return factorial(15)

def light_computation():
    """Light-weight computation."""
    time.sleep(0.1)
    return sum(range(1000))

def main():
    """Main function that calls other functions."""
    print("Starting profiling test...")
    heavy_computation()
    for i in range(3):
        medium_computation()
    for i in range(5):
        light_computation()
    print("Profiling test complete!")

if __name__ == "__main__":
    # Run with profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    main()
    
    profiler.disable()
    
    # Save stats to file
    profiler.dump_stats("test_profile.prof")
    
    # Print stats summary
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(20)
    
    print("\nProfile saved to test_profile.prof")
    print("To visualize: snakeviz test_profile.prof")
