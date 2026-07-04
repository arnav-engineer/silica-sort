#!/usr/bin/env python3
"""
Benchmark script for internal (in-memory) sorting in Silica Sort.
Compares Silica Sort's hybrid learned sort and standard Rust sort against NumPy's built-in sorting.
"""

import argparse
import json
import sys
import time
from typing import Dict, List, Tuple, Any

import numpy as np

# Try importing silica_sort. If not built/installed, suggest building.
try:
    import silica_sort
except ImportError:
    print("Error: silica_sort module not found. Make sure you build/install the library first.")
    print("Example: uv run python benchmark/benchmark_internal.py")
    sys.exit(1)

# Try importing optional visualization dependencies
HAS_MATPLOTLIB = False
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    pass

HAS_TABULATE = False
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    pass


def generate_data(size: int, distribution: str, seed: int = 42) -> np.ndarray:
    """Generates an array of f64s based on the selected distribution."""
    rng = np.random.default_rng(seed)
    
    if distribution == "uniform":
        return rng.random(size, dtype=np.float64)
    
    elif distribution == "sorted":
        # Ascending values
        return np.linspace(0.0, 1.0, size, dtype=np.float64)
    
    elif distribution == "reverse":
        # Descending values
        return np.linspace(1.0, 0.0, size, dtype=np.float64)
    
    elif distribution == "mostly_sorted":
        # Sorted with 5% of elements perturbed
        arr = np.linspace(0.0, 1.0, size, dtype=np.float64)
        num_swaps = max(1, int(size * 0.05))
        idx1 = rng.integers(0, size, size=num_swaps)
        idx2 = rng.integers(0, size, size=num_swaps)
        arr[idx1], arr[idx2] = arr[idx2], arr[idx1]
        return arr
    
    elif distribution == "low_cardinality":
        # Low cardinality - choose from 10 distinct values
        choices = np.linspace(0.0, 100.0, 10, dtype=np.float64)
        return rng.choice(choices, size=size)
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def run_benchmark(
    sizes: List[int],
    distributions: List[str],
    runs: int,
    seed: int
) -> Dict[str, Any]:
    """Runs the benchmark across all sizes and distributions for all algorithms."""
    results = {}
    
    # Define algorithms to test
    # (name, function, is_inplace)
    algorithms = [
        ("silica_sort.sort_numpy", lambda arr: silica_sort.sort_numpy(arr), False),
        ("silica_sort.sort_numpy_inplace", lambda arr: silica_sort.sort_numpy_inplace(arr), True),
        ("silica_sort.sort_numpy_rust_standard", lambda arr: silica_sort.sort_numpy_rust_standard(arr), True),
        ("numpy.sort", lambda arr: np.sort(arr), False),
        ("numpy.ndarray.sort", lambda arr: arr.sort(), True)
    ]
    
    for size in sizes:
        results[size] = {}
        for dist in distributions:
            results[size][dist] = {}
            print(f"Benchmarking Size: {size:,} | Distribution: {dist}...")
            
            # Generate source data
            src_data = generate_data(size, dist, seed)
            
            for algo_name, algo_fn, is_inplace in algorithms:
                durations = []
                
                # Run multiple iterations for reliability
                for run in range(runs):
                    # Always copy the source data so each run gets the same unsorted input
                    data = src_data.copy()
                    
                    # Garbage collect if needed and force sync/memory reset if possible
                    # (though numpy operations are quick, we want clean timing)
                    start_time = time.perf_counter()
                    
                    if is_inplace:
                        algo_fn(data)
                        sorted_arr = data
                    else:
                        sorted_arr = algo_fn(data)
                        
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    durations.append(duration)
                    
                    # Verify correctness of sorting on the first run
                    if run == 0:
                        # Ensure array is sorted
                        is_sorted = np.all(sorted_arr[:-1] <= sorted_arr[1:])
                        if not is_sorted:
                            print(f"Warning: Algorithm '{algo_name}' failed to sort the array correctly!")
                
                mean_time = np.mean(durations)
                std_time = np.std(durations)
                
                results[size][dist][algo_name] = {
                    "mean_seconds": mean_time,
                    "std_seconds": std_time,
                    "throughput_mps": size / mean_time / 1e6  # Millions of elements per second
                }
                
                print(f"  - {algo_name:<38} : {mean_time*1000:8.2f} ms ({size / mean_time / 1e6:6.2f} M elements/sec)")
            print()
            
    return results


def print_table_report(results: Dict[int, Any]):
    """Prints a beautiful markdown table of results."""
    headers = ["Size", "Distribution", "Algorithm", "Mean Time (ms)", "Std Dev (ms)", "Throughput (M/sec)"]
    rows = []
    
    for size, dists in results.items():
        for dist, algos in dists.items():
            for algo, metrics in algos.items():
                rows.append([
                    f"{size:,}",
                    dist,
                    algo,
                    f"{metrics['mean_seconds']*1000:.2f}",
                    f"{metrics['std_seconds']*1000:.2f}",
                    f"{metrics['throughput_mps']:.2f}"
                ])
                
    if HAS_TABULATE:
        print("\n=== Benchmark Summary ===")
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        # Simple manual printing
        print("\n=== Benchmark Summary ===")
        fmt = "{:<12} | {:<16} | {:<38} | {:>14} | {:>12} | {:>18}"
        print(fmt.format(*headers))
        print("-" * 120)
        for r in rows:
            print(fmt.format(*r))


def save_markdown_report(results: Dict[int, Any], path: str):
    """Saves a markdown report to file."""
    headers = ["Size", "Distribution", "Algorithm", "Mean Time (ms)", "Std Dev (ms)", "Throughput (M/sec)"]
    rows = []
    
    for size, dists in results.items():
        for dist, algos in dists.items():
            for algo, metrics in algos.items():
                rows.append([
                    f"{size:,}",
                    dist,
                    algo,
                    f"{metrics['mean_seconds']*1000:.2f}",
                    f"{metrics['std_seconds']*1000:.2f}",
                    f"{metrics['throughput_mps']:.2f}"
                ])
                
    with open(path, "w") as f:
        f.write("# Silica Sort Internal Sort Benchmark Report\n\n")
        f.write("Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        
        if HAS_TABULATE:
            f.write(tabulate(rows, headers=headers, tablefmt="github"))
        else:
            # Simple markdown table manual formatting
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for r in rows:
                f.write("| " + " | ".join(r) + " |\n")
        print(f"Saved Markdown report to {path}")


def save_plot(results: Dict[int, Any], path: str):
    """Generates and saves a bar chart plot of the performance comparison."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not installed. Skipping plot generation.")
        return

    # Let's create one plot per size or group them
    sizes = list(results.keys())
    if not sizes:
        return
    
    # We will plot the largest size by default to showcase performance under scale
    target_size = sizes[-1]
    dists_data = results[target_size]
    distributions = list(dists_data.keys())
    
    # Get all algorithms
    algos = list(dists_data[distributions[0]].keys())
    
    x = np.arange(len(distributions))
    width = 0.8 / len(algos)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Setup color palette
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(algos)))
    
    for i, algo in enumerate(algos):
        throughputs = [dists_data[d][algo]["throughput_mps"] for d in distributions]
        offset = (i - len(algos)/2 + 0.5) * width
        rects = ax.bar(x + offset, throughputs, width, label=algo, color=colors[i])
        
    ax.set_ylabel("Throughput (Million elements / sec) - Higher is Better")
    ax.set_title(f"Silica Sort vs NumPy Internal Sorting Performance (Size: {target_size:,} floats)")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", " ").title() for d in distributions])
    ax.legend(loc="upper left")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved benchmark plot to {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Silica Sort Internal sorting performance.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000_000, 5_000_000, 10_000_000],
                        help="Sizes of the numpy arrays to sort.")
    parser.add_argument("--distributions", type=str, nargs="+", 
                        default=["uniform", "sorted", "reverse", "mostly_sorted", "low_cardinality"],
                        help="Data distributions to test.")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark iterations to average.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output-json", type=str, default=None, help="Save results to a JSON file.")
    parser.add_argument("--output-md", type=str, default=None, help="Save results to a Markdown file.")
    parser.add_argument("--plot-path", type=str, default="benchmark/internal_sort_benchmark.png", 
                        help="Path to save the performance comparison plot.")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting even if matplotlib is installed.")
    
    args = parser.parse_args()
    
    # Run the benchmarks
    print("==================================================")
    print("      SILICA SORT INTERNAL SORT BENCHMARK        ")
    print("==================================================")
    print(f"Systems Detected: {silica_sort.get_system_info()}\n")
    
    results = run_benchmark(args.sizes, args.distributions, args.runs, args.seed)
    
    # Print results
    print_table_report(results)
    
    # Save outputs
    if args.output_json:
        # Convert keys to str for JSON serialization
        json_results = {str(k): v for k, v in results.items()}
        with open(args.output_json, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"Saved JSON results to {args.output_json}")
        
    if args.output_md:
        save_markdown_report(results, args.output_md)
        
    if HAS_MATPLOTLIB and not args.no_plot:
        save_plot(results, args.plot_path)


if __name__ == "__main__":
    main()
