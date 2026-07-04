#!/usr/bin/env python3
"""
Benchmark script for external (disk-based) sorting in Silica Sort.
Compares Silica Sort's high-performance external sort against:
1. In-memory NumPy sort (as an upper bound, memory-unconstrained baseline).
2. A custom Python external sort implemented using heapq.merge.
"""

import argparse
import heapq
import json
import os
import sys
import tempfile
import time
from typing import Dict, List, Any, Tuple

import numpy as np

# Try importing silica_sort
try:
    import silica_sort
except ImportError:
    print("Error: silica_sort module not found. Make sure you build/install the library first.")
    print("Example: uv run python benchmark/benchmark_external.py")
    sys.exit(1)

HAS_TABULATE = False
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    pass


def generate_random_float_file(file_path: str, size_mb: float, seed: int = 42) -> int:
    """Generates a binary file of random float64s. Returns the number of elements."""
    element_size = 8  # float64 is 8 bytes
    total_bytes = int(size_mb * 1024 * 1024)
    # Align to 8 bytes
    total_bytes = (total_bytes // element_size) * element_size
    num_elements = total_bytes // element_size

    print(f"Generating {size_mb} MB input file ({num_elements:,} elements) at '{file_path}'...")
    
    rng = np.random.default_rng(seed)
    chunk_size = 10_000_000  # Write in chunks of 80MB to avoid high memory spikes during generation
    
    written = 0
    with open(file_path, "wb") as f:
        while written < num_elements:
            to_write = min(chunk_size, num_elements - written)
            data = rng.random(to_write, dtype=np.float64)
            f.write(data.tobytes())
            written += to_write
            
    print("Generation complete.")
    return num_elements


def float_generator(file_path: str, chunk_size_elements: int = 16384):
    """Generator to yield float64 values from a binary file in chunks to minimize disk I/O."""
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size_elements * 8)
            if not chunk:
                break
            arr = np.frombuffer(chunk, dtype=np.float64)
            for val in arr:
                yield val


def python_external_sort(input_path: str, output_path: str, chunk_bytes: int = 64 * 1024 * 1024):
    """
    Pure Python external sort baseline.
    Reads chunk_bytes of floats, sorts in memory, writes to temporary run files,
    then merges them using heapq.merge.
    """
    element_size = 8
    chunk_elements = chunk_bytes // element_size
    
    temp_files = []
    
    # Phase 1: Split and Sort Chunks
    with open(input_path, "rb") as f:
        while True:
            chunk_data = f.read(chunk_bytes)
            if not chunk_data:
                break
            
            arr = np.frombuffer(chunk_data, dtype=np.float64).copy()
            arr.sort()  # In-place numpy sort
            
            # Write to temp run file
            temp_f = tempfile.NamedTemporaryFile(delete=False, suffix=".run")
            temp_f.write(arr.tobytes())
            temp_f.close()
            temp_files.append(temp_f.name)
            
    if not temp_files:
        # Empty input
        open(output_path, "wb").close()
        return

    try:
        # Phase 2: K-Way Merge
        # Setup generators for each sorted run file
        generators = [float_generator(p) for p in temp_files]
        merged = heapq.merge(*generators)
        
        # Write merged results back in chunks
        write_buffer = []
        write_buffer_limit = 65536
        
        with open(output_path, "wb") as out_f:
            for val in merged:
                write_buffer.append(val)
                if len(write_buffer) == write_buffer_limit:
                    out_f.write(np.array(write_buffer, dtype=np.float64).tobytes())
                    write_buffer.clear()
            if write_buffer:
                out_f.write(np.array(write_buffer, dtype=np.float64).tobytes())
    finally:
        # Cleanup temp run files
        for p in temp_files:
            try:
                os.remove(p)
            except OSError:
                pass


def verify_sorted_file(path: str, expected_elements: int) -> Tuple[bool, str]:
    """Verifies that the file exists, has correct size, and elements are sorted in ascending order."""
    if not os.path.exists(path):
        return False, "File does not exist"
        
    size_bytes = os.path.getsize(path)
    if size_bytes != expected_elements * 8:
        return False, f"Expected size {expected_elements * 8} bytes, got {size_bytes} bytes"
        
    # Read chunk-by-chunk to verify order and avoid loading entire file
    chunk_size = 5_000_000
    last_val = -float('inf')
    total_read = 0
    
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size * 8)
            if not chunk:
                break
            arr = np.frombuffer(chunk, dtype=np.float64)
            total_read += len(arr)
            
            if len(arr) == 0:
                continue
                
            # Verify first element against last element of previous chunk
            if arr[0] < last_val:
                return False, f"Sort verification failed at boundary: {arr[0]} < {last_val}"
                
            # Verify order within chunk
            if not np.all(arr[:-1] <= arr[1:]):
                return False, "Sort verification failed within chunk"
                
            last_val = arr[-1]
            
    if total_read != expected_elements:
        return False, f"Read {total_read} elements, expected {expected_elements}"
        
    return True, "Verification passed"


def run_benchmark(
    size_mb: float,
    runs: int,
    seed: int,
    skip_numpy_in_memory: bool = False
) -> Dict[str, Any]:
    """Runs the external sort benchmarks for the chosen file size."""
    results = {}
    
    # Create temp files
    fd_in, input_file = tempfile.mkstemp(suffix=".bin")
    os.close(fd_in)
    
    fd_out, output_file = tempfile.mkstemp(suffix=".bin")
    os.close(fd_out)
    
    try:
        # Generate random input file
        num_elements = generate_random_float_file(input_file, size_mb, seed)
        actual_size_bytes = os.path.getsize(input_file)
        actual_size_mb = actual_size_bytes / (1024 * 1024)
        
        # 1. Benchmark: silica_sort.sort_file
        print("\n--- Benchmarking silica_sort.sort_file ---")
        silica_times = []
        for run in range(runs):
            # Clean output
            if os.path.exists(output_file):
                os.remove(output_file)
                
            start = time.perf_counter()
            silica_sort.sort_file(input_file, output_file)
            duration = time.perf_counter() - start
            
            # Verify
            ok, msg = verify_sorted_file(output_file, num_elements)
            if not ok:
                print(f"  Warning: silica_sort verification failed! {msg}")
            
            silica_times.append(duration)
            print(f"  Run {run + 1}: {duration:.3f} s ({actual_size_mb / duration:.2f} MB/s)")
            
        results["silica_sort.sort_file"] = {
            "mean_seconds": np.mean(silica_times),
            "std_seconds": np.std(silica_times),
            "throughput_mbps": actual_size_mb / np.mean(silica_times)
        }

        # 2. Benchmark: Python Heapq External Sort
        print("\n--- Benchmarking Python heapq external sort ---")
        py_external_times = []
        for run in range(runs):
            # Clean output
            if os.path.exists(output_file):
                os.remove(output_file)
                
            start = time.perf_counter()
            # Use 64MB chunks for partitioning
            python_external_sort(input_file, output_file, chunk_bytes=64 * 1024 * 1024)
            duration = time.perf_counter() - start
            
            # Verify
            ok, msg = verify_sorted_file(output_file, num_elements)
            if not ok:
                print(f"  Warning: python external sort verification failed! {msg}")
                
            py_external_times.append(duration)
            print(f"  Run {run + 1}: {duration:.3f} s ({actual_size_mb / duration:.2f} MB/s)")
            
        results["python_heapq_external"] = {
            "mean_seconds": np.mean(py_external_times),
            "std_seconds": np.std(py_external_times),
            "throughput_mbps": actual_size_mb / np.mean(py_external_times)
        }

        # 3. Benchmark: In-Memory NumPy Sort (if permitted)
        if not skip_numpy_in_memory:
            print("\n--- Benchmarking In-Memory NumPy Sort (Baseline) ---")
            numpy_in_memory_times = []
            for run in range(runs):
                # Clean output
                if os.path.exists(output_file):
                    os.remove(output_file)
                    
                start = time.perf_counter()
                
                # In-memory pipeline
                arr = np.fromfile(input_file, dtype=np.float64)
                arr.sort()
                arr.tofile(output_file)
                
                duration = time.perf_counter() - start
                
                # Verify
                ok, msg = verify_sorted_file(output_file, num_elements)
                if not ok:
                    print(f"  Warning: numpy in-memory verification failed! {msg}")
                    
                numpy_in_memory_times.append(duration)
                print(f"  Run {run + 1}: {duration:.3f} s ({actual_size_mb / duration:.2f} MB/s)")
                
            results["numpy_in_memory"] = {
                "mean_seconds": np.mean(numpy_in_memory_times),
                "std_seconds": np.std(numpy_in_memory_times),
                "throughput_mbps": actual_size_mb / np.mean(numpy_in_memory_times)
            }
            
    finally:
        # Cleanup input and output benchmark files
        for p in [input_file, output_file]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
                    
    return results, actual_size_mb


def print_table_report(results: Dict[str, Any], size_mb: float):
    """Prints a beautiful markdown table of results."""
    headers = ["Algorithm", "File Size (MB)", "Mean Time (s)", "Std Dev (s)", "Throughput (MB/s)", "Speedup vs Python Heapq"]
    rows = []
    
    py_mean = results["python_heapq_external"]["mean_seconds"]
    
    for algo, metrics in results.items():
        speedup = py_mean / metrics["mean_seconds"]
        rows.append([
            algo,
            f"{size_mb:.1f}",
            f"{metrics['mean_seconds']:.3f}",
            f"{metrics['std_seconds']:.3f}",
            f"{metrics['throughput_mbps']:.2f}",
            f"{speedup:.2f}x"
        ])
        
    if HAS_TABULATE:
        print("\n=== External Sort Benchmark Summary ===")
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        print("\n=== External Sort Benchmark Summary ===")
        fmt = "{:<25} | {:<14} | {:>13} | {:>11} | {:>17} | {:>23}"
        print(fmt.format(*headers))
        print("-" * 115)
        for r in rows:
            print(fmt.format(*r))


def save_markdown_report(results: Dict[str, Any], size_mb: float, path: str):
    """Saves a markdown report to file."""
    headers = ["Algorithm", "File Size (MB)", "Mean Time (s)", "Std Dev (s)", "Throughput (MB/s)", "Speedup vs Python Heapq"]
    rows = []
    
    py_mean = results["python_heapq_external"]["mean_seconds"]
    
    for algo, metrics in results.items():
        speedup = py_mean / metrics["mean_seconds"]
        rows.append([
            algo,
            f"{size_mb:.1f}",
            f"{metrics['mean_seconds']:.3f}",
            f"{metrics['std_seconds']:.3f}",
            f"{metrics['throughput_mbps']:.2f}",
            f"{speedup:.2f}x"
        ])
        
    with open(path, "w") as f:
        f.write("# Silica Sort External Sort Benchmark Report\n\n")
        f.write("Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        f.write(f"**Benchmark File Size**: {size_mb:.1f} MB\n\n")
        
        if HAS_TABULATE:
            f.write(tabulate(rows, headers=headers, tablefmt="github"))
        else:
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for r in rows:
                f.write("| " + " | ".join(r) + " |\n")
        print(f"Saved Markdown report to {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Silica Sort External sorting performance.")
    parser.add_argument("--size-mb", type=float, default=100.0,
                        help="Size of the binary file to sort (in Megabytes).")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark iterations to average.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--skip-numpy-in-memory", action="store_true", 
                        help="Skip NumPy in-memory sort baseline (useful for sizes larger than RAM).")
    parser.add_argument("--output-json", type=str, default=None, help="Save results to a JSON file.")
    parser.add_argument("--output-md", type=str, default=None, help="Save results to a Markdown file.")
    
    args = parser.parse_args()
    
    print("==================================================")
    print("      SILICA SORT EXTERNAL SORT BENCHMARK        ")
    print("==================================================")
    print(f"Systems Detected: {silica_sort.get_system_info()}\n")
    
    results, actual_size_mb = run_benchmark(
        args.size_mb, 
        args.runs, 
        args.seed, 
        args.skip_numpy_in_memory
    )
    
    print_table_report(results, actual_size_mb)
    
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON results to {args.output_json}")
        
    if args.output_md:
        save_markdown_report(results, actual_size_mb, args.output_md)


if __name__ == "__main__":
    main()
