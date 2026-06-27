use numpy::{PyArray1, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::prelude::*;

use crate::hardware::SystemInfo;
use crate::rmi::MonotonicRMI;
use pyo3::types::PyDict;
use std::fs::File;
use std::path::PathBuf;

use crate::learned_sort;
use crate::storage::{read_f64_chunk, write_f64_file};



/// Internal shared sort logic.
pub fn sort_slice(data: &[f64]) -> Vec<f64> {
    learned_sort::learned_sort_f64_vec(data)
}

/// Sort a NumPy array and return a new sorted 1D array.
#[pyfunction]
pub fn sort_numpy<'py>(
    py: Python<'py>,
    array: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<&'py PyArray1<f64>> {
    let array = array.as_array();
    let raw_slice = array.as_slice().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Non-contiguous array not supported")
    })?;
    
    // Release the GIL during heavy sorting
    let sorted_vec = py.allow_threads(|| sort_slice(raw_slice));
    Ok(PyArray1::from_vec(py, sorted_vec))
}

/// Sort a NumPy array in place.
#[pyfunction]
pub fn sort_numpy_inplace<'py>(
    py: Python<'py>,
    mut array: PyReadwriteArrayDyn<'py, f64>,
) -> PyResult<()> {
    let mut array = array.as_array_mut();
    let raw_slice = array.as_slice_mut().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Non-contiguous array not supported")
    })?;
    
    // Cast to usize to satisfy Send/Sync requirements for allow_threads.
    // Safety: we have exclusive mutable access via PyReadwriteArrayDyn.
    let addr = raw_slice.as_mut_ptr() as usize;
    let len = raw_slice.len();
    
    py.allow_threads(move || {
        let slice = unsafe { std::slice::from_raw_parts_mut(addr as *mut f64, len) };
        learned_sort::learned_sort_f64(slice);
    });
    
    Ok(())
}

/// Sort a NumPy array in place using Rust's standard sort_unstable_by.
#[pyfunction]
pub fn sort_numpy_rust_standard<'py>(
    py: Python<'py>,
    mut array: PyReadwriteArrayDyn<'py, f64>,
) -> PyResult<()> {
    let mut array = array.as_array_mut();
    let raw_slice = array.as_slice_mut().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Non-contiguous array not supported")
    })?;
    
    let addr = raw_slice.as_mut_ptr() as usize;
    let len = raw_slice.len();
    
    py.allow_threads(move || {
        let slice = unsafe { std::slice::from_raw_parts_mut(addr as *mut f64, len) };
        slice.sort_by(|a, b| a.total_cmp(b));
    });
    
    Ok(())
}

/// Sort a file of f64s externally.
#[pyfunction]
pub fn sort_file(_py: Python<'_>, input_path: String, output_path: String) -> PyResult<()> {
    let mut input_file = File::open(&input_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    let input_len = input_file
        .metadata()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?
        .len() as usize;

    if input_len % std::mem::size_of::<f64>() != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input file length is not a multiple of 8 bytes",
        ));
    }

    let n = input_len / std::mem::size_of::<f64>();
    let direct_sort_limit = recommended_single_run_limit(n);

    if n <= direct_sort_limit {
        let mut chunk = read_f64_chunk(&mut input_file, n)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        learned_sort::learned_sort_f64(&mut chunk);
        write_f64_file(&output_path, &chunk)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        return Ok(());
    }

    let chunk_size = recommended_external_chunk_size(n);

    // Release the GIL during the long-running external sort
    let merge_result = _py.allow_threads(move || -> Result<(), String> {
        let (read_tx, read_rx) = std::sync::mpsc::sync_channel::<Vec<f64>>(2);
        let (write_tx, write_rx) = std::sync::mpsc::sync_channel::<Vec<f64>>(2);

        // 1. Reader Thread
        let reader_thread = std::thread::spawn(move || -> Result<(), String> {
            loop {
                let chunk = read_f64_chunk(&mut input_file, chunk_size)
                    .map_err(|e| e.to_string())?;
                if chunk.is_empty() {
                    break;
                }
                if read_tx.send(chunk).is_err() {
                    break; // Receiver disconnected
                }
            }
            Ok(())
        });

        // 2. Sorter Thread
        let sorter_thread = std::thread::spawn(move || {
            while let Ok(mut chunk) = read_rx.recv() {
                learned_sort::learned_sort_f64(&mut chunk);
                if write_tx.send(chunk).is_err() {
                    break; // Receiver disconnected
                }
            }
        });

        // 3. Writer Thread (Main thread handles writing to avoid spawning unnecessary threads)
        struct RunCleanup(Vec<PathBuf>);
        impl Drop for RunCleanup {
            fn drop(&mut self) {
                for path in &self.0 {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
        let mut run_paths = RunCleanup(Vec::new());
        let mut write_err = None;
        while let Ok(sorted_chunk) = write_rx.recv() {
            let run_path = format!("{}.run.{}", input_path, run_paths.0.len());
            if let Err(e) = write_f64_file(&run_path, &sorted_chunk) {
                write_err = Some(e.to_string());
                break;
            }
            run_paths.0.push(PathBuf::from(run_path));
        }

        // Synchronize and check for errors in the pipeline
        let reader_res = reader_thread.join().unwrap_or_else(|_| Err("Reader thread panicked".into()));
        sorter_thread.join().unwrap_or_else(|_| ());

        if let Some(e) = write_err {
            return Err(e);
        }
        reader_res?;

        // 4. K-Way Merge
        let merge_res = crate::external_sort::k_way_merge(&run_paths.0, &output_path, n)
            .map_err(|e| e.to_string());

        merge_res
    });

    merge_result.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))
}

/// Get system hardware information properly detected by Rust
#[pyfunction]
pub fn get_system_info(py: Python<'_>) -> PyResult<&PyDict> {
    let info = SystemInfo::detect();
    let dict = PyDict::new(py);
    dict.set_item("l1_cache_size", info.l1_cache_size)?;
    dict.set_item("l2_cache_size", info.l2_cache_size)?;
    dict.set_item("simd_level", format!("{:?}", info.simd_level))?;
    Ok(dict)
}

/// Test RMI training on a numpy array.
#[pyfunction]
pub fn test_rmi(
    _py: Python<'_>,
    data: PyReadonlyArrayDyn<'_, f64>,
    num_buckets: usize,
) -> PyResult<PyObject> {
    let array = data.as_array();
    let mut vec = array.iter().cloned().collect::<Vec<_>>();
    vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rmi = MonotonicRMI::train(&vec, num_buckets);
    let predictions: Vec<usize> = vec.iter().map(|&x| rmi.predict(x)).collect();
    let py_array = PyArray1::from_vec(_py, predictions);
    Ok(py_array.to_object(_py))
}

fn recommended_external_chunk_size(total_elements: usize) -> usize {
    const MIN_CHUNK_BYTES: usize = 64 * 1024 * 1024;
    const MAX_CHUNK_BYTES: usize = 512 * 1024 * 1024;
    const FALLBACK_CHUNK_BYTES: usize = 256 * 1024 * 1024;

    if total_elements == 0 {
        return 1;
    }

    let chunk_bytes = detect_available_memory_bytes()
        .map(|available| {
            (available / 6).clamp(MIN_CHUNK_BYTES as u64, MAX_CHUNK_BYTES as u64) as usize
        })
        .unwrap_or(FALLBACK_CHUNK_BYTES);

    (chunk_bytes / std::mem::size_of::<f64>()).clamp(1, total_elements)
}

fn recommended_single_run_limit(total_elements: usize) -> usize {
    const MIN_DIRECT_BYTES: usize = 96 * 1024 * 1024;
    const MAX_DIRECT_BYTES: usize = 768 * 1024 * 1024;
    const FALLBACK_DIRECT_BYTES: usize = 192 * 1024 * 1024;

    if total_elements == 0 {
        return 1;
    }

    let direct_bytes = detect_available_memory_bytes()
        .map(|available| {
            (available / 5).clamp(MIN_DIRECT_BYTES as u64, MAX_DIRECT_BYTES as u64) as usize
        })
        .unwrap_or(FALLBACK_DIRECT_BYTES);

    (direct_bytes / std::mem::size_of::<f64>()).clamp(1, total_elements)
}

fn detect_available_memory_bytes() -> Option<u64> {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    Some(sys.available_memory())
}
