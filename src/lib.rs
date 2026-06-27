use pyo3::prelude::*;

mod external_sort;
mod hardware;
mod interface;
pub mod learned_sort;
mod radix_sort;
mod rmi;
mod sampler;
mod simd_sort;
mod storage;

#[pymodule]
fn _silica_sort(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(interface::sort_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(interface::sort_numpy_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(interface::sort_numpy_rust_standard, m)?)?;
    m.add_function(wrap_pyfunction!(interface::sort_file, m)?)?;
    m.add_function(wrap_pyfunction!(interface::get_system_info, m)?)?;
    m.add_function(wrap_pyfunction!(interface::test_rmi, m)?)?;
    Ok(())
}
