use crate::storage::ExternalRunReader;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufWriter, Error, Write};
use std::path::{Path, PathBuf};

/// A wrapper to use in the BinaryHeap for K-Way merging.
struct HeapElement {
    val: f64,
    run_idx: usize,
}

impl PartialEq for HeapElement {
    fn eq(&self, other: &Self) -> bool {
        self.val.to_bits() == other.val.to_bits()
    }
}

impl Eq for HeapElement {}

impl PartialOrd for HeapElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapElement {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for Min-Heap
        other.val.total_cmp(&self.val)
    }
}

pub fn k_way_merge(
    run_paths: &[PathBuf],
    output_path: impl AsRef<Path>,
    total_elements: usize,
) -> Result<(), crate::storage::StorageError> {
    let mut readers: Vec<ExternalRunReader> = run_paths
        .iter()
        .map(|p| ExternalRunReader::open(p))
        .collect::<Result<Vec<_>, _>>()?;

    let mut heap = BinaryHeap::new();

    // Initial fill
    for (i, reader) in readers.iter_mut().enumerate() {
        if let Some(val) = reader.next() {
            heap.push(HeapElement { val, run_idx: i });
        }
    }

    let file = File::create(output_path)?;
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
    let mut buffer = Vec::with_capacity(32 * 1024);

    let mut count = 0;
    while let Some(HeapElement { val, run_idx }) = heap.pop() {
        buffer.push(val);
        count += 1;

        if buffer.len() == buffer.capacity() {
            write_f64s(&mut writer, &buffer)?;
            buffer.clear();
        }

        if let Some(next_val) = readers[run_idx].next() {
            heap.push(HeapElement {
                val: next_val,
                run_idx,
            });
        }
    }

    if !buffer.is_empty() {
        write_f64s(&mut writer, &buffer)?;
    }
    writer.flush()?;

    if count != total_elements {
        return Err(crate::storage::StorageError::Io(Error::other(
            "merge output length does not match expected element count",
        )));
    }

    Ok(())
}

fn write_f64s(
    writer: &mut BufWriter<File>,
    data: &[f64],
) -> Result<(), crate::storage::StorageError> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    writer.write_all(bytes)?;
    Ok(())
}
