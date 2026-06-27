use memmap2::MmapOptions;
use std::fs::File;
use std::io::{BufWriter, Error, ErrorKind, Read, Write};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub struct MemoryMappedFile {
    mmap: memmap2::Mmap,
    len: usize, // Number of elements (f64)
}

impl MemoryMappedFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        if mmap.len() % std::mem::size_of::<f64>() != 0 {
            return Err(StorageError::Io(Error::new(
                ErrorKind::InvalidData,
                "input file length is not a multiple of 8 bytes",
            )));
        }
        let len = mmap.len() / std::mem::size_of::<f64>();
        Ok(Self { mmap, len })
    }

    pub fn as_slice(&self) -> &[f64] {
        unsafe {
            let ptr = self.mmap.as_ptr() as *const f64;
            std::slice::from_raw_parts(ptr, self.len)
        }
    }
}

/// A reader for a sorted "run" on disk.
pub struct ExternalRunReader {
    _mmap: MemoryMappedFile,
    ptr: *const f64,
    len: usize,
    current: usize,
}

impl ExternalRunReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let mmap = MemoryMappedFile::open(path)?;
        let ptr = mmap.as_slice().as_ptr();
        let len = mmap.as_slice().len();
        Ok(Self {
            _mmap: mmap,
            ptr,
            len,
            current: 0,
        })
    }
}

impl Iterator for ExternalRunReader {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.len {
            let val = unsafe { *self.ptr.add(self.current) };
            self.current += 1;
            Some(val)
        } else {
            None
        }
    }
}

pub fn read_f64_chunk(file: &mut File, max_elements: usize) -> Result<Vec<f64>, StorageError> {
    if max_elements == 0 {
        return Ok(Vec::new());
    }

    let mut data: Vec<f64> = Vec::with_capacity(max_elements);

    let buffer = unsafe {
        std::slice::from_raw_parts_mut(
            data.as_mut_ptr() as *mut u8,
            max_elements * std::mem::size_of::<f64>(),
        )
    };

    let mut bytes_read = 0;
    while bytes_read < buffer.len() {
        let read_now = file.read(&mut buffer[bytes_read..])?;
        if read_now == 0 {
            break;
        }
        bytes_read += read_now;
    }

    if bytes_read == 0 {
        return Ok(data);
    }

    if bytes_read % std::mem::size_of::<f64>() != 0 {
        return Err(StorageError::Io(Error::new(
            ErrorKind::UnexpectedEof,
            "read ended on a partial f64",
        )));
    }

    unsafe { data.set_len(bytes_read / std::mem::size_of::<f64>()) };
    Ok(data)
}

pub fn write_f64_file(path: impl AsRef<Path>, data: &[f64]) -> Result<(), StorageError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    writer.write_all(bytes)?;
    writer.flush()?;
    Ok(())
}
