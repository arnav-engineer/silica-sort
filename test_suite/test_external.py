import os
import tempfile
import unittest
import numpy as np
import silica_sort

class TestExternalSort(unittest.TestCase):
    
    def setUp(self):
        self.temp_files = []

    def tearDown(self):
        # Cleanup any remaining temporary files
        for path in self.temp_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def create_temp_file(self, prefix="silica_test_", suffix=".bin"):
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
        os.close(fd)
        self.temp_files.append(path)
        return path

    def test_sort_file_valid(self):
        """Test sorting a valid binary file of float64 values."""
        input_path = self.create_temp_file()
        output_path = self.create_temp_file()
        
        # Write random float64s
        np.random.seed(42)
        expected_data = np.random.rand(20000)
        expected_data.tofile(input_path)
        
        # Execute external sort
        silica_sort.sort_file(input_path, output_path)
        
        # Read and verify sorted file
        self.assertTrue(os.path.exists(output_path))
        sorted_data = np.fromfile(output_path, dtype=np.float64)
        
        self.assertEqual(len(sorted_data), len(expected_data))
        self.assertTrue(np.all(sorted_data[:-1] <= sorted_data[1:]))
        
        # Match element check
        expected_sorted = np.sort(expected_data)
        self.assertTrue(np.allclose(sorted_data, expected_sorted, rtol=1e-12, atol=1e-12))

    def test_sort_file_empty(self):
        """Test sorting an empty file."""
        input_path = self.create_temp_file()
        output_path = self.create_temp_file()
        
        # Empty input
        open(input_path, "wb").close()
        
        # Execute external sort
        silica_sort.sort_file(input_path, output_path)
        
        # Verify output is empty
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(os.path.getsize(output_path), 0)

    def test_sort_file_invalid_length(self):
        """Test sorting a file whose size is not a multiple of 8 bytes."""
        input_path = self.create_temp_file()
        output_path = self.create_temp_file()
        
        # Write 15 bytes (invalid alignment for float64)
        with open(input_path, "wb") as f:
            f.write(b"0" * 15)
            
        # Verify it raises ValueError
        with self.assertRaises(ValueError):
            silica_sort.sort_file(input_path, output_path)

    def test_sort_file_nonexistent(self):
        """Test sorting a nonexistent input file."""
        nonexistent_input = "nonexistent_input_file_path.bin"
        output_path = self.create_temp_file()
        
        # Verify it raises IOError
        with self.assertRaises(IOError):
            silica_sort.sort_file(nonexistent_input, output_path)


if __name__ == '__main__':
    unittest.main()
