import unittest
import numpy as np
import silica_sort

class TestInternalSort(unittest.TestCase):
    
    def test_sort_numpy_random(self):
        """Test out-of-place Learned Sort on random data."""
        np.random.seed(42)
        arr = np.random.rand(10000)
        sorted_arr = silica_sort.sort_numpy(arr)
        
        # Verify result is sorted
        self.assertTrue(np.all(sorted_arr[:-1] <= sorted_arr[1:]))
        # Verify original array was not mutated
        self.assertFalse(np.array_equal(arr, sorted_arr))
        # Verify shape and type
        self.assertEqual(sorted_arr.shape, arr.shape)
        self.assertEqual(sorted_arr.dtype, np.float64)

    def test_sort_numpy_inplace_random(self):
        """Test in-place Learned Sort on random data."""
        np.random.seed(42)
        arr = np.random.rand(10000)
        original_copy = arr.copy()
        
        silica_sort.sort_numpy_inplace(arr)
        
        # Verify array is sorted
        self.assertTrue(np.all(arr[:-1] <= arr[1:]))
        # Verify elements are the same (permutation match)
        self.assertAlmostEqual(np.sum(arr), np.sum(original_copy), places=10)
        self.assertEqual(arr.shape, original_copy.shape)

    def test_sort_numpy_rust_standard_random(self):
        """Test in-place standard Rust sort on random data."""
        np.random.seed(42)
        arr = np.random.rand(10000)
        original_copy = arr.copy()
        
        silica_sort.sort_numpy_rust_standard(arr)
        
        # Verify array is sorted
        self.assertTrue(np.all(arr[:-1] <= arr[1:]))
        # Verify elements are the same
        self.assertAlmostEqual(np.sum(arr), np.sum(original_copy), places=10)

    def test_empty_array(self):
        """Verify sorting an empty array."""
        arr = np.array([], dtype=np.float64)
        
        # Out-of-place
        res = silica_sort.sort_numpy(arr)
        self.assertEqual(len(res), 0)
        
        # In-place
        silica_sort.sort_numpy_inplace(arr)
        self.assertEqual(len(arr), 0)

    def test_single_element(self):
        """Verify sorting a single-element array."""
        arr = np.array([42.0], dtype=np.float64)
        
        # Out-of-place
        res = silica_sort.sort_numpy(arr)
        self.assertEqual(res[0], 42.0)
        
        # In-place
        silica_sort.sort_numpy_inplace(arr)
        self.assertEqual(arr[0], 42.0)

    def test_already_sorted(self):
        """Verify sorting of an already sorted array."""
        arr = np.linspace(0.0, 100.0, 1000, dtype=np.float64)
        original = arr.copy()
        
        # In-place
        silica_sort.sort_numpy_inplace(arr)
        self.assertTrue(np.all(arr[:-1] <= arr[1:]))
        self.assertTrue(np.array_equal(arr, original))

    def test_reverse_sorted(self):
        """Verify sorting of a reverse-sorted array."""
        arr = np.linspace(100.0, 0.0, 1000, dtype=np.float64)
        
        # In-place
        silica_sort.sort_numpy_inplace(arr)
        self.assertTrue(np.all(arr[:-1] <= arr[1:]))

    def test_non_contiguous_array(self):
        """Verify that passing non-contiguous NumPy arrays raises ValueError."""
        arr = np.random.rand(100)
        # Create non-contiguous slice (step = 2)
        non_contiguous_arr = arr[::2]
        
        # Verify it raises ValueError for out-of-place
        with self.assertRaises(ValueError):
            silica_sort.sort_numpy(non_contiguous_arr)
            
        # Verify it raises ValueError for in-place
        with self.assertRaises(ValueError):
            silica_sort.sort_numpy_inplace(non_contiguous_arr)
            
        # Verify it raises ValueError for standard Rust sort
        with self.assertRaises(ValueError):
            silica_sort.sort_numpy_rust_standard(non_contiguous_arr)

    def test_data_integrity(self):
        """Verify data integrity: ensures all original values exist in the sorted array."""
        np.random.seed(123)
        arr = np.random.rand(100)
        sorted_arr = silica_sort.sort_numpy(arr)
        
        # Sort both using numpy to check exact match
        np_sorted = np.sort(arr)
        self.assertTrue(np.allclose(sorted_arr, np_sorted, rtol=1e-12, atol=1e-12))


if __name__ == '__main__':
    unittest.main()
