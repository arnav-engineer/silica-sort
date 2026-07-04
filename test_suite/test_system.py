import unittest
import numpy as np
import silica_sort

class TestSystemAndHelpers(unittest.TestCase):
    
    def test_get_system_info(self):
        """Test detection and structure of detected hardware information."""
        info = silica_sort.get_system_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn("l1_cache_size", info)
        self.assertIn("l2_cache_size", info)
        self.assertIn("simd_level", info)
        
        # Verify types
        self.assertIsInstance(info["l1_cache_size"], int)
        self.assertIsInstance(info["l2_cache_size"], int)
        self.assertIsInstance(info["simd_level"], str)
        
        # Verify cache values are reasonable
        self.assertGreaterEqual(info["l1_cache_size"], 0)
        self.assertGreaterEqual(info["l2_cache_size"], 0)

    def test_test_rmi(self):
        """Test training of Monotonic RMI predictions."""
        np.random.seed(42)
        arr = np.random.rand(1000)
        arr.sort()
        
        num_buckets = 128
        predictions = silica_sort.test_rmi(arr, num_buckets)
        
        # Verify shape
        self.assertEqual(predictions.shape, arr.shape)
        # Verify elements are integers within [0, num_buckets-1]
        self.assertTrue(np.issubdtype(predictions.dtype, np.integer))
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions < num_buckets))


if __name__ == '__main__':
    unittest.main()
