#!/usr/bin/env python3
"""
Test runner for Silica Sort test suite.
Discovers and runs all unittest cases in the test_suite folder.
"""

import sys
import unittest

def main():
    # Discover all tests in this directory
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='test_suite', pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with code 0 if successful, 1 if failures/errors occurred
    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == '__main__':
    main()
