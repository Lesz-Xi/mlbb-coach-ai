#!/usr/bin/env python3
"""
Test file for relaxed linting - this should NOT trigger linting errors now!
"""

import os, sys, json  # Multiple imports on one line
import numpy as np  # Unused import - should be ignored

def test_function(x, y):  # No type annotations - should be fine
    result = x + y
    unused_var = 42  # Unused variable - should be ignored
    
    # This line is intentionally very long to test the 120 character limit instead of 88 - should be fine now!
    very_long_string = "This is a very long string that would have failed the 88-character limit but should pass the 120-character limit"
    
    return result

# Test it
if __name__ == "__main__":
    print("✅ Relaxed linting test passed!")
    print(f"Result: {test_function(1, 2)}")
    print("No linting errors should appear in Cursor!") 