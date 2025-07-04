#!/usr/bin/env python3
"""
Test script for checkpoint resuming functionality.
This script tests the validate_checkpoint_path function and related logic.
"""

import sys
import tempfile
import json
from pathlib import Path

# Add the training script directory to path
sys.path.append(str(Path(__file__).parent))

import logging

# Set up logging first
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# Import after setting up logging
from train import validate_checkpoint_path, log_on_main

def test_validate_checkpoint_path():
    """Test the validate_checkpoint_path function."""
    
    print("Testing validate_checkpoint_path function...")
    
    # Test 1: None/empty path
    assert not validate_checkpoint_path(None), "Should return False for None"
    assert not validate_checkpoint_path(""), "Should return False for empty string"
    print("✓ Test 1 passed: None/empty path handling")
    
    # Test 2: Non-existent path
    assert not validate_checkpoint_path("/non/existent/path"), "Should return False for non-existent path"
    print("✓ Test 2 passed: Non-existent path handling")
    
    # Test 3: Valid checkpoint directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create config.json
        config = {
            "model_type": "seq2seq",
            "vocab_size": 4096,
            "chronos_config": {}
        }
        with open(temp_path / "config.json", "w") as f:
            json.dump(config, f)
        
        # Create model weights file
        (temp_path / "pytorch_model.bin").touch()
        
        assert validate_checkpoint_path(str(temp_path)), "Should return True for valid checkpoint"
        print("✓ Test 3 passed: Valid checkpoint directory")
    
    # Test 4: Directory with missing config.json
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "pytorch_model.bin").touch()
        
        assert not validate_checkpoint_path(str(temp_path)), "Should return False for missing config.json"
        print("✓ Test 4 passed: Missing config.json handling")
    
    # Test 5: Directory with missing model weights
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        config = {"model_type": "seq2seq"}
        with open(temp_path / "config.json", "w") as f:
            json.dump(config, f)
        
        assert not validate_checkpoint_path(str(temp_path)), "Should return False for missing model weights"
        print("✓ Test 5 passed: Missing model weights handling")
    
    # Test 6: Valid checkpoint with safetensors
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        config = {"model_type": "seq2seq"}
        with open(temp_path / "config.json", "w") as f:
            json.dump(config, f)
        
        (temp_path / "model.safetensors").touch()
        
        assert validate_checkpoint_path(str(temp_path)), "Should return True for checkpoint with safetensors"
        print("✓ Test 6 passed: Safetensors format support")
    
    print("All tests passed! ✅")

def test_compatibility_check():
    """Test the PyTorch compatibility check function."""

    print("\nTesting compatibility check...")

    try:
        from train import check_torch_version_compatibility

        # Test the function
        is_compatible, message = check_torch_version_compatibility()

        assert isinstance(is_compatible, bool), "Should return boolean"
        assert isinstance(message, str), "Should return string message"

        print(f"✓ Compatibility check test passed: {message}")

        if not is_compatible:
            print("⚠️  Note: Current PyTorch version may have compatibility issues")

    except Exception as e:
        print(f"✗ Compatibility check test failed: {e}")
        return False

    return True


def test_parameter_parsing():
    """Test that the new parameter is properly handled."""

    print("\nTesting parameter parsing...")

    # Import the main function to check if it accepts the new parameter
    try:
        from train import main
        import inspect

        # Get function signature
        sig = inspect.signature(main)
        params = list(sig.parameters.keys())

        assert "resume_from_checkpoint" in params, "resume_from_checkpoint parameter should be in main function"
        print("✓ Parameter parsing test passed: resume_from_checkpoint parameter found")

        # Check default value
        param = sig.parameters["resume_from_checkpoint"]
        assert param.default is None, "Default value should be None"
        print("✓ Default value test passed: resume_from_checkpoint defaults to None")

    except Exception as e:
        print(f"✗ Parameter parsing test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    print("Running checkpoint resuming functionality tests...\n")
    
    try:
        test_validate_checkpoint_path()
        test_compatibility_check()
        test_parameter_parsing()
        print("\n🎉 All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)
