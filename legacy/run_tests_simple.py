#!/usr/bin/env python3
"""
Simple test runner for Horse ID unit tests.
This bypasses dependency conflicts by running tests in isolation.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors gracefully."""
    print(f"\n{'='*50}")
    print(f"{description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"Warning: Python {version.major}.{version.minor} detected. Python 3.8+ recommended.")
    else:
        print(f"Python {version.major}.{version.minor}.{version.micro} detected. Good!")

def install_minimal_deps():
    """Install only the minimal dependencies needed for testing."""
    minimal_deps = [
        "pytest>=7.0.0",
        "pytest-mock>=3.10.0", 
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "PyYAML>=6.0",
        "requests>=2.25.0"
    ]
    
    for dep in minimal_deps:
        success = run_command(f"pip install '{dep}'", f"Installing {dep}")
        if not success:
            print(f"Warning: Failed to install {dep}")

def run_individual_test_files():
    """Run each test file individually to isolate dependency issues."""
    test_files = [
        "tests/test_detection_algorithms.py",
        "tests/test_webhook_responder.py", 
        "tests/test_horse_processor.py",
        "tests/test_image_processing.py",
        "tests/test_config_loading.py",
        "tests/test_email_ingestion.py",
        "tests/test_identity_merging.py"
    ]
    
    results = {}
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n{'='*60}")
            print(f"Running {test_file}")
            print(f"{'='*60}")
            
            # Set PYTHONPATH to include current directory
            env = os.environ.copy()
            env['PYTHONPATH'] = '.'
            
            try:
                cmd = [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"]
                result = subprocess.run(cmd, env=env, capture_output=True, text=True)
                
                print(result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
                
                results[test_file] = result.returncode == 0
                
            except Exception as e:
                print(f"Error running {test_file}: {e}")
                results[test_file] = False
        else:
            print(f"Test file not found: {test_file}")
            results[test_file] = False
    
    return results

def main():
    """Main test execution function."""
    print("Horse ID Unit Test Runner (Dependency-Safe)")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Install minimal dependencies
    print("\nInstalling minimal test dependencies...")
    install_minimal_deps()
    
    # Run tests individually
    print("\nRunning individual test files...")
    results = run_individual_test_files()
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test_file, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{test_file:<40} {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed + failed}, Passed: {passed}, Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {failed} test file(s) failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)