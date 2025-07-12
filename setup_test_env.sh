#!/bin/bash

# Setup Test Environment for Horse ID Unit Tests
# This script creates a clean test environment avoiding dependency conflicts

set -e

echo "=========================================="
echo "Setting Up Horse ID Test Environment"
echo "=========================================="

# Remove existing test environment if it exists
if [[ -d "test_env" ]]; then
    echo "Removing existing test environment..."
    rm -rf test_env
fi

# Create fresh virtual environment for testing
echo "Creating fresh test environment..."
python3 -m venv test_env

# Activate test environment
echo "Activating test environment..."
source test_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install test dependencies only
echo "Installing test dependencies..."
pip install -r test-requirements.txt

echo "=========================================="
echo "Test Environment Setup Complete!"
echo "=========================================="

echo "To activate the test environment:"
echo "  source test_env/bin/activate"
echo ""
echo "To run tests:"
echo "  source test_env/bin/activate"
echo "  pytest tests/ -v"
echo ""
echo "To run with coverage:"
echo "  source test_env/bin/activate"
echo "  pytest tests/ --cov=. --cov-report=html -v"