#!/bin/bash

# Quick Test Runner - Fast execution without coverage
# Use this for rapid development feedback

set -e

echo "=========================================="
echo "Horse ID Quick Test Runner"
echo "=========================================="

# Activate virtual environment if it exists
if [[ -d "venv" ]]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Please run ./run_tests.sh first."
    exit 1
fi

# Ensure test dependencies are installed
pip install -q -r test-requirements.txt

# Run tests without coverage for speed
pytest tests/ --tb=short --quiet

echo "✅ All tests passed!"

# Run the optimal high-confidence subset
echo ""
echo "Running high-confidence validation..."
./run_tests_optimal.sh