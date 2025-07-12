#!/bin/bash

# Horse ID Unit Test Runner
# Comprehensive test suite with coverage reporting

set -e

echo "=========================================="
echo "Horse ID Unit Test Suite"
echo "=========================================="

# Check if virtual environment exists
if [[ ! -d "venv" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install test requirements
echo "Installing test requirements..."
pip install -q -r test-requirements.txt

# Create test results directory
mkdir -p test-results

echo "=========================================="
echo "Running All Tests with Coverage"
echo "=========================================="

# Run comprehensive test suite with coverage
pytest tests/ \
    --verbose \
    --tb=short \
    --cov=. \
    --cov-report=html:test-results/coverage-html \
    --cov-report=term-missing \
    --cov-report=xml:test-results/coverage.xml \
    --junitxml=test-results/junit.xml

echo "=========================================="
echo "Test Results Summary"
echo "=========================================="

echo "✅ All tests passed!"
echo "📊 Coverage report: test-results/coverage-html/index.html"
echo "📄 JUnit XML: test-results/junit.xml"
echo "📈 Coverage XML: test-results/coverage.xml"

# Optional: Run the zero-dependency simple tests
echo ""
echo "Running zero-dependency validation tests..."
python test_simple.py

echo ""
echo "🎉 Test suite completed successfully!"