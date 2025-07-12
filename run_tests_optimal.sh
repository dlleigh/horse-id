#!/bin/bash

# High-Confidence Test Runner
# Validates core functionality with minimal dependencies

set -e

echo "=========================================="
echo "Horse ID Core Functionality Tests"
echo "=========================================="

# Core tests that validate critical functionality
CORE_TESTS=(
    "tests/test_webhook_responder.py"
    "tests/test_detection_algorithms.py::TestBboxOverlap"
    "tests/test_detection_algorithms.py::TestDistanceFromCenter"
    "tests/test_detection_algorithms.py::TestEdgeCropping"
    "tests/test_config_loading.py::TestConfigValidation"
)

echo "Running core functionality tests..."

passed=0
total=${#CORE_TESTS[@]}

for test_spec in "${CORE_TESTS[@]}"; do
    if pytest "$test_spec" --tb=short --quiet; then
        echo "✅ $test_spec"
        ((passed++))
    else
        echo "❌ $test_spec"
    fi
done

echo ""
echo "Running zero-dependency validation..."
if python test_simple.py > /dev/null 2>&1; then
    echo "✅ Zero-dependency tests"
    ((passed++))
    ((total++))
else
    echo "❌ Zero-dependency tests"
    ((total++))
fi

echo ""
echo "=========================================="
echo "Results: $passed/$total tests passed"
echo "=========================================="

if [ $passed -eq $total ]; then
    echo "🎉 All core tests passed!"
    exit 0
else
    echo "⚠️  Some core tests failed"
    exit 1
fi