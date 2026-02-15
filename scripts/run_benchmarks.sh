#!/bin/bash

# Run All Benchmarks Script
# Executes complete evaluation suite

set -e

echo "=================================="
echo "Running Complete Benchmark Suite"
echo "=================================="

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found"
    echo "Run: bash scripts/setup_env.sh"
    exit 1
fi

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ ANTHROPIC_API_KEY not set"
    echo "Add to .env file or export ANTHROPIC_API_KEY=your_key"
    exit 1
fi

echo ""
echo "API Key: ✓ Found"

# Run evaluation
echo ""
echo "Starting benchmark evaluation..."
echo ""

python examples/04_evaluation_demo.py

echo ""
echo "=================================="
echo "Benchmarks Complete!"
echo "=================================="
echo ""
echo "Results saved to: EVALUATION_RESULTS.md"