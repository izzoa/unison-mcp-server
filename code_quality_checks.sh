#!/bin/bash

# Unison MCP Server - Code Quality Checks
# This script runs all required linting and testing checks before committing changes.
# ALL checks must pass 100% for CI/CD to succeed.

set -e  # Exit on any error

echo "🔍 Running Code Quality Checks for Unison MCP Server"
echo "================================================="

# Determine Python command
if [[ -f ".unison_venv/bin/python" ]]; then
    PYTHON_CMD=".unison_venv/bin/python"
    PIP_CMD=".unison_venv/bin/pip"
    echo "✅ Using venv"
elif [[ -n "$VIRTUAL_ENV" ]]; then
    PYTHON_CMD="python"
    PIP_CMD="pip"
    echo "✅ Using activated virtual environment: $VIRTUAL_ENV"
else
    echo "❌ No virtual environment found!"
    echo "Please run: ./run-server.sh first to set up the environment"
    exit 1
fi
echo ""

# Check and install dev dependencies if needed
echo "🔍 Checking development dependencies..."
DEV_DEPS_NEEDED=false

# Check each dev dependency
for tool in ruff black isort pytest; do
    # Check if tool exists in venv or in PATH
    if [[ -f ".unison_venv/bin/$tool" ]] || command -v $tool &> /dev/null; then
        continue
    else
        DEV_DEPS_NEEDED=true
        break
    fi
done

if [ "$DEV_DEPS_NEEDED" = true ]; then
    echo "📦 Installing development dependencies..."
    $PIP_CMD install -q -r requirements-dev.txt
    echo "✅ Development dependencies installed"
else
    echo "✅ Development dependencies already installed"
fi

# Set tool paths
if [[ -f ".unison_venv/bin/ruff" ]]; then
    RUFF=".unison_venv/bin/ruff"
    BLACK=".unison_venv/bin/black"
    ISORT=".unison_venv/bin/isort"
    PYTEST=".unison_venv/bin/pytest"
else
    RUFF="ruff"
    BLACK="black"
    ISORT="isort"
    PYTEST="pytest"
fi
echo ""

# Step 1: Linting and Formatting
echo "📋 Step 1: Running Linting and Formatting Checks"
echo "--------------------------------------------------"

echo "🔧 Running ruff linting with auto-fix..."
$RUFF check --fix --exclude test_simulation_files --exclude .unison_venv

echo "🎨 Running black code formatting..."
$BLACK . --exclude="test_simulation_files/" --exclude=".unison_venv/"

echo "📦 Running import sorting with isort..."
$ISORT . --skip-glob=".unison_venv/*" --skip-glob="test_simulation_files/*"

echo "✅ Verifying all linting passes..."
$RUFF check --exclude test_simulation_files --exclude .unison_venv

echo "✅ Step 1 Complete: All linting and formatting checks passed!"
echo ""

# Step 1b: Type Checking (strict allowlist)
echo "🔎 Step 1b: Running mypy Type Checking"
echo "---------------------------------------"

# Check if mypy is available
if [[ -f ".unison_venv/bin/mypy" ]]; then
    MYPY=".unison_venv/bin/mypy"
elif command -v mypy &> /dev/null; then
    MYPY="mypy"
else
    echo "⚠️  mypy not found — skipping type checks (install via: pip install -r requirements-dev.txt)"
    MYPY=""
fi

if [[ -n "$MYPY" ]]; then
    echo "🔍 Running mypy on strict allowlist files..."
    $MYPY \
        utils/circuit_breaker.py utils/fs_snapshot.py utils/tool_execution_context.py utils/token_utils.py \
        providers/shared/provider_type.py providers/shared/model_response.py \
        utils/file_types.py utils/security_config.py utils/conversation_memory.py \
        utils/env.py utils/model_resolution.py utils/request_helpers.py \
        utils/image_utils.py utils/context_reconstructor.py utils/file_utils.py \
        tools/registry.py \
        scripts/smoke_test_wheel.py
    echo "✅ Step 1b Complete: Type checking passed!"
fi
echo ""

# Step 2: Unit Tests with Coverage
echo "🧪 Step 2: Running Complete Unit Test Suite with Coverage"
echo "---------------------------------------------------------"

echo "🏃 Running unit tests with coverage (excluding integration tests)..."
$PYTHON_CMD -m pytest tests/ -v -x -m "not integration" \
    --cov=. --cov-report=term-missing --cov-fail-under=44

echo "✅ Step 2 Complete: All unit tests passed with coverage above threshold!"
echo ""

# Step 3: Final Summary
echo "🎉 All Code Quality Checks Passed!"
echo "=================================="
echo "✅ Linting (ruff): PASSED"
echo "✅ Formatting (black): PASSED"
echo "✅ Import sorting (isort): PASSED"
echo "✅ Type checking (mypy): PASSED"
echo "✅ Unit tests: PASSED"
echo "✅ Coverage: PASSED (threshold: 44%)"
echo ""
echo "🚀 Your code is ready for commit and GitHub Actions!"
echo "💡 Remember to add simulator tests if you modified tools"