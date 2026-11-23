# Claude Development Guide for Unison MCP Server

This file contains essential commands and workflows for developing and maintaining the Unison MCP Server when working with Claude. Use these instructions to efficiently run quality checks, manage the server, check logs, and run tests.

## Quick Reference Commands

### Code Quality Checks

Before making any changes or submitting PRs, always run the comprehensive quality checks:

```bash
# Activate virtual environment first
source .unison_venv/bin/activate

# Run all quality checks (linting, formatting, tests)
./code_quality_checks.sh
```

This script automatically runs:
- Ruff linting with auto-fix
- Black code formatting 
- Import sorting with isort
- Mypy type checking (strict on allowlist files)
- README mockup drift check (docs/assets/mockups vs docs/mockup-scenes)
- Complete unit test suite (excluding integration tests)
- Verification that all checks pass 100%

**Run Integration Tests (requires API keys):**
```bash
# Run integration tests that make real API calls
./run_integration_tests.sh

# Run integration tests + simulator tests
./run_integration_tests.sh --with-simulator
```

### Server Management

#### Setup/Update the Server
```bash
# Run setup script (handles everything)
./run-server.sh
```

This script will:
- Set up Python virtual environment
- Install all dependencies
- Create/update .env file
- Configure MCP with Claude
- Verify API keys

#### View Logs
```bash
# Follow logs in real-time
./run-server.sh -f

# Or manually view logs
tail -f logs/mcp_server.log
```

### Log Management

#### View Server Logs
```bash
# View last 500 lines of server logs
tail -n 500 logs/mcp_server.log

# Follow logs in real-time
tail -f logs/mcp_server.log

# View specific number of lines
tail -n 100 logs/mcp_server.log

# Search logs for specific patterns
grep "ERROR" logs/mcp_server.log
grep "tool_name" logs/mcp_activity.log
```

#### Monitor Tool Executions Only
```bash
# View tool activity log (focused on tool calls and completions)
tail -n 100 logs/mcp_activity.log

# Follow tool activity in real-time
tail -f logs/mcp_activity.log

# Use simple tail commands to monitor logs
tail -f logs/mcp_activity.log | grep -E "(TOOL_CALL|TOOL_COMPLETED|ERROR|WARNING)"
```

#### Available Log Files

**Current log files (with proper rotation):**
```bash
# Main server log (all activity including debug info) - 20MB max, 5 backups
tail -f logs/mcp_server.log

# Tool activity only (TOOL_CALL, TOOL_COMPLETED, etc.) - 10MB max, 2 backups  
tail -f logs/mcp_activity.log
```

**For programmatic log analysis (used by tests):**
```python
# Import the LogUtils class from simulator tests
from simulator_tests.log_utils import LogUtils

# Get recent logs
recent_logs = LogUtils.get_recent_server_logs(lines=500)

# Check for errors
errors = LogUtils.check_server_logs_for_errors()

# Search for specific patterns
matches = LogUtils.search_logs_for_pattern("TOOL_CALL.*debug")
```

### Testing

Simulation tests are available to test the MCP server in a 'live' scenario, using your configured
API keys to ensure the models are working and the server is able to communicate back and forth. 

**IMPORTANT**: After any code changes, restart your Claude session for the changes to take effect.

#### Run All Simulator Tests
```bash
# Run the complete test suite
python communication_simulator_test.py

# Run tests with verbose output
python communication_simulator_test.py --verbose
```

#### Quick Test Mode (Recommended for Time-Limited Testing)
```bash
# Run quick test mode - 6 essential tests that provide maximum functionality coverage
python communication_simulator_test.py --quick

# Run quick test mode with verbose output
python communication_simulator_test.py --quick --verbose
```

**Quick mode runs these 6 essential tests:**
- `cross_tool_continuation` - Cross-tool conversation memory testing (chat, thinkdeep, codereview, analyze, debug)
- `basic_conversation` - Basic chat functionality
- `content_validation` - Content validation and duplicate detection
- `model_thinking_config` - Thinking configuration behavior on pro/flash
- `o3_model_selection` - O3 model selection and routing
- `per_tool_deduplication` - File deduplication for individual tools

**Why these 6 tests:** They cover the core functionality including conversation memory (`utils/conversation_memory.py`), chat tool functionality, file processing and deduplication, model selection (flash/flashlite/o3), and cross-tool conversation workflows. These tests validate the most critical parts of the system in minimal time.

**Note:** Some workflow tools (analyze, codereview, planner, consensus, etc.) require specific workflow parameters and are not in quick mode; run them individually (for example `consensus_workflow_accurate`, `codereview_validation`, `planner_validation`, `conversation_chain_validation`). `python communication_simulator_test.py --list-tests` is the authoritative list of tests.

#### Run Individual Simulator Tests (For Detailed Testing)
```bash
# List all available tests
python communication_simulator_test.py --list-tests

# RECOMMENDED: Run tests individually for better isolation and debugging
python communication_simulator_test.py --individual basic_conversation
python communication_simulator_test.py --individual content_validation
python communication_simulator_test.py --individual cross_tool_continuation
python communication_simulator_test.py --individual conversation_chain_validation

# Run multiple specific tests
python communication_simulator_test.py --tests basic_conversation content_validation

# Run individual test with verbose output for debugging
python communication_simulator_test.py --individual conversation_chain_validation --verbose
```

Available simulator tests include:
- `basic_conversation` - Basic conversation flow with chat tool
- `content_validation` - Content validation and duplicate detection
- `per_tool_deduplication` - File deduplication for individual tools
- `cross_tool_continuation` - Cross-tool conversation continuation scenarios
- `cross_tool_comprehensive` - Comprehensive cross-tool file deduplication and continuation
- `line_number_validation` - Line number handling validation across tools
- `model_thinking_config` - Model-specific thinking configuration behavior
- `o3_model_selection` - O3 model selection and usage validation
- `ollama_custom_url` - Ollama custom URL endpoint functionality
- `openrouter_fallback` - OpenRouter fallback behavior when only provider
- `openrouter_models` - OpenRouter model functionality and alias mapping
- `token_allocation_validation` - Token allocation and conversation history validation
- `testgen_validation` - TestGen tool validation with specific test function
- `refactor_validation` - Refactor tool validation with codesmells
- `conversation_chain_validation` - Conversation chain and threading validation
- `consensus_workflow_accurate` - Consensus tool concurrent consultation: all models consulted in step 1, which returns consensus_workflow_complete
- `consensus_three_models` - Consensus with three models and stances (for/against/neutral)
- `consensus_conversation` - Consensus conversation continuation via continuation_id

**Note**: All simulator tests should be run individually for optimal testing and better error isolation.

#### Run Unit Tests Only
```bash
# Run all unit tests (excluding integration tests that require API keys)
python -m pytest tests/ -v -m "not integration"

# Run specific test file
python -m pytest tests/test_refactor.py -v

# Run specific test function
python -m pytest tests/test_refactor.py::TestRefactorTool::test_format_response -v

# Run tests with coverage (enforces 44% minimum threshold)
python -m pytest tests/ --cov=. --cov-report=term-missing --cov-fail-under=44 -m "not integration"

# Generate HTML coverage report (browse htmlcov/index.html)
python -m pytest tests/ --cov=. --cov-report=html -m "not integration"
```

**Coverage Notes:**
- Baseline coverage is ~46%, threshold set to 44% to prevent regression
- `--cov-report=term-missing` shows uncovered line numbers in terminal output
- HTML report in `htmlcov/` provides line-by-line colour-coded coverage view
- CI uploads coverage reports as downloadable artifacts on each PR

#### Run Integration Tests (Uses Free Local Models)

**Setup Requirements:**
```bash
# 1. Install Ollama (if not already installed)
# Visit https://ollama.ai or use brew install ollama

# 2. Start Ollama service
ollama serve

# 3. Pull a model (e.g., llama3.2)
ollama pull llama3.2

# 4. Set environment variable for custom provider
export CUSTOM_API_URL="http://localhost:11434"
```

**Run Integration Tests:**
```bash
# Run integration tests (local Ollama models, recorded cassettes, locally installed CLIs, Docker)
python -m pytest tests/ -v -m "integration"

# Run specific integration test
python -m pytest tests/test_prompt_regression.py::TestPromptIntegration::test_chat_normal_prompt -v

# Run all tests (unit + integration)
python -m pytest tests/ -v
```

**Note**: The `integration` marker covers tests that need an external service, a real CLI, Docker, recorded cassettes or a local Ollama model. `tests/test_prompt_regression.py` uses the local-llama model via Ollama (FREE to run unlimited times) and needs `CUSTOM_API_URL` set to your local Ollama endpoint; `tests/test_consensus_integration.py` replays recorded OpenAI/Gemini cassettes (no network, no keys needed unless a cassette is missing); `tests/test_clink_integration.py` and `tests/test_clink_opencode_v2_integration.py` spawn locally installed CLIs and skip when they are absent; `tests/test_docker_mcp_validation.py` needs Docker. They can be run safely in CI/CD but are excluded from code quality checks to keep them fast.

### Development Workflow

#### Before Making Changes
1. Ensure virtual environment is activated: `source .unison_venv/bin/activate`
2. Run quality checks: `./code_quality_checks.sh`
3. Check logs to ensure server is healthy: `tail -n 50 logs/mcp_server.log`

#### After Making Changes
1. Run quality checks again: `./code_quality_checks.sh`
2. Run integration tests locally: `./run_integration_tests.sh`
3. Run quick test mode for fast validation: `python communication_simulator_test.py --quick`
4. Run relevant specific simulator tests if needed: `python communication_simulator_test.py --individual <test_name>`
5. Check logs for any issues: `tail -n 100 logs/mcp_server.log`
6. Restart Claude session to use updated code

#### Before Committing/PR

> **MANDATORY: Update CHANGELOG.md and README.md BEFORE committing.**
> A pre-commit hook enforces this — commits with staged `.py` files will be
> **rejected** if `CHANGELOG.md` is not also staged. Do not bypass with `--no-verify`
> unless the commit is purely non-code (docs, config, CI).

1. **Update `CHANGELOG.md`** under the `[Unreleased]` section for ANY code changes. Use [Keep a Changelog](https://keepachangelog.com/) categories: Added, Changed, Deprecated, Removed, Fixed, Security. This is not optional.
2. **Update `README.md`** for substantial feature releases — specifically the "What this fork adds over PAL" callout, the "PAL vs Unison: full comparison" table and the "Key Features" section. Skip only for internal refactors with no user-facing change.
3. Final quality check: `./code_quality_checks.sh`
4. Run integration tests: `./run_integration_tests.sh`
5. Run quick test mode: `python communication_simulator_test.py --quick`
6. Run full simulator test suite (optional): `./run_integration_tests.sh --with-simulator`
7. Verify all tests pass 100%

### Common Troubleshooting

#### Server Issues
```bash
# Check if Python environment is set up correctly
./run-server.sh

# View recent errors
grep "ERROR" logs/mcp_server.log | tail -20

# Check virtual environment
which python
# Should show: .../unison-mcp-server/.unison_venv/bin/python

# Capture the actual stdin sent to a clink CLI (debugging hallucinated paths,
# wrong outputs, etc.). Set LOG_LEVEL=DEBUG in .env then grep for the
# "Executing CLI command" line emitted by clink.runner.<cli_name>:
LOG_LEVEL=DEBUG ./run-server.sh
grep "clink.runner.opencode" logs/mcp_server.log | tail -10
```

#### Test Failures
```bash
# First try quick test mode to see if it's a general issue
python communication_simulator_test.py --quick --verbose

# Run individual failing test with verbose output
python communication_simulator_test.py --individual <test_name> --verbose

# Check server logs during test execution
tail -f logs/mcp_server.log

# Run tests with debug output
LOG_LEVEL=DEBUG python communication_simulator_test.py --individual <test_name>
```

#### Linting Issues
```bash
# Auto-fix most linting issues
ruff check . --fix
black .
isort .

# Check what would be changed without applying
ruff check .
black --check .
isort --check-only .
```

#### Type Checking (mypy)
```bash
# The full strict allowlist is also Step 1b of ./code_quality_checks.sh (which additionally runs ruff --fix,
# black, isort, the mockup check and the unit suite). To run only the type check, reuse the script's file list:
mypy $(sed -n '/\$MYPY \\/,/^    echo/p' code_quality_checks.sh | grep -oE '[A-Za-z_/0-9]+\.py')

# Run mypy on a specific file
mypy utils/circuit_breaker.py
```

**Gradual strictness strategy:** the modules listed in `pyproject.toml` `[[tool.mypy.overrides]]` are under strict mypy checking (currently 45; `code_quality_checks.sh`, `code_quality_checks.ps1` and `.github/workflows/test.yml` carry the same list). Other packages run with permissive defaults.

> **MANDATORY: When creating new `.py` files or substantially refactoring existing ones, you MUST
> run `mypy --strict <file>` on the new/changed file.** If it passes with zero errors, add it to
> the strict allowlist in **all four places**:
> 1. `pyproject.toml` — `[[tool.mypy.overrides]]` module list
> 2. `code_quality_checks.sh` — the `$MYPY` command's file list
> 3. `code_quality_checks.ps1` — the `$mypyFiles` list
> 4. `.github/workflows/test.yml` — the `mypy` step's file list
>
> If strict mypy fails due to third-party library issues (untyped decorators, missing stubs),
> document the blockers in the commit message and skip adding the file — do not weaken the
> strictness settings to force it through.

### File Structure Context

- `./code_quality_checks.sh` - Comprehensive quality check script
- `./run-server.sh` - Server setup and management
- `communication_simulator_test.py` - End-to-end testing framework
- `simulator_tests/` - Individual test modules
- `tests/` - Unit test suite
- `tools/` - MCP tool implementations
- `providers/` - AI provider implementations
- `systemprompts/` - System prompt definitions
- `logs/` - Server log files

### Environment Requirements

- Python 3.10+ with virtual environment
- All dependencies from `requirements.lock.txt` installed (the committed lock; ranges live in `pyproject.toml`)
- Proper API keys configured in `.env` file

This guide provides everything needed to efficiently work with the Unison MCP Server codebase using Claude. Always run quality checks before and after making changes to ensure code integrity.