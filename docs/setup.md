# Setup Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Environment Setup](#environment-setup)
4. [Demo Mode Setup](#demo-mode-setup)
5. [Production Mode Setup](#production-mode-setup)
6. [Verification](#verification)

## Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows (with WSL)
- **Python**: 3.8 or higher (3.10+ recommended)
- **Memory**: Minimum 8GB RAM (16GB recommended for parallel execution)
- **Storage**: At least 5GB free space

### Software Dependencies
- Git
- pip or conda
- Virtual environment tools (venv, virtualenv, or conda)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Archipelogic/agentic-framework-testing.git
cd agentic-framework-testing
```

### 2. Automated Setup (Recommended)

The easiest way to set up the project is using the unified runner script:

```bash
# This will create venv, install dependencies, and generate test data
./run.sh setup
```

### 3. Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development
pip install -e .
```

### 4. Framework Support

All 13 frameworks are supported through mock adapters for testing:
- **Graph-Based**: LangGraph, Haystack
- **Multi-Agent**: CrewAI, AutoGen, BeeAI  
- **Type-Safe**: Pydantic AI, Atomic Agents
- **Optimized**: DSPy, Agno, Smolagents
- **Cloud-Native**: AWS Bedrock AgentCore, Strands
- **RAG-Focused**: LlamaIndex

Note: Most frameworks use mock adapters. Real framework packages can be installed if available on PyPI.

## Environment Setup

### 1. Create Environment File

```bash
cp .env.example .env
```

### 2. Configure API Keys

Edit `.env` and add your API keys:

```bash
# LLM Provider Keys (at least one required for live mode)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
AZURE_OPENAI_API_KEY=...

# AWS Configuration (for Bedrock frameworks)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
AWS_BEDROCK_ENABLED=true  # Auto-enabled when AWS credentials present

# Configuration
DEFAULT_MODEL=gpt-4  # or gpt-4o-mini, claude-3-haiku, etc.
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=2048
```

## Demo Mode Setup

Demo mode uses mock adapters and doesn't require API keys.

### Quick Demo Run

```bash
# Run mock evaluation (no API keys needed)
./run.sh test --mock

# Or run with Python directly
python3 run_evaluation.py --mock

# Quick test with 3 frameworks
./run.sh test --mock --quick

# Test with specific sample size
./run.sh test --mock --samples 50
```

### Demo Features

- **No API keys required** - Uses mock adapters
- **Auto-generates test data** - Creates data/ directory if missing
- **All 13 frameworks** - Tests complete framework set
- **5 real-world use cases** - Comprehensive evaluation
- **HTML reports** - Auto-opens in browser

## Production Mode Setup

### 1. Verify API Keys

```bash
# Test API connectivity
python3 -c "from src.core.config import Config; Config().validate_api_keys()"
```

### 2. Run Live Evaluation

```bash
# Run with real API calls
./run.sh test --live

# Or with specific options
python3 run_evaluation.py --live --samples 100 --parallel
```

### 3. Production Options

```bash
# Run with all options
./run.sh test --live \
  --samples 200 \      # Use all test data
  --parallel \         # Parallel execution
  --no-open            # Don't auto-open report

# View latest report
./run.sh report

# Clean old results
./run.sh clean
```

### 3. Set Resource Limits

```bash
# Set environment variables for production
export MAX_PARALLEL_WORKERS=4
export REQUEST_TIMEOUT=60
export MAX_RETRIES=3
export RATE_LIMIT_DELAY=1
```

## Verification

### 1. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_integration.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### 2. Verify Framework Support

```bash
python3 -c "
from src.adapters.framework_adapters import create_adapter
from src.core.types import FrameworkType
config = {'model': 'gpt-4', 'aws_region': 'us-east-1'}
for fw in FrameworkType:
    try:
        adapter = create_adapter(fw, config)
        print(f'✓ {fw.value}: {adapter.get_framework_version()}')
    except Exception as e:
        print(f'✗ {fw.value}: {e}')
"
```

### 3. Test Single Framework

```bash
# Test a single framework
python3 -c "
from run_evaluation import UnifiedBenchmarkRunner
runner = UnifiedBenchmarkRunner(mode='mock', quick=True)
results = runner.run()
print(f'Success: {bool(results)}')
"
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'xxx'**
   - Solution: Install the missing framework with `pip install xxx`

2. **API Key errors**
   - Solution: Verify `.env` file exists and contains valid keys
   - Check: `echo $OPENAI_API_KEY` to verify environment variable

3. **Memory errors during parallel execution**
   - Solution: Reduce `MAX_PARALLEL_WORKERS` or disable parallel mode

4. **Rate limiting errors**
   - Solution: Increase `RATE_LIMIT_DELAY` in environment variables

### Getting Help

- Check [Troubleshooting Guide](./troubleshooting.md) for detailed solutions
- Review logs in `benchmark_results/` directory
- Enable debug mode: `export DEBUG=true`

## Next Steps

- [Configure your frameworks](./configuration.md)
- [Run your first benchmark](./quickstart.md)
- [Understand the evaluation system](./evaluation.md)
