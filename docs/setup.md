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
git clone <repository-url>
cd agentic-framework-testing
```

### 2. Create Virtual Environment

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n agentic-testing python=3.10
conda activate agentic-testing
```

### 3. Install Core Dependencies

```bash
# Install core requirements
pip install -r requirements.txt

# For development
pip install -e .
```

### 4. Install Framework Dependencies

You can install all frameworks or only the ones you need:

```bash
# Install all frameworks (warning: large download)
pip install langgraph crewai autogen pydantic-ai haystack-ai llama-index dspy-ai

# Or install specific frameworks only
pip install langgraph  # For LangGraph
pip install crewai     # For CrewAI
# etc...
```

## Environment Setup

### 1. Create Environment File

```bash
cp .env.example .env
```

### 2. Configure API Keys

Edit `.env` and add your API keys:

```bash
# LLM Provider Keys (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
AZURE_OPENAI_API_KEY=...

# Optional: Framework-specific keys
LANGCHAIN_API_KEY=...
HUGGINGFACE_TOKEN=...

# Configuration
DEFAULT_MODEL=gpt-4o-mini  # or claude-3-haiku-20240307
DEFAULT_TEMPERATURE=0.7
MAX_TOKENS=2000
```

## Demo Mode Setup

Demo mode uses mock adapters and doesn't require API keys.

### Quick Demo Run

```bash
# Run demo with mock data
python3 demo.py

# Or use the demo script
./run_demo.sh
```

### Demo Configuration

Create `config/demo.yaml`:

```yaml
mode: demo
frameworks:
  - langgraph
  - crewai
use_cases:
  - movie_recommendation
  - github_triage
test_cases_per_use_case: 3
parallel: false
```

## Production Mode Setup

### 1. Verify API Keys

```bash
# Test API connectivity
python3 -c "from src.core.config import Config; Config().validate_api_keys()"
```

### 2. Configure Production Settings

Create `config/production.yaml`:

```yaml
mode: production
frameworks:
  - langgraph
  - crewai
  - autogen
  - pydantic_ai
  - haystack
use_cases:
  - movie_recommendation
  - github_triage
  - recipe_generation
  - research_summary
  - email_automation
test_cases_per_use_case: 10
parallel: true
max_workers: 4
timeout: 300
retry_attempts: 3
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

### 2. Verify Framework Installation

```bash
python3 -c "
from src.adapters import create_adapter
from src.core.types import FrameworkType
for fw in FrameworkType:
    try:
        adapter = create_adapter(fw)
        print(f'✓ {fw.value}: {adapter.get_framework_version()}')
    except Exception as e:
        print(f'✗ {fw.value}: {e}')
"
```

### 3. Test Single Framework

```bash
# Test a single framework
python3 -c "
from src.benchmark.runner import BenchmarkRunner
from src.core.types import FrameworkType, UseCaseType
runner = BenchmarkRunner()
results = runner.run_benchmark(
    frameworks=[FrameworkType.LANGGRAPH],
    use_cases=[UseCaseType.MOVIE_RECOMMENDATION],
    test_cases_per_use_case=1
)
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
