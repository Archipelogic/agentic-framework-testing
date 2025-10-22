# Quick Start Guide

## 5-Minute Setup

Get up and running with the Agentic AI Framework Testing Harness in just 5 minutes!

## Step 1: Clone and Setup (2 minutes)

```bash
# Clone the repository
git clone <repository-url>
cd agentic-framework-testing

# Run automated setup (creates venv, installs deps, generates data)
./run.sh setup
```

## Step 2: Run Your First Evaluation (1 minute)

```bash
# Run mock evaluation (no API keys needed)
./run.sh test
```

## Step 3: View Results (1 minute)

```bash
# Open the interactive HTML report
./run.sh report
```

Expected output:
```
🚀 Agentic Framework Testing Harness
==================================================
🏃 Running benchmark tests...
  Mode: MOCK
  Frameworks: 13
  Use cases: 5
  Using ALL test data (200 per use case)
==================================================

Testing frameworks: 100%|██████████| 13/13
✅ Report opened successfully

Testing crewai...
  ✓ movie_recommendation: Success (0.4s)
  ✓ github_triage: Success (0.6s)

📊 Generating reports...
✅ Report saved to: benchmark_results/report_20241021_143022.html
```

## Running Your First Real Benchmark

### Option 1: Command Line Interface

```bash
# Run benchmark with specific frameworks and use cases
python3 -m src.cli benchmark \
  --frameworks langgraph crewai \
  --use-cases movie_recommendation github_triage \
  --test-cases 5 \
  --output-dir results
```

### Option 2: Python Script

Create `my_benchmark.py`:

```python
from src.benchmark.runner import BenchmarkRunner
from src.core.types import FrameworkType, UseCaseType
from src.reporting.unified_reporter import UnifiedReporter

# Initialize runner
runner = BenchmarkRunner()

# Run benchmark
results = runner.run_benchmark(
    frameworks=[
        FrameworkType.LANGGRAPH,
        FrameworkType.CREWAI
    ],
    use_cases=[
        UseCaseType.MOVIE_RECOMMENDATION,
        UseCaseType.GITHUB_TRIAGE
    ],
    test_cases_per_use_case=5,
    parallel=True
)

# Generate report
reporter = UnifiedReporter()
report_path = reporter.generate_comprehensive_report(results)
print(f"Report generated: {report_path}")
```

Run it:
```bash
python3 my_benchmark.py
```

### Option 3: Interactive Mode

```bash
# Start interactive benchmark session
python3 -m src.cli interactive

# Follow the prompts:
# 1. Select frameworks (space to select, enter to confirm)
# 2. Select use cases
# 3. Configure test parameters
# 4. Review and confirm
# 5. Run benchmark
```

## Understanding the Output

### Report Structure

The generated HTML report contains:

1. **Executive Summary**
   - Overall success rate
   - Total tests run
   - Framework rankings

2. **Performance Metrics**
   - Accuracy scores
   - Latency measurements
   - Cost analysis

3. **Capability Matrix**
   - Framework strengths/weaknesses
   - Use case compatibility

4. **Detailed Results**
   - Individual test results
   - Error logs
   - Response samples

### Interpreting Results

```python
# Load and analyze results programmatically
import json

with open('benchmark_results/latest/results.json') as f:
    results = json.load(f)

# Check overall winner
winner = results['summary']['overall_winner']
print(f"Best performing framework: {winner}")

# Get framework rankings
rankings = results['summary']['framework_rankings']
for framework, data in rankings.items():
    print(f"{framework}: Rank {data['rank']}, Score {data['score']:.2f}")

# Analyze specific use case
use_case_results = results['results']['langgraph']['movie_recommendation']
print(f"LangGraph movie accuracy: {use_case_results['aggregate_metrics']['avg_accuracy']:.2%}")
```

## Common Benchmark Scenarios

### 1. Compare Two Frameworks Head-to-Head

```bash
python3 -m src.cli compare \
  --framework1 langgraph \
  --framework2 crewai \
  --use-case github_triage
```

### 2. Test New Framework Integration

```bash
# Test a single framework thoroughly
python3 -m src.cli test-framework \
  --framework autogen \
  --all-use-cases \
  --verbose
```

### 3. Production Benchmark

```bash
# Full production run with all frameworks
python3 -m src.cli benchmark \
  --config config/production.yaml \
  --parallel \
  --max-workers 8 \
  --output-format json html csv
```

### 4. Cost-Optimized Testing

```bash
# Run with cost limits
ENABLE_COST_TRACKING=true \
MAX_COST_PER_RUN=5.0 \
python3 -m src.cli benchmark \
  --frameworks all \
  --use-cases movie_recommendation \
  --test-cases 3
```

## Quick Tips

### 1. Use Demo Mode for Development

```python
# Set demo mode in code
from src.core.config import Config

config = Config()
config.mode = "demo"  # No API calls, uses mock data
```

### 2. Speed Up Testing with Caching

```bash
# Enable response caching
export ENABLE_RESPONSE_CACHE=true
export CACHE_TTL=3600
```

### 3. Debug Failed Tests

```bash
# Run with debug logging
DEBUG=true python3 -m src.cli benchmark \
  --frameworks crewai \
  --use-cases github_triage \
  --test-cases 1 \
  --verbose
```

### 4. Monitor Progress

```bash
# Use progress bars and live updates
python3 -m src.cli benchmark \
  --show-progress \
  --live-updates
```

## Troubleshooting Quick Fixes

### API Key Issues
```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API connection
python3 -c "import openai; print(openai.Model.list())"
```

### Import Errors
```bash
# Reinstall a specific framework
pip install --upgrade langgraph

# Verify installation
python3 -c "import langgraph; print(langgraph.__version__)"
```

### Out of Memory
```bash
# Reduce parallel workers
export MAX_PARALLEL_WORKERS=2

# Or disable parallel execution
python3 demo.py --no-parallel
```

## What's Next?

Now that you've run your first benchmark:

1. **[Add more frameworks](./frameworks.md)** - Test additional AI frameworks
2. **[Create custom use cases](./use-cases.md)** - Evaluate domain-specific scenarios
3. **[Configure for production](./configuration.md)** - Set up for large-scale testing
4. **[Understand the metrics](./evaluation.md)** - Deep dive into evaluation criteria
5. **[Deploy to cloud](./deployment.md)** - Scale your benchmarks

## Getting Help

- **Logs**: Check `benchmark_results/latest/logs/`
- **Debug mode**: Set `DEBUG=true` environment variable
- **Documentation**: See [Troubleshooting Guide](./troubleshooting.md)
- **Community**: File issues on GitHub
