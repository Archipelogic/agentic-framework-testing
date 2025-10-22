# API Reference

## Core Types

### FrameworkType

Enumeration of supported frameworks.

```python
from src.core.types import FrameworkType

# Available frameworks
FrameworkType.LANGGRAPH
FrameworkType.CREWAI
FrameworkType.AUTOGEN
FrameworkType.PYDANTIC_AI
FrameworkType.HAYSTACK
FrameworkType.LLAMAINDEX
FrameworkType.DSPY
FrameworkType.AGNO
FrameworkType.BEEAI
FrameworkType.STRANDS_AGENTS
FrameworkType.BEDROCK_AGENTCORE
FrameworkType.SMOLAGENTS
FrameworkType.ATOMIC_AGENTS
```

### UseCaseType

Enumeration of evaluation use cases.

```python
from src.core.types import UseCaseType

# Available use cases
UseCaseType.MOVIE_RECOMMENDATION
UseCaseType.GITHUB_TRIAGE
UseCaseType.RECIPE_GENERATION
UseCaseType.RESEARCH_SUMMARY
UseCaseType.EMAIL_AUTOMATION
```

### TestCase

Data class for test cases.

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TestCase:
    """Individual test case."""
    
    id: str
    use_case: UseCaseType
    input_data: Dict[str, Any]
    ground_truth: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### FrameworkResult

Result from framework execution.

```python
@dataclass
class FrameworkResult:
    """Result from framework execution."""
    
    succeeded: bool
    result: Dict[str, Any]
    latency_seconds: float
    tokens_used: int = 0
    cost_usd: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## UnifiedBenchmarkRunner

Main class for running benchmarks.

### Constructor

```python
from run_evaluation import UnifiedBenchmarkRunner
from src.core.config import Config

runner = UnifiedBenchmarkRunner(
    mode: str = 'mock',  # 'mock' or 'live'
    config: Optional[Config] = None,
    quick: bool = False,
    samples: Optional[int] = None,
    parallel: bool = False,
    verbose: bool = False
)
```

**Parameters:**
- `config` (Optional[Config]): Configuration object. Uses default if not provided.

### run_benchmark

Run a comprehensive benchmark.

```python
results = runner.run_benchmark(
    frameworks: List[FrameworkType],
    use_cases: List[UseCaseType],
    test_cases_per_use_case: int = 5,
    parallel: bool = False
) -> Dict[str, Any]
```

**Parameters:**
- `frameworks` (List[FrameworkType]): Frameworks to evaluate
- `use_cases` (List[UseCaseType]): Use cases to test
- `test_cases_per_use_case` (int): Number of test cases per use case
- `parallel` (bool): Whether to run in parallel

**Returns:**
```python
{
    'benchmark_id': str,
    'timestamp': str,
    'config': Dict,
    'results': Dict[str, Dict[str, Any]],
    'summary': Dict[str, Any]
}
```

**Example:**
```python
from run_evaluation import UnifiedBenchmarkRunner

runner = UnifiedBenchmarkRunner(mode='mock')
results = runner.run()  # Runs all frameworks and use cases

# Or with options
runner = UnifiedBenchmarkRunner(
    mode='live',
    quick=True,  # Only test 3 frameworks
    samples=50,  # 50 samples per use case
    test_cases_per_use_case=10,
    parallel=True
)

print(f"Benchmark ID: {results['benchmark_id']}")
print(f"Overall winner: {results['summary']['overall_winner']}")
```

## Framework Adapters

### BaseFrameworkAdapter

Abstract base class for framework adapters.

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseFrameworkAdapter(ABC):
    """Base adapter interface."""
    
    @abstractmethod
    def get_framework_name(self) -> FrameworkType:
        """Get framework type."""
        pass
    
    @abstractmethod
    def get_framework_version(self) -> str:
        """Get framework version."""
        pass
    
    @abstractmethod
    def supports_use_case(self, use_case: UseCaseType) -> bool:
        """Check if framework supports use case."""
        pass
    
    @abstractmethod
    def run(
        self,
        use_case: UseCaseType,
        input_data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> FrameworkResult:
        """Execute framework on use case."""
        pass
```

### create_adapter

Factory function to create framework adapters.

```python
from src.adapters import create_adapter

adapter = create_adapter(
    framework: FrameworkType,
    config: Optional[Dict[str, Any]] = None
) -> BaseFrameworkAdapter
```

**Parameters:**
- `framework` (FrameworkType): Framework to create adapter for
- `config` (Optional[Dict]): Framework-specific configuration

**Example:**
```python
from src.adapters import create_adapter
from src.core.types import FrameworkType

# Create LangGraph adapter
adapter = create_adapter(
    FrameworkType.LANGGRAPH,
    config={'max_iterations': 10}
)

# Check version
version = adapter.get_framework_version()
print(f"LangGraph version: {version}")

# Run use case
result = adapter.run(
    UseCaseType.MOVIE_RECOMMENDATION,
    {"preferences": ["action", "sci-fi"]}
)
```

## Evaluator

### Evaluator Class

Evaluates framework outputs against ground truth.

```python
from src.metrics.evaluator import Evaluator

evaluator = Evaluator(config: Optional[Dict[str, Any]] = None)
```

### evaluate

Evaluate framework output.

```python
scores = evaluator.evaluate(
    output: Dict[str, Any],
    ground_truth: Dict[str, Any],
    use_case: UseCaseType
) -> Dict[str, float]
```

**Parameters:**
- `output` (Dict): Framework output
- `ground_truth` (Dict): Expected output
- `use_case` (UseCaseType): Use case being evaluated

**Returns:**
```python
{
    'accuracy': float,        # 0-1 score
    'completeness': float,    # 0-1 score
    'format_validity': float, # 0-1 score
    'overall': float         # Weighted average
}
```

**Example:**
```python
from src.metrics.evaluator import Evaluator
from src.core.types import UseCaseType

evaluator = Evaluator()

output = {
    "recommendations": [
        {"title": "Movie 1", "genres": ["action"]}
    ]
}

ground_truth = {
    "expected_genres": ["action", "sci-fi"],
    "min_recommendations": 3
}

scores = evaluator.evaluate(
    output,
    ground_truth,
    UseCaseType.MOVIE_RECOMMENDATION
)

print(f"Accuracy: {scores['accuracy']:.2%}")
```

## Configuration

### Config Class

Configuration management.

```python
from src.core.config import Config

config = Config(
    config_file: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
)
```

### Attributes

```python
config.default_model: str           # Default LLM model
config.default_temperature: float   # Default temperature
config.max_tokens: int              # Max tokens per request
config.api_keys: Dict[str, str]    # API keys
config.mode: str                    # "demo", "development", "production"
config.output_dir: Path            # Output directory
```

### Methods

```python
# Get framework-specific config
framework_config = config.get_framework_config("langgraph")

# Validate configuration
is_valid = config.validate()

# Load from file
config.load_from_file("config.yaml")

# Save to file
config.save_to_file("config.yaml")
```

**Example:**
```python
from src.core.config import Config

# Create config with overrides
config = Config(
    config_file="config/production.yaml",
    overrides={
        "default_model": "gpt-4o",
        "max_tokens": 3000
    }
)

# Access configuration
print(f"Model: {config.default_model}")
print(f"Mode: {config.mode}")

# Get framework config
langgraph_config = config.get_framework_config("langgraph")
```

## Reporter

### UnifiedReporter

Generate reports from benchmark results.

```python
from src.reporting.unified_reporter import UnifiedReporter

reporter = UnifiedReporter(
    output_dir: str = "benchmark_results"
)
```

### generate_comprehensive_report

Generate HTML report.

```python
report_path = reporter.generate_comprehensive_report(
    results_file: Union[str, Path, Dict],
    auto_open: bool = True
) -> Path
```

**Parameters:**
- `results_file` (Union[str, Path, Dict]): Results file path or dictionary
- `auto_open` (bool): Whether to open report in browser

**Example:**
```python
from src.reporting.unified_reporter import UnifiedReporter

reporter = UnifiedReporter()

# Generate from results dictionary
report_path = reporter.generate_comprehensive_report(
    results,
    auto_open=True
)

# Generate from JSON file
report_path = reporter.generate_comprehensive_report(
    "benchmark_results/results.json",
    auto_open=False
)
```

### generate_markdown_summary

Generate Markdown summary.

```python
md_path = reporter.generate_markdown_summary(
    results: Dict[str, Any]
) -> Path
```

## CLI Interface

### benchmark Command

Run benchmarks from command line.

```bash
python -m src.cli benchmark [OPTIONS]
```

**Options:**
```
--frameworks TEXT       Frameworks to test (can be specified multiple times)
--use-cases TEXT       Use cases to test (can be specified multiple times)
--test-cases INT       Number of test cases per use case (default: 5)
--parallel             Enable parallel execution
--max-workers INT      Maximum parallel workers (default: 4)
--config TEXT          Configuration file path
--output-dir TEXT      Output directory (default: benchmark_results)
--output-format TEXT   Output format: json, html, csv (default: all)
--verbose              Enable verbose logging
--debug                Enable debug mode
```

**Examples:**
```bash
# Basic benchmark
python -m src.cli benchmark \
  --frameworks langgraph crewai \
  --use-cases movie_recommendation

# Full benchmark with all options
python -m src.cli benchmark \
  --frameworks all \
  --use-cases all \
  --test-cases 10 \
  --parallel \
  --max-workers 8 \
  --config config/production.yaml \
  --output-dir results \
  --output-format json html \
  --verbose
```

### compare Command

Compare two frameworks.

```bash
python -m src.cli compare [OPTIONS]
```

**Options:**
```
--framework1 TEXT      First framework
--framework2 TEXT      Second framework
--use-case TEXT        Use case for comparison
--test-cases INT       Number of test cases (default: 10)
--runs INT            Number of runs for statistical significance (default: 3)
```

**Example:**
```bash
python -m src.cli compare \
  --framework1 langgraph \
  --framework2 crewai \
  --use-case github_triage \
  --runs 5
```

### test-framework Command

Test a single framework.

```bash
python -m src.cli test-framework [OPTIONS]
```

**Options:**
```
--framework TEXT       Framework to test
--all-use-cases       Test all use cases
--verbose             Enable verbose output
```

**Example:**
```bash
python -m src.cli test-framework \
  --framework autogen \
  --all-use-cases \
  --verbose
```

## Utility Functions

### Load Test Cases

```python
from src.data import load_test_cases

test_cases = load_test_cases(
    use_case: UseCaseType,
    num_cases: Optional[int] = None
) -> List[TestCase]
```

### Load Ground Truth

```python
from src.data import load_ground_truth

ground_truth = load_ground_truth(
    use_case: UseCaseType,
    test_case_id: str
) -> Dict[str, Any]
```

### Calculate Metrics

```python
from src.metrics import calculate_metrics

metrics = calculate_metrics(
    results: List[FrameworkResult]
) -> Dict[str, float]
```

**Returns:**
```python
{
    'success_rate': float,
    'avg_latency_seconds': float,
    'avg_tokens': float,
    'total_cost_usd': float,
    'avg_accuracy': float
}
```

## Error Handling

### Custom Exceptions

```python
from src.core.exceptions import (
    FrameworkError,
    ConfigurationError,
    EvaluationError,
    DataLoadError
)

try:
    result = adapter.run(use_case, input_data)
except FrameworkError as e:
    print(f"Framework error: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

## Environment Variables

### Required

- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key (optional)

### Optional

- `DEFAULT_MODEL`: Default model (default: gpt-4o-mini)
- `DEFAULT_TEMPERATURE`: Default temperature (default: 0.7)
- `MAX_TOKENS`: Maximum tokens (default: 2000)
- `LOG_LEVEL`: Logging level (default: INFO)
- `DEBUG`: Enable debug mode (default: false)
- `CACHE_ENABLED`: Enable caching (default: true)
- `CACHE_TTL`: Cache TTL in seconds (default: 3600)

## Examples

### Complete Benchmark Example

```python
from run_evaluation import UnifiedBenchmarkRunner
from src.core.config import Config

# Configure
config = Config()
config.default_model = 'gpt-4'
config.temperature = 0.7
config.max_tokens = 2048

# Run benchmark
runner = UnifiedBenchmarkRunner(
    mode='live',
    config=config,
    samples=100
)
results = runner.run()

# Results are automatically saved and report generated
print("Evaluation complete!")
```

### Custom Use Case Example

```python
from src.core.types import UseCaseType
from src.adapters import create_adapter
from src.metrics.evaluator import Evaluator

# Define custom use case
custom_input = {
    "query": "What is quantum computing?",
    "max_length": 200
}

# Run with framework
adapter = create_adapter(FrameworkType.LANGGRAPH)
result = adapter.run(
    UseCaseType.RESEARCH_SUMMARY,
    custom_input
)

# Evaluate
evaluator = Evaluator()
scores = evaluator.evaluate(
    result.result,
    {"expected_topics": ["quantum", "computing", "qubits"]},
    UseCaseType.RESEARCH_SUMMARY
)

print(f"Accuracy: {scores['accuracy']:.2%}")
print(f"Latency: {result.latency_seconds:.2f}s")
print(f"Cost: ${result.cost_usd:.4f}")
```

## Next Steps

- [Contributing Guide](./contributing.md)
- [Best Practices](./best-practices.md)
- [Troubleshooting](./troubleshooting.md)
