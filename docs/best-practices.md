# Best Practices Guide

## Fair Evaluation Principles

### 1. Ensure Model Consistency

**Use the Same Model Across All Frameworks**

```python
# config/fairness.yaml
evaluation:
  model: "gpt-4o-mini"  # Same model for all frameworks
  temperature: 0.7       # Consistent temperature
  max_tokens: 2000      # Same token limit
  seed: 42              # For reproducibility
```

**Verify Model Configuration**

```python
def verify_model_consistency(framework_configs):
    """Ensure all frameworks use the same model."""
    models = set()
    temperatures = set()
    
    for config in framework_configs:
        models.add(config.get('model'))
        temperatures.add(config.get('temperature'))
    
    if len(models) > 1:
        raise ValueError(f"Model inconsistency detected: {models}")
    
    if len(temperatures) > 1:
        raise ValueError(f"Temperature inconsistency detected: {temperatures}")
    
    print("✓ Model configuration is consistent")
    return True
```

### 2. Version Pinning

**Lock Package Versions for Reproducibility**

```bash
# Generate locked requirements
pip freeze > requirements.lock

# Install exact versions
pip install -r requirements.lock --no-deps

# Verify versions
python3 -c "
import pkg_resources
required = open('requirements.lock').read().splitlines()
for req in required:
    try:
        pkg_resources.require(req)
        print(f'✓ {req}')
    except:
        print(f'✗ {req} - version mismatch')
"
```

**Docker for Complete Reproducibility**

```dockerfile
# Dockerfile.reproducible
FROM python:3.10.12-slim

# Lock system packages
RUN apt-get update && apt-get install -y \
    gcc=4:12.2.0-3 \
    git=1:2.39.2-1.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy locked requirements
COPY requirements.lock /app/

# Install exact versions
RUN pip install --no-cache-dir -r /app/requirements.lock

# Set reproducibility environment
ENV PYTHONHASHSEED=42
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### 3. Statistical Rigor

**Multiple Runs for Statistical Significance**

```python
import numpy as np
from scipy import stats

def run_statistical_benchmark(framework, use_case, n_runs=30):
    """Run multiple times for statistical significance."""
    results = []
    
    for run in range(n_runs):
        # Set different seed for each run
        np.random.seed(42 + run)
        
        result = framework.run(use_case)
        results.append(result.accuracy)
    
    # Calculate statistics
    mean = np.mean(results)
    std = np.std(results)
    ci_95 = stats.t.interval(0.95, len(results)-1, mean, std/np.sqrt(len(results)))
    
    return {
        'mean': mean,
        'std': std,
        'confidence_interval_95': ci_95,
        'n_runs': n_runs,
        'raw_results': results
    }
```

**Compare Frameworks Statistically**

```python
def compare_frameworks_statistically(results_a, results_b):
    """Statistical comparison of two frameworks."""
    from scipy import stats
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(results_a, results_b)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(results_a) + np.var(results_b)) / 2)
    effect_size = (np.mean(results_a) - np.mean(results_b)) / pooled_std
    
    # Perform Mann-Whitney U test (non-parametric alternative)
    u_stat, u_pvalue = stats.mannwhitneyu(results_a, results_b)
    
    return {
        't_test': {
            'statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        },
        'effect_size': {
            'cohens_d': effect_size,
            'interpretation': interpret_effect_size(effect_size)
        },
        'mann_whitney': {
            'statistic': u_stat,
            'p_value': u_pvalue
        }
    }
```

## Performance Optimization

### 1. Efficient Resource Usage

**Batch Processing**

```python
def batch_process_tests(test_cases, batch_size=10):
    """Process test cases in batches for efficiency."""
    results = []
    
    for i in range(0, len(test_cases), batch_size):
        batch = test_cases[i:i + batch_size]
        
        # Process batch
        batch_results = process_batch(batch)
        results.extend(batch_results)
        
        # Garbage collection between batches
        import gc
        gc.collect()
    
    return results
```

**Connection Pooling**

```python
from concurrent.futures import ThreadPoolExecutor
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

class ConnectionPool:
    """Manage API connections efficiently."""
    
    def __init__(self, max_connections=10):
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=max_connections,
            pool_maxsize=max_connections
        )
        
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    def make_request(self, url, **kwargs):
        """Make request using connection pool."""
        return self.session.request(url, **kwargs)
```

### 2. Caching Strategies

**Response Caching**

```python
import hashlib
import json
import os
from functools import lru_cache, wraps

class ResponseCache:
    """Cache API responses to reduce costs and latency."""
    
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, *args, **kwargs):
        """Generate cache key from arguments."""
        key_data = json.dumps({
            'args': args,
            'kwargs': kwargs
        }, sort_keys=True)
        
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def cache_result(self, func):
        """Decorator to cache function results."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self._get_cache_key(*args, **kwargs)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            
            # Check cache
            if os.path.exists(cache_file):
                with open(cache_file) as f:
                    return json.load(f)
            
            # Execute and cache
            result = func(*args, **kwargs)
            
            with open(cache_file, 'w') as f:
                json.dump(result, f)
            
            return result
        
        return wrapper
```

**Memory Caching**

```python
from functools import lru_cache
import time

class TimedLRUCache:
    """LRU cache with time-based expiration."""
    
    def __init__(self, maxsize=128, ttl=3600):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key):
        """Get value from cache if not expired."""
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key, value):
        """Set value in cache."""
        self.cache[key] = value
        self.timestamps[key] = time.time()
        
        # Implement LRU eviction if needed
        if len(self.cache) > self.maxsize:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
```

## Code Organization

### 1. Project Structure

```
agentic-framework-testing/
├── src/
│   ├── core/           # Core types and configurations
│   │   ├── __init__.py
│   │   ├── types.py    # Type definitions
│   │   └── config.py   # Configuration management
│   ├── adapters/       # Framework adapters
│   │   ├── __init__.py
│   │   ├── base.py     # Base adapter class
│   │   └── *.py        # Individual framework adapters
│   ├── use_cases/      # Use case implementations
│   │   ├── __init__.py
│   │   └── *.py        # Individual use cases
│   ├── metrics/        # Evaluation metrics
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── benchmark/      # Benchmark runner
│   │   ├── __init__.py
│   │   └── runner.py
│   └── reporting/      # Report generation
│       ├── __init__.py
│       └── reporter.py
├── data/              # Test data
│   └── <use_case>/
│       ├── test_cases.json
│       └── ground_truth.json
├── tests/             # Test suite
│   ├── unit/
│   └── integration/
├── config/            # Configuration files
├── docs/             # Documentation
└── scripts/          # Utility scripts
```

### 2. Code Style Guidelines

**Follow PEP 8 and Type Hints**

```python
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    
    frameworks: List[str]
    use_cases: List[str]
    test_cases_per_use_case: int = 10
    parallel: bool = False
    max_workers: Optional[int] = None
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.frameworks:
            raise ValueError("At least one framework required")
        
        if not self.use_cases:
            raise ValueError("At least one use case required")
        
        if self.test_cases_per_use_case < 1:
            raise ValueError("Test cases must be >= 1")
        
        return True
```

**Use Descriptive Names**

```python
# Good
def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts."""
    pass

# Bad
def calc_sim(t1, t2):
    pass
```

### 3. Error Handling

**Comprehensive Error Handling**

```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class FrameworkError(Exception):
    """Base exception for framework errors."""
    pass

class APIError(FrameworkError):
    """API-related errors."""
    pass

class ValidationError(FrameworkError):
    """Validation errors."""
    pass

def safe_framework_execution(
    framework,
    use_case,
    input_data,
    max_retries: int = 3
) -> Optional[Dict]:
    """Execute framework with comprehensive error handling."""
    
    for attempt in range(max_retries):
        try:
            # Validate input
            if not validate_input(input_data):
                raise ValidationError("Invalid input data")
            
            # Execute framework
            result = framework.run(use_case, input_data)
            
            # Validate output
            if not validate_output(result):
                raise ValidationError("Invalid output format")
            
            return result
            
        except APIError as e:
            logger.warning(f"API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Max retries reached for API error: {e}")
                raise
                
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            raise  # Don't retry validation errors
            
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            if attempt == max_retries - 1:
                raise
    
    return None
```

## Testing Strategies

### 1. Unit Testing

```python
# tests/unit/test_evaluator.py
import pytest
from src.metrics.evaluator import Evaluator

class TestEvaluator:
    """Test evaluator functionality."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return Evaluator()
    
    def test_accuracy_calculation(self, evaluator):
        """Test accuracy calculation."""
        output = {"result": "test"}
        ground_truth = {"result": "test"}
        
        score = evaluator.calculate_accuracy(output, ground_truth)
        assert score == 1.0
    
    @pytest.mark.parametrize("output,expected,score", [
        ({"a": 1}, {"a": 1}, 1.0),
        ({"a": 1}, {"a": 2}, 0.0),
        ({"a": 1, "b": 2}, {"a": 1}, 0.5),
    ])
    def test_various_outputs(self, evaluator, output, expected, score):
        """Test with various outputs."""
        result = evaluator.calculate_accuracy(output, expected)
        assert result == score
```

### 2. Integration Testing

```python
# tests/integration/test_end_to_end.py
import pytest
from run_evaluation import UnifiedBenchmarkRunner
from src.core.types import FrameworkType, UseCaseType

@pytest.mark.integration
class TestBenchmarkIntegration:
    """Integration tests for benchmark system."""
    
    def test_full_benchmark_cycle(self):
        """Test complete benchmark cycle."""
        runner = UnifiedBenchmarkRunner(mode='mock', quick=True)
        
        results = runner.run()
        
        assert results is not None
        assert 'benchmark_id' in results
        assert 'results' in results
        assert 'summary' in results
    
    @pytest.mark.slow
    def test_parallel_execution(self):
        """Test parallel execution."""
        runner = UnifiedBenchmarkRunner(mode='mock', parallel=True)
        
        results = runner.run()
        
        assert len(results['results']) == 2
```

### 3. Performance Testing

```python
# tests/performance/test_benchmark_performance.py
import time
import pytest

@pytest.mark.performance
def test_benchmark_latency():
    """Test benchmark execution latency."""
    start = time.time()
    
    # Run minimal benchmark
    runner = UnifiedBenchmarkRunner(mode='mock', quick=True)
    results = runner.run()
    
    elapsed = time.time() - start
    
    # Should complete within reasonable time
    assert elapsed < 60  # Less than 1 minute for minimal test
    
    # Check that latency is recorded
    assert results['results']['langgraph']['movie_recommendation']['aggregate_metrics']['avg_latency_seconds'] > 0
```

## Security Best Practices

### 1. API Key Management

```python
import os
from cryptography.fernet import Fernet

class SecureConfig:
    """Secure configuration management."""
    
    def __init__(self, key_file=".secret_key"):
        self.key = self._load_or_generate_key(key_file)
        self.cipher = Fernet(self.key)
    
    def _load_or_generate_key(self, key_file):
        """Load or generate encryption key."""
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Restrict permissions
            return key
    
    def encrypt_api_key(self, api_key: str) -> bytes:
        """Encrypt API key."""
        return self.cipher.encrypt(api_key.encode())
    
    def decrypt_api_key(self, encrypted_key: bytes) -> str:
        """Decrypt API key."""
        return self.cipher.decrypt(encrypted_key).decode()
    
    def get_api_key(self, key_name: str) -> str:
        """Get API key from environment or encrypted storage."""
        # First try environment variable
        key = os.environ.get(key_name)
        if key:
            return key
        
        # Then try encrypted storage
        encrypted_file = f".keys/{key_name}.enc"
        if os.path.exists(encrypted_file):
            with open(encrypted_file, 'rb') as f:
                return self.decrypt_api_key(f.read())
        
        raise ValueError(f"API key {key_name} not found")
```

### 2. Input Validation

```python
from typing import Any, Dict
import re

class InputValidator:
    """Validate and sanitize inputs."""
    
    def validate_input(self, input_data: Dict[str, Any], schema: Dict) -> bool:
        """Validate input against schema."""
        # Check required fields
        for field in schema.get('required', []):
            if field not in input_data:
                raise ValidationError(f"Missing required field: {field}")
        
        # Validate field types
        for field, value in input_data.items():
            if field in schema.get('properties', {}):
                expected_type = schema['properties'][field]['type']
                if not self._check_type(value, expected_type):
                    raise ValidationError(f"Invalid type for {field}")
        
        # Sanitize strings
        for field, value in input_data.items():
            if isinstance(value, str):
                input_data[field] = self._sanitize_string(value)
        
        return True
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string input."""
        # Remove potential injection attacks
        value = re.sub(r'[<>\"\'`;]', '', value)
        
        # Limit length
        max_length = 10000
        if len(value) > max_length:
            value = value[:max_length]
        
        return value.strip()
```

## Monitoring and Observability

### 1. Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge
import time

class MetricsCollector:
    """Collect and expose metrics."""
    
    # Define metrics
    benchmark_counter = Counter(
        'benchmark_runs_total',
        'Total number of benchmark runs',
        ['framework', 'use_case', 'status']
    )
    
    benchmark_duration = Histogram(
        'benchmark_duration_seconds',
        'Benchmark execution duration',
        ['framework', 'use_case']
    )
    
    error_counter = Counter(
        'benchmark_errors_total',
        'Total number of errors',
        ['framework', 'error_type']
    )
    
    def record_benchmark(self, framework, use_case, duration, success):
        """Record benchmark metrics."""
        status = 'success' if success else 'failure'
        
        self.benchmark_counter.labels(
            framework=framework,
            use_case=use_case,
            status=status
        ).inc()
        
        self.benchmark_duration.labels(
            framework=framework,
            use_case=use_case
        ).observe(duration)
    
    def record_error(self, framework, error_type):
        """Record error metrics."""
        self.error_counter.labels(
            framework=framework,
            error_type=error_type
        ).inc()
```

### 2. Logging Best Practices

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """Structured logging for better observability."""
    
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        handler.setFormatter(self.JsonFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    class JsonFormatter(logging.Formatter):
        """Format logs as JSON."""
        
        def format(self, record):
            log_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            if hasattr(record, 'extra'):
                log_data.update(record.extra)
            
            if record.exc_info:
                log_data['exception'] = self.formatException(record.exc_info)
            
            return json.dumps(log_data)
    
    def log_benchmark_start(self, framework, use_case, test_cases):
        """Log benchmark start."""
        self.logger.info(
            "Benchmark started",
            extra={
                'framework': framework,
                'use_case': use_case,
                'test_cases': test_cases,
                'event': 'benchmark_start'
            }
        )
```

## Next Steps

- [API Reference](./api-reference.md)
- [Contributing Guide](./contributing.md)
- [Troubleshooting](./troubleshooting.md)
