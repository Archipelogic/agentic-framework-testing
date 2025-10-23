# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### 1. ImportError: No module named 'xxx'

**Problem**: Missing dependencies or framework not installed

**Solutions**:
```bash
# Check installed packages
pip list | grep framework-name

# Reinstall specific framework
pip uninstall framework-name
pip install framework-name --upgrade

# Install all optional dependencies
pip install -r requirements.txt --upgrade

# For development mode
pip install -e . --upgrade
```

#### 2. Version Conflicts

**Problem**: Incompatible package versions

**Solutions**:
```bash
# Check for conflicts
pip check

# Create fresh environment
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Use specific versions
pip install "openai==1.3.0" "langchain==0.1.0"
```

### API Key Issues

#### 1. Authentication Failed

**Problem**: Invalid or missing API keys

**Diagnosis**:
```bash
# Check if keys are set
echo $OPENAI_API_KEY | head -c 10
echo $ANTHROPIC_API_KEY | head -c 10

# Test API connection
python3 -c "
import openai
openai.api_key = '$OPENAI_API_KEY'
print(openai.Model.list()['data'][0]['id'])
"
```

**Solutions**:
```bash
# Reload environment file
source .env

# Export directly
export OPENAI_API_KEY="sk-..."

# Verify .env file format
cat .env | grep -E "^[A-Z_]+="
```

#### 2. Rate Limiting

**Problem**: Too many requests error (429)

**Solutions**:
```python
# config/rate_limiting.yaml
rate_limiting:
  enabled: true
  requests_per_minute: 20
  retry_after_429: true
  exponential_backoff:
    initial_delay: 1
    max_delay: 60
    multiplier: 2
```

```bash
# Set environment variables
export RATE_LIMIT_DELAY=2
export MAX_RETRIES=5
export RETRY_DELAY=10
```

### Framework-Specific Issues

#### LangGraph Issues

**Problem**: Graph execution errors

```python
# Debug LangGraph execution
import logging
logging.getLogger("langgraph").setLevel(logging.DEBUG)

from langgraph.checkpoint.memory import MemorySaver

# Use memory checkpointing for debugging
memory = MemorySaver()
graph = create_graph(checkpointer=memory)

# Get execution trace
for state in graph.stream(input_data, config={"recursion_limit": 10}):
    print(f"State: {state}")
```

#### CrewAI Issues

**Problem**: Agent delegation failures

```python
# Debug CrewAI agents
from crewai import Crew, Agent, Task

# Enable verbose mode
crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    verbose=True,  # Enable detailed logging
    process="sequential"  # Use sequential for debugging
)

# Test individual agents
for agent in crew.agents:
    try:
        result = agent.execute("test task")
        print(f"{agent.role}: Success")
    except Exception as e:
        print(f"{agent.role}: Failed - {e}")
```

#### AutoGen Issues

**Problem**: Code execution disabled

```python
# Enable code execution safely
from autogen import config_list_from_json

config_list = config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4"],
        "tags": ["code_execution"]
    }
)

# Use Docker for safe execution
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config={
        "work_dir": "workspace",
        "use_docker": True,  # Safer execution
        "timeout": 60,
        "last_n_messages": 3
    }
)
```

### Performance Issues

#### 1. Slow Execution

**Problem**: Benchmarks taking too long

**Diagnosis**:
```python
# Profile execution
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run benchmark
results = runner.run_benchmark(frameworks, use_cases)

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(20)  # Top 20 time consumers
```

**Solutions**:
```bash
# Enable parallel execution
python3 -m src.cli benchmark --parallel --max-workers 8

# Reduce test cases
python3 -m src.cli benchmark --test-cases 3

# Use caching
export ENABLE_RESPONSE_CACHE=true
export CACHE_TTL=3600

# Optimize model selection
export DEFAULT_MODEL=gpt-4o-mini  # Faster than gpt-4o
```

#### 2. Memory Issues

**Problem**: Out of memory errors

**Diagnosis**:
```python
# Monitor memory usage
import psutil
import gc

process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Check for memory leaks
import tracemalloc
tracemalloc.start()

# Run benchmark
results = run_benchmark()

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

**Solutions**:
```bash
# Limit parallel workers
export MAX_PARALLEL_WORKERS=2

# Increase swap (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Clear cache between runs
python3 -c "import gc; gc.collect()"
```

### Data Issues

#### 1. Test Case Loading Failures

**Problem**: Cannot load test cases

**Solutions**:
```python
# Validate test case format
import json
import jsonschema

def validate_test_cases(file_path):
    """Validate test case JSON structure."""
    schema = {
        "type": "object",
        "properties": {
            "test_cases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "input"],
                    "properties": {
                        "id": {"type": "string"},
                        "input": {"type": "object"}
                    }
                }
            }
        },
        "required": ["test_cases"]
    }
    
    with open(file_path) as f:
        data = json.load(f)
    
    try:
        jsonschema.validate(data, schema)
        print("✓ Valid test case file")
        return True
    except jsonschema.ValidationError as e:
        print(f"✗ Invalid: {e.message}")
        return False

# Check all test case files
import glob
for file_path in glob.glob("data/*/test_cases.json"):
    print(f"Checking {file_path}")
    validate_test_cases(file_path)
```

#### 2. Ground Truth Mismatches

**Problem**: Evaluation failures due to ground truth format

**Solutions**:
```python
# Flexible ground truth matching
class FlexibleEvaluator:
    """Evaluator with flexible matching."""
    
    def evaluate(self, output, ground_truth):
        """Evaluate with fuzzy matching."""
        # Normalize formats
        output_normalized = self.normalize_output(output)
        truth_normalized = self.normalize_output(ground_truth)
        
        # Use multiple matching strategies
        exact_match = output_normalized == truth_normalized
        contains_match = self.check_contains(output_normalized, truth_normalized)
        semantic_match = self.check_semantic_similarity(output, ground_truth)
        
        # Weighted scoring
        score = (
            exact_match * 0.5 +
            contains_match * 0.3 +
            semantic_match * 0.2
        )
        
        return score
    
    def normalize_output(self, data):
        """Normalize output format."""
        if isinstance(data, str):
            return data.lower().strip()
        elif isinstance(data, dict):
            return {k: self.normalize_output(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.normalize_output(item) for item in data]
        return data
```

### Debugging Techniques

#### 1. Enable Debug Logging

```python
# Enable comprehensive debug logging
import logging
import sys

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Enable specific loggers
logging.getLogger('openai').setLevel(logging.DEBUG)
logging.getLogger('langchain').setLevel(logging.DEBUG)
logging.getLogger('src.benchmark').setLevel(logging.DEBUG)
```

#### 2. Interactive Debugging

```python
# Use IPython for interactive debugging
import IPython

def debug_checkpoint(framework, test_case):
    """Interactive debugging checkpoint."""
    print(f"Debugging {framework} with {test_case}")
    
    # Drop into interactive shell
    IPython.embed()
    
    # Continue execution after debugging
    return continue_execution()

# Or use pdb
import pdb

def run_with_debugger(framework, test_case):
    """Run with Python debugger."""
    try:
        result = framework.run(test_case)
        return result
    except Exception as e:
        print(f"Error: {e}")
        pdb.set_trace()  # Drop into debugger
```

#### 3. Test in Isolation

```python
# test_single_case.py
from src.adapters import create_adapter
from src.core.types import FrameworkType, UseCaseType

def test_single_case():
    """Test a single case in isolation."""
    # Create adapter
    adapter = create_adapter(FrameworkType.LANGGRAPH)
    
    # Simple test input
    test_input = {
        "preferences": ["action"],
        "year_range": [2020, 2024]
    }
    
    # Run with detailed logging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    result = adapter.run(
        UseCaseType.MOVIE_RECOMMENDATION,
        test_input
    )
    
    print(f"Result: {result}")
    print(f"Success: {result.succeeded}")
    print(f"Output: {result.result}")
    
    if not result.succeeded:
        print(f"Error: {result.error_message}")

if __name__ == "__main__":
    test_single_case()
```

### Error Recovery

#### Checkpoint and Resume

```python
# src/recovery/checkpoint_manager.py
import pickle
import os

class CheckpointManager:
    """Manage benchmark checkpoints for recovery."""
    
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, state, checkpoint_id):
        """Save benchmark state."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"{checkpoint_id}.pkl"
        )
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Checkpoint saved: {checkpoint_id}")
    
    def load_checkpoint(self, checkpoint_id):
        """Load benchmark state."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"{checkpoint_id}.pkl"
        )
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
        
        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)
        
        print(f"Checkpoint loaded: {checkpoint_id}")
        return state
    
    def resume_benchmark(self, checkpoint_id):
        """Resume from checkpoint."""
        state = self.load_checkpoint(checkpoint_id)
        
        # Resume execution
        completed_tasks = state.get('completed_tasks', [])
        pending_tasks = state.get('pending_tasks', [])
        
        print(f"Resuming with {len(pending_tasks)} pending tasks")
        
        # Continue from where it left off
        results = state.get('results', {})
        
        for task in pending_tasks:
            try:
                result = execute_task(task)
                results[task['id']] = result
                self.save_checkpoint({
                    'completed_tasks': completed_tasks + [task],
                    'pending_tasks': pending_tasks[1:],
                    'results': results
                }, checkpoint_id)
            except Exception as e:
                print(f"Task failed: {e}")
                # Save checkpoint and retry later
                break
        
        return results
```

### Getting Help

#### Diagnostic Information

```bash
# Collect diagnostic information
python3 -c "
import sys
import platform
import pkg_resources

print('=== System Information ===')
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.machine()}')

print('\n=== Package Versions ===')
packages = ['openai', 'anthropic', 'langchain', 'langgraph', 'crewai']
for package in packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f'{package}: {version}')
    except:
        print(f'{package}: not installed')

print('\n=== Environment Variables ===')
import os
env_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEBUG', 'LOG_LEVEL']
for var in env_vars:
    value = os.environ.get(var, 'not set')
    if 'KEY' in var and value != 'not set':
        value = value[:10] + '...'
    print(f'{var}: {value}')
"
```

#### Support Resources

1. **Bug Reports**: Submit with diagnostic information
2. **Documentation**: Check docs folder for guides
3. **Logs**: Always include relevant log snippets
4. **Minimal Reproduction**: Provide code to reproduce issue

## Next Steps

- [Review best practices](./best-practices.md)
- [Check API reference](./api-reference.md)
- [Configure system](./configuration.md)
