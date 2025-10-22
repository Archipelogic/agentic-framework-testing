# Configuration Guide

## Overview

The Agentic AI Framework Testing Harness supports multiple configuration methods to accommodate different use cases and deployment scenarios.

## Configuration Hierarchy

Configuration is loaded in the following order (later overrides earlier):

1. Default configuration (built-in)
2. Configuration files (`config.yaml`, `config.json`)
3. Environment variables
4. Command-line arguments

## API Configuration

### Required API Keys

At least one LLM provider API key is required:

```bash
# Option 1: OpenAI
OPENAI_API_KEY=sk-...
OPENAI_ORG_ID=org-...  # Optional

# Option 2: Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Option 3: Google (Gemini)
GOOGLE_API_KEY=...

# Option 4: Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_DEPLOYMENT=...

# Option 5: AWS Bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
AWS_BEDROCK_ENABLED=true  # Auto-enabled when AWS credentials present
```

### AWS Configuration

The harness fully supports AWS services including Bedrock and Ground Truth:

```bash
# AWS Bedrock Configuration
AWS_BEDROCK_ENABLED=true  # Enable Bedrock framework testing
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1  # Default region
AWS_SESSION_TOKEN=...  # Optional for temporary credentials

# AWS Ground Truth Configuration (Optional)
AWS_GROUND_TRUTH_ENABLED=false
GROUND_TRUTH_ROLE_ARN=arn:aws:iam::...
GROUND_TRUTH_S3_INPUT_BUCKET=your-input-bucket
GROUND_TRUTH_S3_OUTPUT_BUCKET=your-output-bucket
```

### Model Configuration

```bash
# Model selection
DEFAULT_MODEL=gpt-4o-mini  # Options: gpt-4o, gpt-4o-mini, claude-3-opus, claude-3-haiku, gemini-pro

# Model parameters
DEFAULT_TEMPERATURE=0.7  # 0.0-2.0, lower = more deterministic
MAX_TOKENS=2000  # Maximum tokens per response
TOP_P=1.0  # Nucleus sampling parameter
FREQUENCY_PENALTY=0.0  # Reduce repetition
PRESENCE_PENALTY=0.0  # Encourage new topics
```

### Cost and Rate Limiting

```bash
# Cost tracking
ENABLE_COST_TRACKING=true
MAX_COST_PER_RUN=10.0  # Maximum USD per benchmark run
WARN_COST_THRESHOLD=5.0  # Warning threshold

# Rate limiting
RATE_LIMIT_DELAY=1.0  # Seconds between API calls
MAX_RETRIES=3  # Number of retries on failure
RETRY_DELAY=5.0  # Seconds between retries
REQUEST_TIMEOUT=60  # Request timeout in seconds
```

## Framework Configuration

### Individual Framework Settings

Create `config/frameworks.yaml`:

```yaml
frameworks:
  langgraph:
    enabled: true
    version: "latest"
    config:
      max_iterations: 10
      enable_tracing: true
      memory_type: "conversation"
  
  crewai:
    enabled: true
    version: "0.80.0"
    config:
      max_rpm: 10
      max_execution_time: 300
      allow_delegation: true
  
  autogen:
    enabled: true
    config:
      max_rounds: 10
      human_input_mode: "NEVER"
      code_execution: false
  
  pydantic_ai:
    enabled: true
    config:
      strict_mode: true
      validate_outputs: true
```

### Framework-Specific Environment Variables

```bash
# LangGraph
LANGGRAPH_TRACING_ENABLED=true
LANGGRAPH_API_KEY=...

# CrewAI
CREWAI_TELEMETRY_ENABLED=false
CREWAI_MAX_RPM=10

# AutoGen
AUTOGEN_USE_DOCKER=false
AUTOGEN_TIMEOUT=300
```

## Use Case Configuration

### Defining Test Cases

Create `config/use_cases.yaml`:

```yaml
use_cases:
  movie_recommendation:
    enabled: true
    test_cases: 10
    timeout: 30
    validation_mode: "strict"
    
  github_triage:
    enabled: true
    test_cases: 15
    timeout: 45
    include_labels: ["bug", "feature", "enhancement"]
    
  recipe_generation:
    enabled: true
    test_cases: 10
    timeout: 40
    dietary_restrictions: ["vegetarian", "vegan", "gluten-free"]
```

### Custom Test Data

Add custom test cases in `data/<use_case>/custom_test_cases.json`:

```json
{
  "test_cases": [
    {
      "id": "custom_001",
      "input": {
        "preferences": ["action", "sci-fi"],
        "year_range": [2020, 2024]
      },
      "expected_output": {
        "movie_count": 5,
        "required_genres": ["action", "sci-fi"]
      }
    }
  ]
}
```

## Execution Configuration

### Benchmark Settings

```yaml
# config/benchmark.yaml
benchmark:
  # Execution mode
  mode: "production"  # Options: demo, development, production
  
  # Parallel execution
  parallel_execution: true
  max_workers: 4
  
  # Timeouts and limits
  global_timeout: 3600  # 1 hour
  per_test_timeout: 60
  max_memory_mb: 8192
  
  # Output settings
  output_format: ["json", "html", "csv"]
  output_dir: "benchmark_results"
  generate_plots: true
  
  # Logging
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_to_file: true
  log_dir: "logs"
```

### Performance Tuning

```bash
# Parallel execution
MAX_PARALLEL_WORKERS=4  # Number of concurrent workers
WORKER_TIMEOUT=300  # Worker timeout in seconds

# Memory management
MAX_MEMORY_PER_WORKER=2048  # MB per worker
ENABLE_MEMORY_PROFILING=true

# Caching
ENABLE_RESPONSE_CACHE=true
CACHE_DIR="~/.agentic_cache"
CACHE_TTL=3600  # Cache TTL in seconds
```

## Ensuring Fair Evaluation

### Model Consistency

```yaml
# config/fairness.yaml
fairness:
  # Use same model for all frameworks
  enforce_same_model: true
  model: "gpt-4o-mini"
  
  # Use same parameters
  enforce_same_params: true
  temperature: 0.7
  max_tokens: 2000
  
  # Version pinning
  pin_versions: true
  versions:
    openai: "1.3.0"
    anthropic: "0.18.0"
    langchain: "0.1.0"
```

### Package Version Management

```bash
# Freeze current versions
pip freeze > requirements.lock

# Install exact versions
pip install -r requirements.lock

# Verify versions
python3 -c "
import pkg_resources
for package in ['openai', 'anthropic', 'langchain']:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f'{package}: {version}')
    except:
        print(f'{package}: not installed')
"
```

## Advanced Configuration

### Custom Evaluators

```python
# config/custom_evaluators.py
from src.metrics.evaluator import BaseEvaluator

class CustomAccuracyEvaluator(BaseEvaluator):
    def evaluate(self, output, ground_truth):
        # Custom evaluation logic
        return score
```

### Webhook Integration

```yaml
# config/webhooks.yaml
webhooks:
  enabled: true
  endpoints:
    - url: "https://your-webhook.com/results"
      events: ["benchmark_complete", "test_failed"]
      headers:
        Authorization: "Bearer token"
```

### Database Configuration

```yaml
# config/database.yaml
database:
  enabled: true
  type: "postgresql"  # sqlite, postgresql, mongodb
  connection:
    host: "localhost"
    port: 5432
    database: "agentic_benchmarks"
    user: "benchmark_user"
    password: "${DB_PASSWORD}"  # From environment
```

## Configuration Validation

### Validate Configuration

```bash
# Validate all configuration
python3 -m src.core.config validate

# Validate specific config
python3 -m src.core.config validate --file config/production.yaml

# Check API keys
python3 -m src.core.config check-keys
```

### Configuration Best Practices

1. **Use environment variables for sensitive data**
   - Never commit API keys to version control
   - Use `.env` files for local development

2. **Version pin for reproducibility**
   - Lock package versions for production
   - Document Python version requirements

3. **Separate configs by environment**
   - `config/development.yaml`
   - `config/staging.yaml`
   - `config/production.yaml`

4. **Monitor resource usage**
   - Set appropriate timeouts
   - Limit parallel workers based on system resources

5. **Enable comprehensive logging**
   - Use DEBUG level during development
   - Archive logs for production runs

## Next Steps

- [Run your first benchmark](./quickstart.md)
- [Add custom use cases](./use-cases.md)
- [Understand evaluation metrics](./evaluation.md)
