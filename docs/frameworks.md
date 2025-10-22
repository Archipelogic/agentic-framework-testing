# Frameworks Guide

## Supported Frameworks

The testing harness currently supports 13 agentic AI frameworks, each with unique capabilities and architectural approaches.

### Framework Comparison Matrix

| Framework | Multi-Agent | Tool Usage | RAG Support | Streaming | Async | Production Ready |
|-----------|------------|------------|-------------|-----------|-------|------------------|
| **LangGraph** | ✅ Excellent | ✅ Native | ✅ Good | ✅ Yes | ✅ Yes | ✅ Yes |
| **CrewAI** | ✅ Excellent | ✅ Native | ✅ Good | ❌ No | ✅ Yes | ✅ Yes |
| **AutoGen** | ✅ Excellent | ✅ Native | ✅ Good | ✅ Yes | ✅ Yes | ✅ Yes |
| **PydanticAI** | ⚠️ Limited | ✅ Native | ⚠️ Limited | ✅ Yes | ✅ Yes | ✅ Yes |
| **Haystack** | ⚠️ Limited | ✅ Native | ✅ Excellent | ✅ Yes | ✅ Yes | ✅ Yes |
| **LlamaIndex** | ⚠️ Limited | ✅ Native | ✅ Excellent | ✅ Yes | ✅ Yes | ✅ Yes |
| **DSPy** | ❌ No | ⚠️ Limited | ✅ Good | ❌ No | ✅ Yes | ⚠️ Beta |
| **Agno** | ✅ Good | ✅ Native | ✅ Good | ✅ Yes | ✅ Yes | ⚠️ Beta |
| **BeeAI** | ✅ Good | ✅ Native | ✅ Good | ✅ Yes | ✅ Yes | ⚠️ Beta |
| **Strands** | ✅ Good | ✅ Native | ✅ Good | ✅ Yes | ✅ Yes | ⚠️ Beta |
| **Bedrock** | ✅ Good | ✅ Native | ✅ Good | ✅ Yes | ✅ Yes | ✅ Yes |
| **Smolagents** | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | ❌ No | ❌ No | ⚠️ Alpha |
| **Atomic Agents** | ✅ Good | ✅ Native | ⚠️ Limited | ✅ Yes | ✅ Yes | ⚠️ Beta |

## Framework Details

### 1. LangGraph
**Strengths**: Excellent graph-based workflow orchestration, robust error handling, comprehensive observability, good RAG integration via LangChain

**Best For**: Complex multi-step workflows, stateful conversations, production systems, RAG-enhanced applications

**Installation**:
```bash
pip install langgraph langchain langchain-openai
```

**Configuration**:
```python
config = {
    "max_iterations": 10,
    "recursion_limit": 25,
    "enable_tracing": True,
    "checkpoint_strategy": "memory"
}
```

### 2. CrewAI
**Strengths**: Intuitive multi-agent collaboration, role-based agents, built-in tools

**Best For**: Team simulations, collaborative tasks, autonomous agent crews

**Installation**:
```bash
pip install crewai crewai-tools
```

**Configuration**:
```python
config = {
    "max_rpm": 10,
    "max_execution_time": 300,
    "allow_delegation": True,
    "verbose": True
}
```

### 3. AutoGen
**Strengths**: Microsoft-backed, code execution capabilities, human-in-the-loop support, RAG-enhanced reasoning

**Best For**: Code generation, complex reasoning, research tasks, knowledge-intensive applications

**Installation**:
```bash
pip install pyautogen
```

### 4. PydanticAI
**Strengths**: Type safety, structured outputs, Pydantic validation

**Best For**: Applications requiring strict data validation, API integrations

**RAG Limitations**: Basic RAG support, best used with external retrieval systems

**Installation**:
```bash
pip install pydantic-ai
```

### 5. Haystack
**Strengths**: Industry-leading RAG capabilities, document processing, modular pipeline architecture, semantic search

**Best For**: Question answering, document search, information retrieval, knowledge-grounded generation

**RAG Features**: 
- Native document stores (Elasticsearch, FAISS, Pinecone, Weaviate)
- Advanced retrieval strategies (BM25, Dense, Hybrid)
- Document preprocessing and chunking
- Multi-modal retrieval support

**Installation**:
```bash
pip install farm-haystack[all]
pip install farm-haystack[faiss]  # For FAISS vector store
pip install farm-haystack[elasticsearch]  # For Elasticsearch
```

### 6. LlamaIndex
**Strengths**: Excellent RAG and data ingestion capabilities, composable indices, multi-document queries, advanced retrieval strategies

**Best For**: Complex document Q&A, structured data queries, knowledge graphs, RAG-first applications

**RAG Features**:
- Multiple index types (Vector, List, Tree, Keyword, Knowledge Graph)
- Advanced query engines (SubQuestion, Router, SQL, Multi-Step)
- Document connectors (PDFs, Notion, Slack, Discord, etc.)
- Hybrid search capabilities
- Agent-based retrieval with tool use

**Installation**:
```bash
pip install llama-index
pip install llama-index-vector-stores-faiss  # For FAISS
pip install llama-index-vector-stores-pinecone  # For Pinecone
```

### 7. DSPy
**Strengths**: Programmatic prompt optimization, systematic approach to prompt engineering, good RAG integration

**Best For**: Research applications, prompt optimization, RAG-enhanced reasoning chains

**Installation**:
```bash
pip install dspy-ai
```

## Adding a New Framework

### Step 1: Create Framework Adapter

Create `src/adapters/new_framework_adapter.py`:

```python
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time

from src.core.types import (
    FrameworkType,
    UseCaseType,
    FrameworkResult,
    TestCase
)
from src.adapters.base import BaseFrameworkAdapter

@dataclass
class NewFrameworkAdapter(BaseFrameworkAdapter):
    """Adapter for NewFramework."""
    
    framework_type: FrameworkType = FrameworkType.NEW_FRAMEWORK
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize the framework."""
        try:
            import new_framework
            self.framework = new_framework
            self._initialize_framework()
        except ImportError:
            raise ImportError("NewFramework not installed. Run: pip install new-framework")
    
    def _initialize_framework(self):
        """Initialize framework-specific components."""
        # Initialize your framework
        self.client = self.framework.Client(
            api_key=self.config.get("api_key"),
            model=self.config.get("model", "gpt-4o-mini")
        )
    
    def get_framework_name(self) -> FrameworkType:
        """Get framework name."""
        return self.framework_type
    
    def get_framework_version(self) -> str:
        """Get framework version."""
        try:
            return self.framework.__version__
        except:
            return "unknown"
    
    def supports_use_case(self, use_case: UseCaseType) -> bool:
        """Check if framework supports the use case."""
        # Define supported use cases
        supported_use_cases = [
            UseCaseType.MOVIE_RECOMMENDATION,
            UseCaseType.GITHUB_TRIAGE,
            UseCaseType.RECIPE_GENERATION,
            # Add more as supported
        ]
        return use_case in supported_use_cases
    
    def run(
        self,
        use_case: UseCaseType,
        input_data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> FrameworkResult:
        """Execute the framework on a use case."""
        start_time = time.time()
        
        try:
            # Route to appropriate handler
            if use_case == UseCaseType.MOVIE_RECOMMENDATION:
                result = self._run_movie_recommendation(input_data)
            elif use_case == UseCaseType.GITHUB_TRIAGE:
                result = self._run_github_triage(input_data)
            else:
                raise ValueError(f"Unsupported use case: {use_case}")
            
            # Calculate metrics
            latency = time.time() - start_time
            cost = self._estimate_cost(input_data, result)
            
            return FrameworkResult(
                succeeded=True,
                result=result,
                latency_seconds=latency,
                tokens_used=self._count_tokens(input_data, result),
                cost_usd=cost,
                metadata={
                    "framework_version": self.get_framework_version(),
                    "use_case": use_case.value
                }
            )
            
        except Exception as e:
            return FrameworkResult(
                succeeded=False,
                result={},
                error_message=str(e),
                latency_seconds=time.time() - start_time
            )
    
    def _run_movie_recommendation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run movie recommendation use case."""
        # Implement framework-specific logic
        prompt = self._build_movie_prompt(input_data)
        response = self.client.complete(prompt)
        return self._parse_movie_response(response)
    
    def _run_github_triage(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run GitHub triage use case."""
        # Implement framework-specific logic
        pass
    
    def _estimate_cost(self, input_data: Dict[str, Any], output: Dict[str, Any]) -> float:
        """Estimate cost of the operation."""
        # Implement cost calculation
        # Example: $0.01 per 1K tokens
        tokens = self._count_tokens(input_data, output)
        return (tokens / 1000) * 0.01
    
    def _count_tokens(self, input_data: Dict[str, Any], output: Dict[str, Any]) -> int:
        """Count tokens used."""
        # Implement token counting
        # Simplified example
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        
        input_str = str(input_data)
        output_str = str(output)
        
        return len(enc.encode(input_str)) + len(enc.encode(output_str))
```

### Step 2: Register Framework Type

Add to `src/core/types.py`:

```python
class FrameworkType(str, Enum):
    """Supported framework types."""
    LANGGRAPH = "langgraph"
    CREWAI = "crewai"
    # ... existing frameworks ...
    NEW_FRAMEWORK = "new_framework"  # Add your framework
```

### Step 3: Update Factory Method

Modify `src/adapters/__init__.py`:

```python
from src.adapters.new_framework_adapter import NewFrameworkAdapter

def create_adapter(
    framework: FrameworkType,
    config: Optional[Dict[str, Any]] = None
) -> BaseFrameworkAdapter:
    """Factory method to create framework adapters."""
    
    if framework == FrameworkType.NEW_FRAMEWORK:
        return NewFrameworkAdapter(config=config or {})
    # ... rest of the factory logic ...
```

### Step 4: Add Framework Tests

Create `tests/test_new_framework.py`:

```python
import pytest
from src.core.types import FrameworkType, UseCaseType
from src.adapters import create_adapter

def test_new_framework_creation():
    """Test that NewFramework adapter can be created."""
    adapter = create_adapter(FrameworkType.NEW_FRAMEWORK)
    assert adapter is not None
    assert adapter.get_framework_name() == FrameworkType.NEW_FRAMEWORK

def test_new_framework_use_case_support():
    """Test use case support."""
    adapter = create_adapter(FrameworkType.NEW_FRAMEWORK)
    assert adapter.supports_use_case(UseCaseType.MOVIE_RECOMMENDATION)

def test_new_framework_execution():
    """Test basic execution."""
    adapter = create_adapter(FrameworkType.NEW_FRAMEWORK)
    
    test_input = {
        "preferences": ["action", "sci-fi"],
        "year_range": [2020, 2024]
    }
    
    result = adapter.run(
        UseCaseType.MOVIE_RECOMMENDATION,
        test_input
    )
    
    assert result is not None
    assert result.succeeded
    assert result.latency_seconds > 0

@pytest.mark.parametrize("use_case", list(UseCaseType))
def test_new_framework_all_use_cases(use_case):
    """Test framework with all use cases."""
    adapter = create_adapter(FrameworkType.NEW_FRAMEWORK)
    
    if adapter.supports_use_case(use_case):
        # Create appropriate test input
        test_input = create_test_input_for_use_case(use_case)
        
        result = adapter.run(use_case, test_input)
        assert result is not None
```

### Step 5: Add Configuration

Create `config/frameworks/new_framework.yaml`:

```yaml
new_framework:
  enabled: true
  version: "1.0.0"
  
  # API Configuration
  api:
    key: "${NEW_FRAMEWORK_API_KEY}"
    endpoint: "https://api.newframework.com/v1"
    timeout: 30
    max_retries: 3
  
  # Model Configuration
  model:
    name: "gpt-4o-mini"
    temperature: 0.7
    max_tokens: 2000
  
  # Framework-specific settings
  settings:
    enable_caching: true
    cache_ttl: 3600
    max_workers: 4
    batch_size: 10
  
  # Use case specific configs
  use_cases:
    movie_recommendation:
      max_recommendations: 5
      include_explanations: true
    
    github_triage:
      max_labels: 3
      auto_assign: false
```

## Framework Development Best Practices

### 1. Implement Graceful Degradation

```python
def run_with_fallback(self, use_case, input_data):
    """Run with fallback handling."""
    try:
        # Try primary method
        return self._run_primary(use_case, input_data)
    except Exception as e:
        logger.warning(f"Primary method failed: {e}")
        
        # Try fallback method
        try:
            return self._run_fallback(use_case, input_data)
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            return FrameworkResult(
                succeeded=False,
                error_message="All methods failed"
            )
```

### 2. Implement Comprehensive Logging

```python
import logging

class FrameworkAdapter:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
    
    def run(self, use_case, input_data):
        self.logger.info(f"Starting {use_case} with input: {input_data}")
        
        try:
            result = self._execute(use_case, input_data)
            self.logger.info(f"Successfully completed {use_case}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to execute {use_case}: {e}", exc_info=True)
            raise
```

### 3. Implement Resource Management

```python
class ResourceManagedAdapter:
    """Adapter with resource management."""
    
    def __init__(self):
        self.resources = []
    
    def __enter__(self):
        """Context manager entry."""
        self._acquire_resources()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._release_resources()
    
    def _acquire_resources(self):
        """Acquire necessary resources."""
        # Open connections, allocate memory, etc.
        pass
    
    def _release_resources(self):
        """Release resources."""
        # Close connections, free memory, etc.
        for resource in self.resources:
            resource.close()
```

### 4. Implement Monitoring and Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

class MonitoredAdapter:
    """Adapter with monitoring."""
    
    # Define metrics
    request_count = Counter('framework_requests_total', 'Total requests', ['framework', 'use_case'])
    request_duration = Histogram('framework_request_duration_seconds', 'Request duration')
    active_requests = Gauge('framework_active_requests', 'Active requests')
    
    def run(self, use_case, input_data):
        """Run with monitoring."""
        self.active_requests.inc()
        self.request_count.labels(
            framework=self.get_framework_name(),
            use_case=use_case.value
        ).inc()
        
        with self.request_duration.time():
            try:
                result = self._execute(use_case, input_data)
                return result
            finally:
                self.active_requests.dec()
```

## Testing Your Framework

### Integration Test Suite

```bash
# Run integration tests for your framework
pytest tests/test_new_framework.py -v

# Run with specific use case
pytest tests/test_new_framework.py::test_movie_recommendation -v

# Run with coverage
pytest tests/test_new_framework.py --cov=src.adapters.new_framework_adapter
```

### Benchmark Your Framework

```python
# benchmark_new_framework.py
from run_evaluation import UnifiedBenchmarkRunner

# Run comprehensive benchmark
runner = UnifiedBenchmarkRunner(
    mode='live',
    samples=100,  # Test with 100 samples per use case
    parallel=True
)
results = runner.run()

# Print results
print(f"Overall Success Rate: {results['summary']['overall']['success_rate']:.2%}")
print(f"Average Latency: {results['summary']['overall']['avg_latency']:.2f}s")
print(f"Total Cost: ${results['summary']['overall']['total_cost']:.4f}")
```

## Framework Optimization Tips

1. **Cache LLM responses** when possible
2. **Batch API calls** to reduce latency
3. **Use async operations** for I/O-bound tasks
4. **Implement retry logic** with exponential backoff
5. **Monitor token usage** to control costs
6. **Use streaming** for long responses
7. **Implement timeout handling** for reliability

## Next Steps

- [Configure your framework](./configuration.md)
- [Add custom use cases](./use-cases.md)
- [Understand evaluation metrics](./evaluation.md)
- [Deploy to production](./deployment.md)
