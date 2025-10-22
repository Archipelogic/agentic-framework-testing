"""
Integration tests for the agentic framework testing harness.
"""

import pytest
from src.core.types import FrameworkType, UseCaseType, TestCase
from src.core.config import Config
from src.adapters import create_adapter
from run_evaluation import UnifiedBenchmarkRunner
from src.metrics.evaluator import Evaluator


def test_config_initialization():
    """Test that configuration loads properly."""
    config = Config()
    assert config is not None
    assert config.default_model is not None


def test_adapter_creation():
    """Test that framework adapters can be created."""
    # Test LangGraph adapter
    langgraph = create_adapter(FrameworkType.LANGGRAPH)
    assert langgraph is not None
    assert langgraph.get_framework_name() == FrameworkType.LANGGRAPH
    
    # Test CrewAI adapter
    crewai = create_adapter(FrameworkType.CREWAI)
    assert crewai is not None
    assert crewai.get_framework_name() == FrameworkType.CREWAI


def test_use_case_support():
    """Test that adapters correctly report use case support."""
    adapter = create_adapter(FrameworkType.LANGGRAPH)
    
    # LangGraph should support all use cases
    assert adapter.supports_use_case(UseCaseType.MOVIE_RECOMMENDATION)
    assert adapter.supports_use_case(UseCaseType.GITHUB_TRIAGE)
    assert adapter.supports_use_case(UseCaseType.RECIPE_GENERATION)
    assert adapter.supports_use_case(UseCaseType.RESEARCH_SUMMARY)
    assert adapter.supports_use_case(UseCaseType.EMAIL_AUTOMATION)


def test_simple_execution():
    """Test basic framework execution."""
    adapter = create_adapter(FrameworkType.LANGGRAPH)
    
    # Create a simple test case
    test_case = TestCase(
        id="test_001",
        use_case=UseCaseType.GITHUB_TRIAGE,
        input_data={
            "issue_id": 123,
            "title": "Test Issue",
            "body": "This is a test issue",
            "author": "test_user"
        },
        ground_truth={}
    )
    
    # Run adapter
    result = adapter.run(UseCaseType.GITHUB_TRIAGE, test_case.input_data)
    
    # Check result structure
    assert result is not None
    assert result.succeeded
    assert result.result is not None
    assert result.latency_seconds > 0


def test_benchmark_runner_initialization():
    """Test that benchmark runner initializes properly."""
    runner = UnifiedBenchmarkRunner(mode='mock')
    assert runner is not None
    assert runner.evaluator is not None
    assert runner.enhanced_evaluator is not None


def test_evaluator_initialization():
    """Test that evaluator initializes properly."""
    evaluator = Evaluator()
    assert evaluator is not None


def test_all_framework_adapters():
    """Test that all framework adapters can be created."""
    frameworks = [
        FrameworkType.LANGGRAPH,
        FrameworkType.CREWAI,
        FrameworkType.AUTOGEN,
        FrameworkType.PYDANTIC_AI,
        FrameworkType.HAYSTACK,
        FrameworkType.LLAMAINDEX,
        FrameworkType.DSPY,
        FrameworkType.AGNO,
        FrameworkType.BEEAI,
        FrameworkType.STRANDS_AGENTS,
        FrameworkType.BEDROCK_AGENTCORE,
        FrameworkType.SMOLAGENTS,
        FrameworkType.ATOMIC_AGENTS
    ]
    
    # Create a basic config for adapters
    config = {'model': 'gpt-4', 'aws_region': 'us-east-1'}
    
    for framework in frameworks:
        adapter = create_adapter(framework, config)
        assert adapter is not None
        assert adapter.get_framework_name() == framework
        assert adapter.get_framework_version() is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
