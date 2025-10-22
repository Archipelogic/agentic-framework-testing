"""
Tests for mock adapters functionality.
"""

import pytest
from src.adapters.mock_adapters import create_all_mock_adapters
from src.core.types import UseCaseType


def test_create_all_mock_adapters():
    """Test that all mock adapters can be created."""
    adapters = create_all_mock_adapters()
    
    # Should create 13 adapters
    assert len(adapters) == 13
    
    # Check specific frameworks exist
    expected_frameworks = [
        'langgraph', 'crewai', 'autogen', 'pydantic_ai',
        'haystack', 'llamaindex', 'dspy', 'agno', 'beeai',
        'strands_agents', 'bedrock_agentcore', 'smolagents',
        'atomic_agents'
    ]
    
    for framework in expected_frameworks:
        assert framework in adapters
        assert adapters[framework] is not None


def test_mock_adapter_execution():
    """Test that mock adapters can execute use cases."""
    adapters = create_all_mock_adapters()
    
    # Test LangGraph on movie recommendation
    langgraph = adapters['langgraph']
    result = langgraph.run(
        UseCaseType.MOVIE_RECOMMENDATION,
        {"user_id": 123, "preferences": ["sci-fi", "action"]}
    )
    
    assert result is not None
    assert hasattr(result, 'succeeded')
    assert hasattr(result, 'latency_seconds')
    assert hasattr(result, 'tokens')
    assert hasattr(result, 'cost')
    assert hasattr(result, 'trajectory')
    
    # Verify trajectory has proper structure
    trajectory = result.trajectory
    assert len(trajectory) > 0
    assert all(hasattr(action, 'action_type') for action in trajectory)


def test_mock_adapter_variability():
    """Test that mock adapters return variable results."""
    adapters = create_all_mock_adapters()
    
    # Run same test multiple times
    results = []
    for _ in range(5):
        result = adapters['crewai'].run(
            UseCaseType.GITHUB_TRIAGE,
            {"issue_id": 123, "title": "Test", "body": "Test issue"}
        )
        results.append(result)
    
    # Check for some variability in latency
    latencies = [r.latency_seconds for r in results]
    assert len(set(latencies)) > 1  # Should have some variation


def test_mock_adapter_failure_simulation():
    """Test that some frameworks simulate failures."""
    adapters = create_all_mock_adapters()
    
    # DSPy and SmolAgents should have lower success rates
    failure_prone = ['dspy', 'smolagents']
    
    for framework in failure_prone:
        success_count = 0
        for _ in range(10):
            result = adapters[framework].run(
                UseCaseType.GITHUB_TRIAGE,  # Use a supported use case
                {"issue_id": 123, "title": "Test", "body": "Test issue"}
            )
            if result.succeeded:
                success_count += 1
        
        # Should have some failures (but not all)
        assert 0 < success_count < 10


def test_mock_adapter_use_case_coverage():
    """Test that mock adapters handle all use cases."""
    adapters = create_all_mock_adapters()
    
    use_cases = [
        UseCaseType.MOVIE_RECOMMENDATION,
        UseCaseType.GITHUB_TRIAGE,
        UseCaseType.RECIPE_GENERATION,
        UseCaseType.RESEARCH_SUMMARY,
        UseCaseType.EMAIL_AUTOMATION
    ]
    
    # Test a sample of frameworks with all use cases
    test_frameworks = ['langgraph', 'crewai', 'autogen']
    
    for framework_name in test_frameworks:
        adapter = adapters[framework_name]
        for use_case in use_cases:
            result = adapter.run(use_case, {"test": "data"})
            assert result is not None
            assert hasattr(result, 'succeeded')
            assert hasattr(result, 'latency_seconds')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
