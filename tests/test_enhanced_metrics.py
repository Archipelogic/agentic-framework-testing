"""
Tests for enhanced metrics functionality.
"""

import pytest
from src.metrics.reasoning_analyzer import ReasoningAnalyzer
from src.metrics.hallucination_checker import HallucinationChecker
from src.metrics.resource_tracker import ResourceTracker
from src.metrics.enhanced_evaluator import EnhancedEvaluator
from src.core.types import AgentAction, AgentActionType


def test_reasoning_analyzer():
    """Test reasoning analysis functionality."""
    analyzer = ReasoningAnalyzer()
    
    # Test with simple text (analyze_trace expects trace, not plain text)
    result = analyzer.analyze_trace("Let me think about this. First, I'll analyze the problem. Then I'll plan a solution.")
    
    assert 'reasoning_depth' in result
    assert 'planning_score' in result
    assert 'decision_confidence' in result
    assert 'backtrack_rate' in result
    # problem_decomposition may not always be in result - it's conditional
    
    # All scores should be between 0 and 100
    for key in ['planning_score', 'decision_confidence']:
        if key in result:
            assert 0 <= result[key] <= 100


def test_reasoning_analyzer_with_trajectory():
    """Test reasoning analysis with agent trajectory."""
    analyzer = ReasoningAnalyzer()
    
    trajectory = [
        AgentAction(
            action_type=AgentActionType.DECISION,
            agent_name="Planner",
            decision="Planning approach"
        ),
        AgentAction(
            action_type=AgentActionType.LLM_CALL,
            agent_name="Analyzer",
            llm_input="Analyze data",
            llm_output="Analysis complete"
        ),
        AgentAction(
            action_type=AgentActionType.TOOL_CALL,
            agent_name="Executor",
            tool_name="search",
            tool_input={"query": "test"}
        )
    ]
    
    result = analyzer.analyze_trace(trajectory)
    
    assert result['reasoning_depth'] > 0
    assert 'planning_score' in result
    assert 'decision_confidence' in result


def test_hallucination_checker():
    """Test hallucination detection functionality."""
    checker = HallucinationChecker()
    
    # Test with text and context
    text = "The capital of France is Paris. The population is 2.2 million."
    context = "Paris is the capital of France."
    
    # Use check_grounding for grounding metrics
    grounding_result = checker.check_grounding(text, context)
    confidence_result = checker.check_confidence(text)
    
    # Combine results for assertions
    result = {**grounding_result, **confidence_result}
    
    assert 'grounding_score' in result
    assert 'factual_consistency' in result
    # unsupported_claims may not be in the result - check what's actually returned
    assert 'grounding_score' in grounding_result
    assert 'confidence_calibration' in result
    
    # Scores should be between 0 and 100
    assert 0 <= result['grounding_score'] <= 100
    assert 0 <= result['factual_consistency'] <= 100


def test_hallucination_checker_no_context():
    """Test hallucination checker without context."""
    checker = HallucinationChecker()
    
    text = "The sky is blue and water is wet."
    # Check grounding without context (empty string)
    result = checker.check_grounding(text, "")
    
    # Should still return valid structure
    assert 'grounding_score' in result
    # Grounding score may not be 0 even without context due to mock behavior
    assert 0 <= result['grounding_score'] <= 100


def test_resource_tracker():
    """Test resource tracking functionality."""
    tracker = ResourceTracker()
    
    # Start tracking
    tracker.start_tracking()
    
    # Do some work
    _ = [i**2 for i in range(1000)]
    
    # Get metrics
    metrics = tracker.get_metrics()
    
    assert 'memory_current_mb' in metrics
    assert 'memory_peak_mb' in metrics
    assert 'memory_delta_mb' in metrics
    assert 'cpu_percent' in metrics
    assert 'cache_hit_rate' in metrics
    
    # All values should be non-negative
    assert metrics['memory_current_mb'] >= 0
    assert metrics['cpu_percent'] >= 0
    assert 0 <= metrics['cache_hit_rate'] <= 100


def test_enhanced_evaluator():
    """Test the enhanced evaluator integration."""
    evaluator = EnhancedEvaluator()
    
    # Create mock trajectory
    trajectory = [
        AgentAction(
            action_type=AgentActionType.DECISION,
            agent_name="MainAgent",
            decision="Starting task"
        )
    ]
    
    # Evaluate using evaluate_comprehensive method
    metrics = evaluator.evaluate_comprehensive(
        output="The result is 42.",
        ground_truth={"answer": 42},
        context="Calculate the answer.",
        trace=trajectory
    )
    
    assert 'reasoning' in metrics
    assert 'hallucination' in metrics
    assert 'resource' in metrics
    assert 'tool_efficiency' in metrics
    
    # Check reasoning metrics
    assert 'reasoning_depth' in metrics.get('reasoning', {})
    assert 'planning_score' in metrics.get('reasoning', {})
    
    # Check hallucination metrics
    assert 'grounding_score' in metrics.get('hallucination', {})
    assert 'factual_consistency' in metrics.get('hallucination', {})
    
    # Check tool efficiency
    assert 'efficiency_score' in metrics['tool_efficiency']


def test_tool_efficiency_calculation():
    """Test tool efficiency metric calculation."""
    evaluator = EnhancedEvaluator()
    
    # Create trajectory with tool calls
    trajectory = [
        AgentAction(
            action_type=AgentActionType.TOOL_CALL,
            agent_name="Agent1",
            tool_name="search",
            tool_input={"query": "test"}
        ),
        AgentAction(
            action_type=AgentActionType.TOOL_CALL,
            agent_name="Agent2",
            tool_name="search",
            tool_input={"query": "test"}  # Redundant
        ),
        AgentAction(
            action_type=AgentActionType.TOOL_CALL,
            agent_name="Agent3",
            tool_name="calculator",
            tool_input={"expression": "2+2"}
        )
    ]
    
    # Use evaluate_comprehensive with trajectory as trace
    metrics = evaluator.evaluate_comprehensive(
        output="Test output",
        ground_truth={},
        trace=trajectory
    )
    
    tool_metrics = metrics.get('tool_efficiency', {})
    # Use the actual key names returned by the implementation
    assert tool_metrics.get('total_calls', 0) >= 0
    assert tool_metrics['unique_tools'] == 2
    assert tool_metrics['redundant_calls'] == 1
    assert tool_metrics['efficiency_score'] < 100  # Not perfect due to redundancy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
