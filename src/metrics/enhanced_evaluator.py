"""
Enhanced evaluator that combines all evaluation metrics.
Integrates resource tracking, hallucination detection, and reasoning analysis.
"""

from typing import Dict, Any, Optional
from src.metrics.resource_tracker import ResourceTracker
from src.metrics.hallucination_checker import HallucinationChecker
from src.metrics.reasoning_analyzer import ReasoningAnalyzer


class EnhancedEvaluator:
    """Enhanced evaluator with comprehensive metrics."""
    
    def __init__(self):
        """Initialize enhanced evaluator with all components."""
        self.resource_tracker = ResourceTracker()
        self.hallucination_checker = HallucinationChecker()
        self.reasoning_analyzer = ReasoningAnalyzer()
    
    def evaluate_comprehensive(
        self,
        output: Any,
        ground_truth: Any,
        context: str = "",
        trace: Any = None,
        test_input: Any = None
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation with all metrics."""
        
        # Convert output to string for analysis
        output_str = str(output) if output else ""
        context_str = str(context) if context else str(test_input) if test_input else ""
        
        # Start resource tracking
        self.resource_tracker.start_tracking()
        
        # Basic accuracy (simplified)
        basic_metrics = self._evaluate_basic_accuracy(output_str, ground_truth)
        
        # Enhanced metrics
        resource_metrics = self.resource_tracker.get_metrics()
        hallucination_metrics = self.hallucination_checker.check_grounding(output_str, context_str)
        confidence_metrics = self.hallucination_checker.check_confidence(output_str)
        reasoning_metrics = self.reasoning_analyzer.analyze_trace(trace if trace else output_str)
        
        # Tool efficiency metrics
        tool_metrics = self._evaluate_tool_efficiency(trace)
        
        # Combine all metrics
        return {
            'basic': basic_metrics,
            'resource': resource_metrics,
            'hallucination': hallucination_metrics,
            'confidence': confidence_metrics,
            'reasoning': reasoning_metrics,
            'tool_efficiency': tool_metrics
        }
    
    def _evaluate_basic_accuracy(self, output: str, ground_truth: Any) -> Dict[str, Any]:
        """Simple accuracy evaluation."""
        # For demo purposes, return simple metrics
        return {
            'accuracy': 0.85,  # Placeholder
            'success': True
        }
    
    def _evaluate_tool_efficiency(self, trace: Any) -> Dict[str, Any]:
        """Evaluate tool usage efficiency."""
        if not trace:
            return {
                'total_calls': 0,
                'unique_tools': 0,
                'redundant_calls': 0,
                'efficiency_score': 100
            }
        
        # Parse tool calls from trace
        tool_calls = []
        trace_str = str(trace) if not isinstance(trace, str) else trace
        
        # Simple pattern matching for tool calls
        import re
        tool_pattern = re.compile(r'(?:tool|function|api)[\s:]+(\w+)', re.IGNORECASE)
        matches = tool_pattern.findall(trace_str)
        
        tool_calls = matches if matches else []
        
        # Calculate metrics
        unique_tools = set(tool_calls)
        redundant = len(tool_calls) - len(unique_tools)
        
        efficiency = 100
        if len(tool_calls) > 0:
            efficiency = (len(unique_tools) / len(tool_calls)) * 100
        
        return {
            'total_calls': len(tool_calls),
            'unique_tools': len(unique_tools),
            'redundant_calls': redundant,
            'efficiency_score': round(efficiency, 1)
        }
