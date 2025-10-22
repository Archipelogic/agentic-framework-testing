"""
Capability scoring based on real test results and enhanced metrics.
"""

from typing import Dict, Any


def calculate_real_capability_scores(results: Dict, enhanced_metrics: Dict) -> Dict:
    """Calculate capability scores based on real accuracy and metrics."""
    capability_scores = {}
    
    for framework_name, framework_results in results.items():
        # Calculate success rate and accuracy
        total_tests = 0
        total_passed = 0
        total_accuracy = 0
        
        for use_case_data in framework_results.values():
            if 'tests_run' in use_case_data:
                total_tests += use_case_data['tests_run']
                total_passed += use_case_data['tests_passed']
                total_accuracy += use_case_data.get('average_accuracy', 0) * use_case_data['tests_run']
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        avg_accuracy = (total_accuracy / total_tests) if total_tests > 0 else 0
        
        # Get enhanced metrics if available
        framework_enhanced = enhanced_metrics.get(framework_name, {})
        
        # Aggregate enhanced metrics across use cases
        avg_metrics = {
            'reasoning_depth': 0,
            'planning_score': 0,
            'decision_confidence': 0,
            'grounding_score': 0,
            'factual_consistency': 0,
            'tool_efficiency': 0,
            'backtrack_rate': 0,
            'resource_efficiency': 0,
            'count': 0
        }
        
        for use_case_metrics in framework_enhanced.values():
            if use_case_metrics:
                avg_metrics['count'] += 1
                if 'reasoning' in use_case_metrics:
                    avg_metrics['reasoning_depth'] += use_case_metrics['reasoning'].get('reasoning_depth', 0)
                    avg_metrics['planning_score'] += use_case_metrics['reasoning'].get('planning_score', 0)
                    avg_metrics['decision_confidence'] += use_case_metrics['reasoning'].get('decision_confidence', 0)
                    avg_metrics['backtrack_rate'] += use_case_metrics['reasoning'].get('backtrack_rate', 0)
                if 'hallucination' in use_case_metrics:
                    avg_metrics['grounding_score'] += use_case_metrics['hallucination'].get('grounding_score', 0)
                    avg_metrics['factual_consistency'] += use_case_metrics['hallucination'].get('factual_consistency', 0)
                if 'tool_efficiency' in use_case_metrics:
                    avg_metrics['tool_efficiency'] += use_case_metrics['tool_efficiency'].get('efficiency_score', 0)
                if 'resource' in use_case_metrics:
                    mem_efficiency = max(0, 100 - use_case_metrics['resource'].get('memory_delta_mb', 0))
                    cpu_efficiency = max(0, 100 - use_case_metrics['resource'].get('cpu_percent', 0))
                    avg_metrics['resource_efficiency'] += (mem_efficiency + cpu_efficiency) / 2
        
        # Calculate averages
        if avg_metrics['count'] > 0:
            for key in avg_metrics:
                if key != 'count':
                    avg_metrics[key] /= avg_metrics['count']
        
        # Calculate capability scores based on REAL metrics
        scores = {
            # Multi-agent based on success rate and decision confidence
            'multi_agent': min(100, success_rate * 0.7 + avg_metrics['decision_confidence'] * 0.3),
            
            # Tool usage based on actual tool efficiency
            'tool_usage': avg_metrics['tool_efficiency'] if avg_metrics['tool_efficiency'] > 0 else success_rate * 0.5,
            
            # Error handling based on accuracy and factual consistency
            'error_handling': min(100, avg_accuracy * 50 + avg_metrics['factual_consistency'] * 0.5),
            
            # Context retention based on grounding score
            'context_retention': avg_metrics['grounding_score'] if avg_metrics['grounding_score'] > 0 else avg_accuracy * 30,
            
            # Adaptability based on planning minus backtracking
            'adaptability': max(0, avg_metrics['planning_score'] - avg_metrics['backtrack_rate']),
            
            # Scalability based on resource efficiency
            'scalability': avg_metrics['resource_efficiency'] if avg_metrics['resource_efficiency'] > 0 else 50,
            
            # Observability based on reasoning depth
            'observability': min(100, avg_metrics['reasoning_depth'] * 5) if avg_metrics['reasoning_depth'] > 0 else success_rate * 0.4,
            
            # RAG capability based on grounding and accuracy
            'rag_capability': avg_metrics['grounding_score'] if avg_metrics['grounding_score'] > 0 else avg_accuracy * 25
        }
        
        # Ensure all scores are within 0-100 range
        for key in scores:
            scores[key] = max(0, min(100, scores[key]))
        
        # Calculate overall score
        scores['overall_score'] = sum(scores.values()) / len(scores)
        
        capability_scores[framework_name] = scores
    
    return capability_scores
