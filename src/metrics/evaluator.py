"""
Artifact 5: Metrics & Evaluation Engine

Evaluates framework outputs using:
1. Task-specific metrics (accuracy, F1, ROUGE, etc.)
2. Agentic metrics (tool accuracy, coordination, trajectory)
3. Efficiency metrics (latency, tokens, cost)
4. LLM-as-judge for qualitative assessment
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
import numpy as np
from collections import defaultdict
import time

# ============================================================================
# CORE METRIC TYPES
# ============================================================================

class MetricType(Enum):
    """Types of metrics we evaluate."""
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    ROUGE = "rouge"
    TOOL_ACCURACY = "tool_accuracy"
    COORDINATION_SCORE = "coordination_score"
    TRAJECTORY_SCORE = "trajectory_score"
    LATENCY = "latency"
    TOKEN_EFFICIENCY = "token_efficiency"
    COST = "cost"
    LLM_JUDGE = "llm_judge"


@dataclass
class MetricResult:
    """Single metric result."""
    metric_type: MetricType
    score: float
    details: Dict[str, Any]
    explanation: Optional[str] = None


@dataclass
class EvaluationResult:
    """Complete evaluation for one test case."""
    use_case: str
    framework: str
    test_id: str
    metrics: List[MetricResult]
    succeeded: bool
    error: Optional[str] = None
    
    @property
    def summary(self) -> Dict[str, float]:
        """Get metric summary."""
        return {m.metric_type.value: m.score for m in self.metrics}
    
    @property
    def overall_accuracy(self) -> float:
        """Get overall accuracy from metrics."""
        accuracy_metrics = [m.score for m in self.metrics if m.metric_type == MetricType.ACCURACY]
        if accuracy_metrics:
            return sum(accuracy_metrics) / len(accuracy_metrics)
        # Fall back to average of all scores if no accuracy metrics
        if self.metrics:
            return sum(m.score for m in self.metrics) / len(self.metrics)
        return 0.0


# ============================================================================
# 1. TASK-SPECIFIC METRICS
# ============================================================================

class TaskMetrics:
    """Calculate task-specific metrics based on use case."""
    
    @staticmethod
    def evaluate_movie_recommendation(
        predicted: List[int],
        ground_truth: List[int],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate movie recommendations.
        
        Metrics:
        - Precision@K: % of recommended movies in ground truth
        - Recall@K: % of ground truth movies that were recommended
        - NDCG@K: Normalized discounted cumulative gain
        """
        predicted_set = set(predicted[:k])
        ground_truth_set = set(ground_truth)
        
        # Precision@K
        precision = len(predicted_set & ground_truth_set) / k if k > 0 else 0
        
        # Recall@K
        recall = len(predicted_set & ground_truth_set) / len(ground_truth_set) if ground_truth_set else 0
        
        # NDCG@K
        dcg = sum([1 / np.log2(i + 2) for i, movie_id in enumerate(predicted[:k]) 
                   if movie_id in ground_truth_set])
        idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(ground_truth_set)))])
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return {
            'precision_at_k': precision,
            'recall_at_k': recall,
            'ndcg_at_k': ndcg,
            'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        }
    
    @staticmethod
    def evaluate_github_triage(
        predicted_label: str,
        predicted_priority: str,
        ground_truth_label: str,
        ground_truth_priority: str
    ) -> Dict[str, float]:
        """
        Evaluate GitHub issue triage.
        
        Metrics:
        - Label accuracy
        - Priority accuracy
        - Combined accuracy
        """
        label_correct = 1.0 if predicted_label == ground_truth_label else 0.0
        priority_correct = 1.0 if predicted_priority == ground_truth_priority else 0.0
        
        return {
            'label_accuracy': label_correct,
            'priority_accuracy': priority_correct,
            'combined_accuracy': (label_correct + priority_correct) / 2
        }
    
    @staticmethod
    def evaluate_recipe_generation(
        generated_recipe: str,
        ground_truth_recipe: str
    ) -> Dict[str, float]:
        """
        Evaluate recipe generation using ROUGE.
        
        Metrics:
        - ROUGE-1: Unigram overlap
        - ROUGE-2: Bigram overlap
        - ROUGE-L: Longest common subsequence
        """
        # Simple ROUGE implementation (use rouge-score library in production)
        def simple_rouge_1(pred: str, ref: str) -> float:
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            overlap = len(pred_words & ref_words)
            return overlap / len(ref_words) if ref_words else 0
        
        def simple_rouge_2(pred: str, ref: str) -> float:
            pred_bigrams = set(zip(pred.lower().split()[:-1], pred.lower().split()[1:]))
            ref_bigrams = set(zip(ref.lower().split()[:-1], ref.lower().split()[1:]))
            overlap = len(pred_bigrams & ref_bigrams)
            return overlap / len(ref_bigrams) if ref_bigrams else 0
        
        return {
            'rouge_1': simple_rouge_1(generated_recipe, ground_truth_recipe),
            'rouge_2': simple_rouge_2(generated_recipe, ground_truth_recipe),
            'rouge_l': simple_rouge_1(generated_recipe, ground_truth_recipe)  # Simplified
        }
    
    @staticmethod
    def evaluate_research_summary(
        summary: str,
        key_themes: List[str],
        ground_truth_summary: str,
        ground_truth_themes: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate research paper summarization.
        
        Metrics:
        - ROUGE scores for summary
        - Theme recall: % of ground truth themes identified
        - Theme precision: % of predicted themes that are correct
        """
        # ROUGE for summary
        rouge_scores = TaskMetrics.evaluate_recipe_generation(summary, ground_truth_summary)
        
        # Theme evaluation
        predicted_themes_set = set([t.lower() for t in key_themes])
        ground_truth_themes_set = set([t.lower() for t in ground_truth_themes])
        
        theme_precision = len(predicted_themes_set & ground_truth_themes_set) / len(predicted_themes_set) if predicted_themes_set else 0
        theme_recall = len(predicted_themes_set & ground_truth_themes_set) / len(ground_truth_themes_set) if ground_truth_themes_set else 0
        
        return {
            **rouge_scores,
            'theme_precision': theme_precision,
            'theme_recall': theme_recall,
            'theme_f1': 2 * (theme_precision * theme_recall) / (theme_precision + theme_recall) if (theme_precision + theme_recall) > 0 else 0
        }
    
    @staticmethod
    def evaluate_email_automation(
        generated_responses: List[str],
        selected_template: str,
        ground_truth_responses: List[str],
        ground_truth_template: str
    ) -> Dict[str, float]:
        """
        Evaluate email automation.
        
        Metrics:
        - Template accuracy
        - ROUGE for generated responses
        """
        template_correct = 1.0 if selected_template == ground_truth_template else 0.0
        
        # Average ROUGE across all responses
        rouge_scores = []
        for gen, gt in zip(generated_responses, ground_truth_responses):
            rouge_scores.append(TaskMetrics.evaluate_recipe_generation(gen, gt))
        
        avg_rouge = {
            'rouge_1': np.mean([s['rouge_1'] for s in rouge_scores]),
            'rouge_2': np.mean([s['rouge_2'] for s in rouge_scores]),
            'rouge_l': np.mean([s['rouge_l'] for s in rouge_scores])
        }
        
        return {
            'template_accuracy': template_correct,
            **avg_rouge
        }


# ============================================================================
# 2. AGENTIC METRICS
# ============================================================================

class AgenticMetrics:
    """Calculate agentic-specific metrics."""
    
    @staticmethod
    def evaluate_tool_accuracy(
        tool_calls: List[Dict[str, Any]],
        expected_tools: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate tool call accuracy.
        
        Metrics:
        - Selection accuracy: % of correct tools called
        - Parameter accuracy: % of calls with valid parameters
        - Unnecessary calls: Count of unneeded tools
        - Missing calls: Count of needed but not called
        """
        actual_tools = [call['tool_name'] for call in tool_calls]
        expected_set = set(expected_tools)
        actual_set = set(actual_tools)
        
        # Selection accuracy
        correct_tools = expected_set & actual_set
        selection_accuracy = len(correct_tools) / len(expected_set) if expected_set else 0
        
        # Parameter accuracy (simplified - check if all have valid params)
        valid_params = sum(1 for call in tool_calls if call.get('parameters'))
        parameter_accuracy = valid_params / len(tool_calls) if tool_calls else 0
        
        # Unnecessary and missing
        unnecessary = len(actual_set - expected_set)
        missing = len(expected_set - actual_set)
        
        # Tool Utilization Efficacy (TUE)
        tue = (selection_accuracy * 0.4 + 
               parameter_accuracy * 0.3 + 
               0.3 -  # Sequence score placeholder
               unnecessary * 0.05 - 
               missing * 0.10)
        tue = max(0, min(1, tue))
        
        return {
            'selection_accuracy': selection_accuracy,
            'parameter_accuracy': parameter_accuracy,
            'unnecessary_calls': unnecessary,
            'missing_calls': missing,
            'tool_utilization_efficacy': tue
        }
    
    @staticmethod
    def evaluate_coordination(
        trajectory: List[Dict[str, Any]],
        num_agents: int
    ) -> Dict[str, float]:
        """
        Evaluate multi-agent coordination.
        
        Metrics:
        - Individual KPI per agent
        - Overall KPI (average)
        - Communication score (heuristic)
        - Planning score (heuristic)
        """
        # Count milestones achieved
        milestones = [step for step in trajectory if step.get('is_milestone')]
        total_milestones = len(milestones)
        
        # Individual KPI per agent
        agent_contributions = defaultdict(int)
        for milestone in milestones:
            agent_id = milestone.get('agent_id', 'unknown')
            agent_contributions[agent_id] += 1
        
        individual_kpis = {
            agent: count / total_milestones if total_milestones > 0 else 0
            for agent, count in agent_contributions.items()
        }
        
        overall_kpi = np.mean(list(individual_kpis.values())) if individual_kpis else 0
        
        # Communication score (heuristic: check for information sharing)
        handoffs = sum(1 for i in range(len(trajectory) - 1) 
                      if trajectory[i].get('agent_id') != trajectory[i+1].get('agent_id'))
        clean_handoffs = sum(1 for i in range(len(trajectory) - 1)
                           if trajectory[i].get('agent_id') != trajectory[i+1].get('agent_id')
                           and trajectory[i+1].get('has_context', False))
        
        communication_score = clean_handoffs / handoffs if handoffs > 0 else 1.0
        communication_score = communication_score * 5  # Scale to 0-5
        
        # Planning score (heuristic: check for logical ordering)
        planning_score = 4.0  # Placeholder - would need more sophisticated analysis
        
        coordination_score = (communication_score + planning_score) / 2
        
        return {
            'overall_kpi': overall_kpi,
            'individual_kpis': individual_kpis,
            'communication_score': communication_score,
            'planning_score': planning_score,
            'coordination_score': coordination_score
        }
    
    @staticmethod
    def evaluate_trajectory(
        trajectory: List[Dict[str, Any]],
        use_case: str
    ) -> Dict[str, float]:
        """
        Evaluate agent trajectory quality.
        
        Metrics:
        - Efficiency: Minimum steps taken?
        - Reasonableness: Logical order?
        - Adaptability: Recovered from errors?
        """
        score = 10.0
        
        # Check for redundant calls
        tool_calls = [step.get('tool_name') for step in trajectory if step.get('tool_name')]
        if len(tool_calls) != len(set(tool_calls)):
            score -= 2  # Redundant calls
        
        # Check for errors and recovery
        errors = [step for step in trajectory if step.get('error')]
        if errors:
            recoveries = sum(1 for i, step in enumerate(trajectory) 
                           if step.get('error') and i < len(trajectory) - 1 
                           and trajectory[i+1].get('recovered'))
            if recoveries == len(errors):
                score += 1  # Good recovery
            else:
                score -= 2  # Poor error handling
        
        # Normalize to 0-10
        score = max(0, min(10, score))
        
        return {
            'trajectory_score': score / 10,  # Normalize to 0-1
            'num_steps': len(trajectory),
            'num_errors': len(errors),
            'redundant_calls': len(tool_calls) - len(set(tool_calls))
        }


# ============================================================================
# 3. EFFICIENCY METRICS
# ============================================================================

class EfficiencyMetrics:
    """Calculate efficiency metrics."""
    
    @staticmethod
    def calculate_efficiency(
        latency_seconds: float,
        token_usage: Dict[str, int],
        model_pricing: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate efficiency metrics.
        
        Args:
            latency_seconds: Total time taken
            token_usage: {'input': X, 'output': Y, 'total': Z}
            model_pricing: {'input_per_1k': X, 'output_per_1k': Y}
        
        Returns:
            Efficiency metrics including cost
        """
        # Cost calculation
        input_cost = (token_usage['input'] / 1000) * model_pricing['input_per_1k']
        output_cost = (token_usage['output'] / 1000) * model_pricing['output_per_1k']
        total_cost = input_cost + output_cost
        
        # Tokens per second
        tokens_per_second = token_usage['total'] / latency_seconds if latency_seconds > 0 else 0
        
        return {
            'latency_seconds': latency_seconds,
            'total_tokens': token_usage['total'],
            'tokens_per_second': tokens_per_second,
            'cost_usd': total_cost,
            'cost_per_token': total_cost / token_usage['total'] if token_usage['total'] > 0 else 0
        }


# ============================================================================
# 4. LLM-AS-JUDGE
# ============================================================================

class LLMJudge:
    """Use LLM to evaluate qualitative aspects."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
    
    def evaluate_task_adherence(
        self,
        request: str,
        agent_output: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate how well output satisfies the request.
        
        Returns:
            {
                'score': 1-5,
                'justification': str,
                'criteria_met': List[bool]
            }
        """
        prompt = f"""Evaluate how well this agent output satisfies the request.

REQUEST:
{request}

AGENT OUTPUT:
{agent_output}

{f"GROUND TRUTH: {ground_truth}" if ground_truth else ""}

Score 1-5:
5 = Perfectly satisfies request with complete, accurate answer
4 = Good answer with minor gaps
3 = Acceptable but incomplete or slightly off-target
2 = Partially addresses request but significant issues
1 = Fails to address request

Provide:
1. Score (1-5)
2. Justification (2-3 sentences)
3. Which criteria were met (completeness, accuracy, relevance)

Format as JSON:
{{
    "score": <1-5>,
    "justification": "<explanation>",
    "completeness": <true/false>,
    "accuracy": <true/false>,
    "relevance": <true/false>
}}
"""
        
        # In production, would call LLM API here
        # For testing harness, return placeholder
        return {
            'score': 4.0,
            'justification': "Good response addressing main points",
            'completeness': True,
            'accuracy': True,
            'relevance': True
        }
    
    def evaluate_output_quality(
        self,
        output: str,
        use_case: str
    ) -> Dict[str, float]:
        """
        Evaluate overall output quality.
        
        Returns:
            {
                'coherence': 0-1,
                'relevance': 0-1,
                'completeness': 0-1,
                'overall': 0-1
            }
        """
        # Placeholder - would use LLM in production
        return {
            'coherence': 0.85,
            'relevance': 0.90,
            'completeness': 0.80,
            'overall': 0.85
        }


# ============================================================================
# 5. MAIN EVALUATOR
# ============================================================================

class Evaluator:
    """Main evaluation engine."""
    
    def __init__(self, llm_judge: Optional[LLMJudge] = None):
        self.task_metrics = TaskMetrics()
        self.agentic_metrics = AgenticMetrics()
        self.efficiency_metrics = EfficiencyMetrics()
        self.llm_judge = llm_judge or LLMJudge()
    
    def evaluate(
        self,
        use_case: str,
        framework: str,
        test_id: str,
        framework_result: Any,  # FrameworkResult from adapter
        ground_truth: Dict[str, Any],
        expected_tools: Optional[List[str]] = None
    ) -> EvaluationResult:
        """
        Complete evaluation of framework result.
        
        Returns:
            EvaluationResult with all metrics
        """
        metrics = []
        
        try:
            # 1. Task-specific metrics
            task_metrics = self._evaluate_task_metrics(
                use_case, 
                framework_result.result, 
                ground_truth
            )
            for metric_name, score in task_metrics.items():
                metrics.append(MetricResult(
                    metric_type=MetricType.ACCURACY,
                    score=score,
                    details={'metric': metric_name}
                ))
            
            # 2. Agentic metrics (if trajectory available)
            if hasattr(framework_result, 'trajectory') and framework_result.trajectory:
                # Tool accuracy
                if expected_tools:
                    tool_acc = self.agentic_metrics.evaluate_tool_accuracy(
                        framework_result.trajectory,
                        expected_tools
                    )
                    metrics.append(MetricResult(
                        metric_type=MetricType.TOOL_ACCURACY,
                        score=tool_acc['tool_utilization_efficacy'],
                        details=tool_acc
                    ))
                
                # Trajectory quality
                traj_score = self.agentic_metrics.evaluate_trajectory(
                    framework_result.trajectory,
                    use_case
                )
                metrics.append(MetricResult(
                    metric_type=MetricType.TRAJECTORY_SCORE,
                    score=traj_score['trajectory_score'],
                    details=traj_score
                ))
            
            # 3. Efficiency metrics
            eff_metrics = self.efficiency_metrics.calculate_efficiency(
                framework_result.latency_seconds,
                {
                    'input': framework_result.token_usage.input_tokens,
                    'output': framework_result.token_usage.output_tokens,
                    'total': framework_result.token_usage.total_tokens
                },
                {'input_per_1k': 0.00015, 'output_per_1k': 0.0006}  # GPT-4o-mini
            )
            metrics.append(MetricResult(
                metric_type=MetricType.LATENCY,
                score=1.0 / (1.0 + eff_metrics['latency_seconds'] / 60),  # Normalize
                details=eff_metrics
            ))
            
            # 4. LLM-as-judge (optional)
            if hasattr(framework_result, 'explanation'):
                judge_score = self.llm_judge.evaluate_task_adherence(
                    request=f"Complete {use_case} task",
                    agent_output=str(framework_result.result),
                    ground_truth=str(ground_truth)
                )
                metrics.append(MetricResult(
                    metric_type=MetricType.LLM_JUDGE,
                    score=judge_score['score'] / 5.0,  # Normalize to 0-1
                    details=judge_score,
                    explanation=judge_score['justification']
                ))
            
            return EvaluationResult(
                use_case=use_case,
                framework=framework,
                test_id=test_id,
                metrics=metrics,
                succeeded=True
            )
        
        except Exception as e:
            return EvaluationResult(
                use_case=use_case,
                framework=framework,
                test_id=test_id,
                metrics=[],
                succeeded=False,
                error=str(e)
            )
    
    def _evaluate_task_metrics(
        self,
        use_case: str,
        result: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, float]:
        """Route to appropriate task metric evaluator."""
        if use_case == "movie_recommendation":
            return self.task_metrics.evaluate_movie_recommendation(
                result['recommendations'],
                ground_truth['recommendations']
            )
        elif use_case == "github_triage":
            return self.task_metrics.evaluate_github_triage(
                result['classification'],
                result['priority'],
                ground_truth['classification'],
                ground_truth['priority']
            )
        elif use_case == "recipe_generation":
            return self.task_metrics.evaluate_recipe_generation(
                result['recipe'],
                ground_truth['recipe']
            )
        elif use_case == "research_summary":
            return self.task_metrics.evaluate_research_summary(
                result['summary'],
                result['key_themes'],
                ground_truth['summary'],
                ground_truth['key_themes']
            )
        elif use_case == "email_automation":
            return self.task_metrics.evaluate_email_automation(
                result['responses'],
                result['template'],
                ground_truth['responses'],
                ground_truth['template']
            )
        else:
            return {'score': 0.0}


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example: Evaluate movie recommendation
    from src.core.types import FrameworkResult, TokenUsage
    
    # Mock framework result
    framework_result = FrameworkResult(
        succeeded=True,
        result={
            'recommendations': [101, 205, 789, 456, 123]
        },
        latency_seconds=2.5,
        token_usage=TokenUsage(
            input_tokens=500,
            output_tokens=200,
            total_tokens=700
        ),
        trajectory=[
            {'agent_id': 'retrieval', 'tool_name': 'get_user_history', 'is_milestone': True},
            {'agent_id': 'filtering', 'tool_name': 'filter_by_genre', 'is_milestone': True},
            {'agent_id': 'ranking', 'tool_name': 'rank_by_score', 'is_milestone': True}
        ]
    )
    
    # Ground truth
    ground_truth = {
        'recommendations': [101, 205, 456, 789, 999]
    }
    
    # Evaluate
    evaluator = Evaluator()
    result = evaluator.evaluate(
        use_case="movie_recommendation",
        framework="langgraph",
        test_id="test_001",
        framework_result=framework_result,
        ground_truth=ground_truth,
        expected_tools=['get_user_history', 'filter_by_genre', 'rank_by_score']
    )
    
    print(f"\nEvaluation Result for {result.framework} on {result.use_case}:")
    print(f"Success: {result.succeeded}")
    print("\nMetrics:")
    for metric in result.metrics:
        print(f"  {metric.metric_type.value}: {metric.score:.3f}")
        if metric.explanation:
            print(f"    Explanation: {metric.explanation}")
