# Evaluation System Guide

## Overview

The evaluation system provides comprehensive metrics to assess framework performance across multiple dimensions, ensuring fair and accurate comparisons.

## Comprehensive Metrics System (22 Total Metrics)

### Core Capabilities (60% weight)
- **Multi-Agent (9%)**: Orchestrating multiple AI agents
- **Tool Usage (7%)**: External tool integration effectiveness
- **Error Handling (7%)**: Robustness and recovery from errors
- **Context Retention (6%)**: Memory and state management
- **Adaptability (6%)**: Flexibility to changing requirements
- **Scalability (7%)**: Performance under load
- **Observability (5%)**: Logging, debugging, transparency
- **RAG (13%)**: Document retrieval and grounding

### Enhanced Metrics (25% weight)
- **Reasoning Depth (3%)**: Number of reasoning steps taken
- **Planning Score (4%)**: Quality of upfront planning
- **Decision Confidence (3%)**: Certainty in decisions made
- **Backtrack Rate (2%)**: How often decisions are reversed (lower is better)
- **Grounding Score (4%)**: Percentage of claims supported by context
- **Factual Consistency (4%)**: Absence of contradictions
- **Tool Efficiency (5%)**: Optimal tool usage without redundancy

### Resource Metrics (5% weight)
- **Memory Efficiency (2%)**: RAM usage (lower is better)
- **CPU Efficiency (2%)**: Processor utilization (lower is better)
- **Cache Hit Rate (1%)**: Data reuse efficiency

### Performance Metrics (10% weight)
- **Success Rate (7%)**: Percentage of tests passing
- **Latency Score (2%)**: Processing speed (lower is better)
- **Cost Efficiency (1%)**: Token usage costs (lower is better)

## Core Metrics

### 1. Accuracy Metrics

**Semantic Accuracy** (0-1 score)
- Measures how well the output matches expected results
- Uses embedding similarity and keyword matching
- Weighted by importance of fields

```python
def calculate_semantic_accuracy(output, ground_truth):
    """Calculate semantic similarity between output and ground truth."""
    # Extract key fields
    output_text = extract_text_fields(output)
    truth_text = extract_text_fields(ground_truth)
    
    # Calculate embedding similarity
    output_embedding = get_embedding(output_text)
    truth_embedding = get_embedding(truth_text)
    
    similarity = cosine_similarity(output_embedding, truth_embedding)
    return similarity
```

**Structural Accuracy** (0-1 score)
- Verifies presence of required fields
- Checks data types and formats
- Validates schema compliance

```python
def calculate_structural_accuracy(output, schema):
    """Validate output structure against schema."""
    required_fields = schema.get("required", [])
    present_fields = set(output.keys())
    
    # Check required fields
    missing = set(required_fields) - present_fields
    if missing:
        return 1.0 - (len(missing) / len(required_fields))
    
    return 1.0
```

### 2. Performance Metrics

**Latency**
- End-to-end execution time in seconds
- Includes model inference, processing, and I/O
- Measured per test case and aggregated

**Throughput**
- Requests processed per second
- Tests under various load conditions
- Identifies bottlenecks and scaling limits

```python
class PerformanceMetrics:
    """Track performance metrics."""
    
    def measure_latency(self, func, *args, **kwargs):
        """Measure function execution time."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        latency = time.perf_counter() - start
        
        return result, latency
    
    def calculate_throughput(self, latencies):
        """Calculate throughput from latencies."""
        if not latencies:
            return 0
        
        total_time = sum(latencies)
        num_requests = len(latencies)
        
        return num_requests / total_time if total_time > 0 else 0
```

### 3. Cost Metrics

**Token Usage**
- Input tokens
- Output tokens
- Total tokens per request

**Monetary Cost**
- Cost per request in USD
- Aggregated cost per use case
- Total benchmark cost

```python
class CostCalculator:
    """Calculate costs for different models."""
    
    PRICING = {
        "gpt-4o": {"input": 0.01, "output": 0.03},  # per 1K tokens
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
    }
    
    def calculate_cost(self, model, input_tokens, output_tokens):
        """Calculate cost for a request."""
        if model not in self.PRICING:
            return 0.0
        
        pricing = self.PRICING[model]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
```

### 4. Reliability Metrics

**Success Rate**
- Percentage of successful completions
- Tracks failures and errors
- Identifies stability issues

**Error Rate**
- Types of errors encountered
- Error frequency by use case
- Recovery success rate

```python
class ReliabilityMetrics:
    """Track reliability metrics."""
    
    def __init__(self):
        self.total_attempts = 0
        self.successful = 0
        self.errors = defaultdict(int)
    
    def record_attempt(self, success, error_type=None):
        """Record an attempt."""
        self.total_attempts += 1
        if success:
            self.successful += 1
        elif error_type:
            self.errors[error_type] += 1
    
    def get_success_rate(self):
        """Calculate success rate."""
        if self.total_attempts == 0:
            return 0
        return self.successful / self.total_attempts
    
    def get_error_distribution(self):
        """Get error type distribution."""
        return dict(self.errors)
```

## Composite Scoring

### Overall Score Calculation

The overall score combines multiple metrics using weighted averaging:

```python
class CompositeScorer:
    """Calculate composite scores."""
    
    DEFAULT_WEIGHTS = {
        "accuracy": 0.35,
        "performance": 0.25,
        "cost": 0.20,
        "reliability": 0.20
    }
    
    def calculate_composite_score(self, metrics, weights=None):
        """Calculate weighted composite score."""
        weights = weights or self.DEFAULT_WEIGHTS
        
        score = 0
        for metric, weight in weights.items():
            if metric in metrics:
                # Normalize metric to 0-1 range
                normalized = self.normalize_metric(metric, metrics[metric])
                score += normalized * weight
        
        return score
    
    def normalize_metric(self, metric_name, value):
        """Normalize metric to 0-1 range."""
        if metric_name == "accuracy":
            return value  # Already 0-1
        elif metric_name == "performance":
            # Lower latency is better
            return 1.0 / (1.0 + value)  # Inverse with smoothing
        elif metric_name == "cost":
            # Lower cost is better
            return 1.0 / (1.0 + value * 100)  # Scale and inverse
        elif metric_name == "reliability":
            return value  # Success rate, already 0-1
        return 0
```

## Use Case Specific Evaluation

### Movie Recommendation Evaluation

```python
class MovieRecommendationEvaluator:
    """Evaluate movie recommendation outputs."""
    
    def evaluate(self, output, ground_truth):
        """Comprehensive evaluation for movie recommendations."""
        scores = {}
        
        # Check if recommendations exist
        if "recommendations" not in output:
            return {"accuracy": 0, "completeness": 0}
        
        recommendations = output["recommendations"]
        
        # Genre accuracy
        scores["genre_accuracy"] = self._evaluate_genres(
            recommendations, ground_truth
        )
        
        # Rating range compliance
        scores["rating_compliance"] = self._check_rating_range(
            recommendations, ground_truth.get("min_rating", 0)
        )
        
        # Diversity score
        scores["diversity"] = self._calculate_diversity(recommendations)
        
        # Explanation quality
        scores["explanation_quality"] = self._evaluate_explanations(
            recommendations
        )
        
        # Overall accuracy
        scores["accuracy"] = np.mean(list(scores.values()))
        
        return scores
    
    def _evaluate_genres(self, recommendations, ground_truth):
        """Check if recommended movies match requested genres."""
        requested_genres = set(ground_truth.get("preferences", []))
        excluded_genres = set(ground_truth.get("exclude_genres", []))
        
        correct = 0
        for movie in recommendations:
            movie_genres = set(movie.get("genres", []))
            
            # Check if has requested genres
            if movie_genres & requested_genres:
                correct += 0.5
            
            # Check if doesn't have excluded genres
            if not (movie_genres & excluded_genres):
                correct += 0.5
        
        return correct / len(recommendations) if recommendations else 0
```

### GitHub Triage Evaluation

```python
class GitHubTriageEvaluator:
    """Evaluate GitHub issue triage outputs."""
    
    VALID_PRIORITIES = ["low", "medium", "high", "critical"]
    VALID_LABELS = ["bug", "feature", "enhancement", "documentation", "question"]
    
    def evaluate(self, output, ground_truth):
        """Evaluate GitHub triage output."""
        scores = {}
        
        # Priority accuracy
        if "priority" in output:
            scores["priority_accuracy"] = self._evaluate_priority(
                output["priority"], ground_truth.get("priority")
            )
        
        # Label accuracy
        if "labels" in output:
            scores["label_accuracy"] = self._evaluate_labels(
                output["labels"], ground_truth.get("labels", [])
            )
        
        # Assignee appropriateness
        if "assignee" in output:
            scores["assignee_score"] = self._evaluate_assignee(
                output["assignee"], output.get("labels", [])
            )
        
        # Triage notes quality
        if "triage_notes" in output:
            scores["notes_quality"] = self._evaluate_notes(
                output["triage_notes"]
            )
        
        return scores
    
    def _evaluate_priority(self, predicted, actual):
        """Evaluate priority prediction."""
        if predicted == actual:
            return 1.0
        
        # Partial credit for close priorities
        priority_map = {p: i for i, p in enumerate(self.VALID_PRIORITIES)}
        
        if predicted in priority_map and actual in priority_map:
            distance = abs(priority_map[predicted] - priority_map[actual])
            return max(0, 1.0 - (distance * 0.25))
        
        return 0
```

## Enhanced Metrics (Advanced)

### 1. Reasoning Analysis

**Purpose**: Understand HOW agents solve problems, not just if they succeed.

```python
reasoning_metrics = {
    'reasoning_depth': 12,          # Number of reasoning steps
    'max_depth': 3,                  # Maximum hierarchy depth
    'decision_points': 4,            # Number of decisions made
    'planning_score': 75,            # Quality of planning (0-100)
    'backtrack_rate': 10.5,          # % of steps that were backtracks
    'problem_decomposition_score': 80,  # How well problems are broken down
    'decision_confidence': 85        # Confidence in decisions (0-100)
}
```

**Key Insights**:
- Higher reasoning depth indicates more thorough analysis
- Lower backtrack rates suggest better initial planning
- Good problem decomposition correlates with success

### 2. Hallucination Detection

**Purpose**: Ensure outputs are grounded in facts and context.

```python
hallucination_metrics = {
    'grounding_score': 92.5,         # % of claims supported by context
    'numbers_grounded': 100,          # % of numbers from context
    'entities_grounded': 85,          # % of entities from context
    'factual_consistency': 100,       # No contradictions detected
    'confidence_score': 65,           # Appropriate confidence level
    'appropriate_confidence': True    # Not overconfident
}
```

**Detection Methods**:
- Compare output claims against input context
- Check numeric consistency
- Identify unsupported entities
- Detect contradiction patterns

### 3. Resource Usage Tracking

**Purpose**: Monitor computational efficiency.

```python
resource_metrics = {
    'memory_current_mb': 245.3,      # Current memory usage
    'memory_peak_mb': 312.7,          # Peak memory usage
    'memory_delta_mb': 67.4,          # Memory increase during execution
    'cpu_percent': 23.5,              # CPU utilization
    'cache_hit_rate': 78,             # Cache efficiency %
    'cache_total_keys': 42,           # Unique cache entries
}
```

**Optimization Targets**:
- Keep memory delta < 100MB for most tasks
- Maintain cache hit rate > 70%
- CPU usage should scale with complexity

### 4. Tool Efficiency Analysis

**Purpose**: Evaluate how effectively agents use available tools.

```python
tool_metrics = {
    'total_calls': 8,                # Total tool invocations
    'unique_tools': 3,               # Different tools used
    'redundant_calls': 2,            # Duplicate/unnecessary calls
    'efficiency_score': 75.0         # Overall efficiency %
}
```

**Best Practices**:
- Minimize redundant calls
- Use appropriate tools for tasks
- Batch operations when possible

## Ensuring Fair Evaluation

### 1. Standardized Test Conditions

```python
class FairEvaluationManager:
    """Ensure fair evaluation conditions."""
    
    def __init__(self):
        self.config = {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 2000,
            "seed": 42  # For reproducibility
        }
    
    def prepare_test_environment(self):
        """Prepare standardized test environment."""
        # Set random seeds
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        
        # Clear caches
        self.clear_caches()
        
        # Verify model availability
        self.verify_model_access()
        
        # Warm up models
        self.warmup_models()
    
    def validate_fairness(self, framework_configs):
        """Validate that all frameworks use same conditions."""
        base_model = framework_configs[0].get("model")
        
        for config in framework_configs:
            if config.get("model") != base_model:
                raise ValueError(f"Model mismatch: {config['model']} != {base_model}")
            
            if config.get("temperature") != self.config["temperature"]:
                raise ValueError("Temperature settings differ")
        
        return True
```

### 2. Statistical Significance Testing

```python
from scipy import stats

class StatisticalValidator:
    """Validate statistical significance of results."""
    
    def compare_frameworks(self, results_a, results_b, metric="accuracy"):
        """Compare two frameworks using statistical tests."""
        scores_a = [r[metric] for r in results_a]
        scores_b = [r[metric] for r in results_b]
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
        
        # Calculate effect size (Cohen's d)
        effect_size = self.calculate_cohens_d(scores_a, scores_b)
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "effect_size": effect_size,
            "interpretation": self.interpret_effect_size(effect_size)
        }
    
    def calculate_cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def interpret_effect_size(self, d):
        """Interpret Cohen's d value."""
        if abs(d) < 0.2:
            return "negligible"
        elif abs(d) < 0.5:
            return "small"
        elif abs(d) < 0.8:
            return "medium"
        else:
            return "large"
```

### 3. Cross-Validation

```python
from sklearn.model_selection import KFold

class CrossValidator:
    """Perform cross-validation for robust evaluation."""
    
    def cross_validate(self, framework_adapter, test_cases, n_folds=5):
        """Perform k-fold cross-validation."""
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(test_cases)):
            # Split data
            train_cases = [test_cases[i] for i in train_idx]
            test_cases_fold = [test_cases[i] for i in test_idx]
            
            # Optionally fine-tune on train set
            # framework_adapter.fine_tune(train_cases)
            
            # Evaluate on test fold
            fold_scores = []
            for test_case in test_cases_fold:
                result = framework_adapter.run(
                    test_case.use_case,
                    test_case.input_data
                )
                score = self.evaluate_result(result, test_case.ground_truth)
                fold_scores.append(score)
            
            fold_results.append({
                "fold": fold,
                "scores": fold_scores,
                "mean": np.mean(fold_scores),
                "std": np.std(fold_scores)
            })
        
        return {
            "fold_results": fold_results,
            "overall_mean": np.mean([f["mean"] for f in fold_results]),
            "overall_std": np.std([f["mean"] for f in fold_results])
        }
```

## Evaluation Best Practices

### 1. Multiple Evaluation Runs

```python
def run_multiple_evaluations(framework, test_cases, n_runs=3):
    """Run multiple evaluations for stability."""
    all_results = []
    
    for run in range(n_runs):
        run_results = []
        for test_case in test_cases:
            result = framework.run(test_case)
            run_results.append(result)
        all_results.append(run_results)
    
    # Calculate statistics
    return {
        "mean_accuracy": np.mean([r.accuracy for run in all_results for r in run]),
        "std_accuracy": np.std([r.accuracy for run in all_results for r in run]),
        "consistency": calculate_consistency(all_results)
    }
```

### 2. Error Analysis

```python
class ErrorAnalyzer:
    """Analyze errors for improvement insights."""
    
    def analyze_errors(self, failed_results):
        """Analyze failed results for patterns."""
        error_patterns = defaultdict(list)
        
        for result in failed_results:
            error_type = self.categorize_error(result.error_message)
            error_patterns[error_type].append(result)
        
        analysis = {}
        for error_type, results in error_patterns.items():
            analysis[error_type] = {
                "count": len(results),
                "percentage": len(results) / len(failed_results),
                "examples": results[:3],  # Sample examples
                "common_features": self.find_common_features(results)
            }
        
        return analysis
```

## Next Steps

- [Configure evaluation parameters](./configuration.md)
- [Run production benchmarks](./deployment.md)
- [Troubleshoot evaluation issues](./troubleshooting.md)
