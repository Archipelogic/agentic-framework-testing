# Use Cases Guide

## Overview

Use cases are standardized evaluation scenarios that test different capabilities of agentic AI frameworks. Each use case represents a real-world task with specific inputs, expected behaviors, and evaluation criteria.

## Built-in Use Cases

### 1. Movie Recommendation
**Purpose**: Test content recommendation and user preference understanding

**Capabilities Tested**:
- User preference analysis
- Content filtering
- Ranking and scoring
- Explanation generation

**Input Format**:
```json
{
  "preferences": ["action", "sci-fi"],
  "exclude_genres": ["horror"],
  "year_range": [2020, 2024],
  "min_rating": 7.0
}
```

**Expected Output**:
```json
{
  "recommendations": [
    {
      "title": "Movie Name",
      "year": 2023,
      "genres": ["action", "sci-fi"],
      "rating": 8.2,
      "reason": "Matches your preference for action and sci-fi"
    }
  ]
}
```

### 2. GitHub Issue Triage
**Purpose**: Test issue classification and prioritization

**Capabilities Tested**:
- Text classification
- Priority assessment
- Label assignment
- Technical understanding

**Input Format**:
```json
{
  "issue_id": 123,
  "title": "Application crashes on startup",
  "body": "When I try to launch the app...",
  "author": "user123",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Expected Output**:
```json
{
  "priority": "high",
  "labels": ["bug", "critical"],
  "assignee": "backend-team",
  "estimated_effort": "medium",
  "triage_notes": "Critical startup issue affecting users"
}
```

### 3. Recipe Generation
**Purpose**: Test creative generation with constraints

**Capabilities Tested**:
- Creative content generation
- Constraint satisfaction
- Structured output formatting
- Domain knowledge

**Input Format**:
```json
{
  "ingredients": ["chicken", "rice", "vegetables"],
  "dietary_restrictions": ["gluten-free"],
  "cuisine_type": "asian",
  "difficulty": "easy",
  "prep_time_max": 30
}
```

### 4. Research Summary
**Purpose**: Test information synthesis and summarization

**Capabilities Tested**:
- Information extraction
- Summarization
- Source citation
- Fact organization

**Input Format**:
```json
{
  "topic": "quantum computing applications",
  "max_length": 500,
  "target_audience": "technical",
  "include_citations": true
}
```

### 5. Email Automation
**Purpose**: Test context understanding and professional writing

**Capabilities Tested**:
- Context interpretation
- Tone adaptation
- Professional writing
- Action extraction

**Input Format**:
```json
{
  "email_type": "follow_up",
  "context": "Meeting about Q4 planning",
  "recipient": "team",
  "key_points": ["deadline moved", "budget approved"]
}
```

## Adding Custom Use Cases

### Step 1: Define Use Case Class

Create `src/use_cases/custom_use_case.py`:

```python
from dataclasses import dataclass
from typing import Dict, Any, List
from src.core.types import UseCaseType

@dataclass
class CustomUseCaseEvaluator:
    """Evaluator for custom use case."""
    
    def generate_test_cases(self, num_cases: int) -> List[Dict[str, Any]]:
        """Generate test cases for the use case."""
        test_cases = []
        for i in range(num_cases):
            test_cases.append({
                "id": f"custom_{i:03d}",
                "input": self._generate_input(),
                "expected": self._generate_expected_output()
            })
        return test_cases
    
    def _generate_input(self) -> Dict[str, Any]:
        """Generate input data."""
        return {
            "parameter1": "value1",
            "parameter2": "value2",
            # Add your input parameters
        }
    
    def _generate_expected_output(self) -> Dict[str, Any]:
        """Generate expected output."""
        return {
            "result_field": "expected_value",
            # Add expected output structure
        }
    
    def evaluate(self, output: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the output against expected results."""
        scores = {}
        
        # Custom evaluation logic
        if "result_field" in output:
            scores["field_presence"] = 1.0
        else:
            scores["field_presence"] = 0.0
            
        # Add more evaluation metrics
        scores["accuracy"] = self._calculate_accuracy(output, expected)
        scores["completeness"] = self._calculate_completeness(output)
        
        return scores
```

### Step 2: Register Use Case

Add to `src/core/types.py`:

```python
class UseCaseType(str, Enum):
    # Existing use cases
    MOVIE_RECOMMENDATION = "movie_recommendation"
    GITHUB_TRIAGE = "github_triage"
    # ...
    
    # Add your custom use case
    CUSTOM_USE_CASE = "custom_use_case"
```

### Step 3: Add Test Data

Create test data files:

1. `data/custom_use_case/test_cases.json`:
```json
{
  "test_cases": [
    {
      "id": "custom_001",
      "input": {
        "parameter1": "test_value_1",
        "parameter2": "test_value_2"
      }
    }
  ]
}
```

2. `data/custom_use_case/ground_truth.json`:
```json
{
  "custom_001": {
    "expected_output": {
      "result_field": "expected_value",
      "score": 0.95
    },
    "evaluation_criteria": {
      "min_score": 0.8,
      "required_fields": ["result_field"]
    }
  }
}
```

### Step 4: Update Framework Adapters

Modify framework adapters to handle the new use case:

```python
# In src/adapters/framework_adapters.py

def supports_use_case(self, use_case: UseCaseType) -> bool:
    """Check if framework supports use case."""
    supported = [
        UseCaseType.MOVIE_RECOMMENDATION,
        UseCaseType.GITHUB_TRIAGE,
        UseCaseType.CUSTOM_USE_CASE  # Add your use case
    ]
    return use_case in supported

def _execute_custom_use_case(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute custom use case logic."""
    # Implement framework-specific logic
    result = {
        "result_field": self._process_custom_logic(input_data),
        # Add more fields
    }
    return result
```

## Use Case Best Practices

### 1. Define Clear Evaluation Criteria

```python
class EvaluationCriteria:
    """Standard criteria for use case evaluation."""
    
    REQUIRED_FIELDS = ["field1", "field2"]  # Must be present
    OPTIONAL_FIELDS = ["field3", "field4"]  # Nice to have
    
    MIN_ACCURACY = 0.8  # Minimum acceptable accuracy
    MAX_LATENCY = 30.0  # Maximum seconds allowed
    
    SCORING_WEIGHTS = {
        "accuracy": 0.4,
        "completeness": 0.3,
        "latency": 0.2,
        "format": 0.1
    }
```

### 2. Generate Diverse Test Cases

```python
import random
from typing import List, Dict, Any

def generate_diverse_test_cases(num_cases: int) -> List[Dict[str, Any]]:
    """Generate diverse test cases for robust evaluation."""
    test_cases = []
    
    # Vary difficulty levels
    difficulties = ["easy", "medium", "hard"]
    
    # Vary input types
    input_types = ["simple", "complex", "edge_case"]
    
    for i in range(num_cases):
        difficulty = random.choice(difficulties)
        input_type = random.choice(input_types)
        
        test_case = {
            "id": f"test_{i:03d}",
            "difficulty": difficulty,
            "type": input_type,
            "input": generate_input_for_type(input_type, difficulty),
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "category": input_type
            }
        }
        test_cases.append(test_case)
    
    return test_cases
```

### 3. Implement Robust Evaluation

```python
from typing import Dict, Any
import numpy as np

class RobustEvaluator:
    """Robust evaluation with multiple metrics."""
    
    def evaluate(self, output: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Comprehensive evaluation."""
        metrics = {}
        
        # 1. Structural validity
        metrics["structural_validity"] = self._check_structure(output)
        
        # 2. Semantic accuracy
        metrics["semantic_accuracy"] = self._calculate_semantic_similarity(
            output, ground_truth
        )
        
        # 3. Completeness
        metrics["completeness"] = self._calculate_completeness(output, ground_truth)
        
        # 4. Consistency
        metrics["consistency"] = self._check_consistency(output)
        
        # 5. Performance
        metrics["performance_score"] = self._evaluate_performance(output)
        
        # Calculate weighted final score
        weights = [0.3, 0.3, 0.2, 0.1, 0.1]
        scores = [metrics[k] for k in metrics.keys()]
        metrics["final_score"] = np.average(scores, weights=weights)
        
        return metrics
```

### 4. Handle Edge Cases

```python
class EdgeCaseHandler:
    """Handle edge cases in evaluation."""
    
    EDGE_CASES = {
        "empty_input": {},
        "null_values": {"field": None},
        "very_long_input": {"text": "x" * 10000},
        "special_characters": {"text": "!@#$%^&*()"},
        "unicode": {"text": "你好世界 🌍"},
        "malformed": {"field": "[INVALID]"}
    }
    
    def test_edge_cases(self, evaluator):
        """Test evaluator with edge cases."""
        results = {}
        
        for case_name, case_input in self.EDGE_CASES.items():
            try:
                output = evaluator.process(case_input)
                results[case_name] = "passed"
            except Exception as e:
                results[case_name] = f"failed: {str(e)}"
        
        return results
```

## Loading Data from APIs

### Example: Loading GitHub Issues

```python
import requests
from typing import List, Dict, Any

class GitHubDataLoader:
    """Load real GitHub issues for testing."""
    
    def __init__(self, token: str):
        self.token = token
        self.headers = {"Authorization": f"token {token}"}
    
    def load_issues(self, repo: str, count: int = 10) -> List[Dict[str, Any]]:
        """Load issues from GitHub repository."""
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {"state": "open", "per_page": count}
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        
        issues = []
        for issue in response.json():
            issues.append({
                "id": issue["number"],
                "title": issue["title"],
                "body": issue["body"],
                "labels": [l["name"] for l in issue["labels"]],
                "created_at": issue["created_at"]
            })
        
        return issues
```

### Example: Loading from Database

```python
import sqlite3
from typing import List, Dict, Any

class DatabaseLoader:
    """Load test cases from database."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def load_test_cases(self, use_case: str) -> List[Dict[str, Any]]:
        """Load test cases from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT id, input_json, expected_json
            FROM test_cases
            WHERE use_case = ?
        """
        
        cursor.execute(query, (use_case,))
        
        test_cases = []
        for row in cursor.fetchall():
            test_cases.append({
                "id": row[0],
                "input": json.loads(row[1]),
                "expected": json.loads(row[2])
            })
        
        conn.close()
        return test_cases
```

## Validation and Quality Assurance

### Test Case Validation

```python
from jsonschema import validate
from typing import Dict, Any

class TestCaseValidator:
    """Validate test cases against schema."""
    
    SCHEMA = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "input": {"type": "object"},
            "expected": {"type": "object"}
        },
        "required": ["id", "input"]
    }
    
    def validate_test_case(self, test_case: Dict[str, Any]) -> bool:
        """Validate single test case."""
        try:
            validate(test_case, self.SCHEMA)
            return True
        except Exception as e:
            print(f"Validation failed: {e}")
            return False
    
    def validate_all(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate all test cases."""
        results = {
            "valid": 0,
            "invalid": 0,
            "errors": []
        }
        
        for test_case in test_cases:
            if self.validate_test_case(test_case):
                results["valid"] += 1
            else:
                results["invalid"] += 1
                results["errors"].append(test_case["id"])
        
        return results
```

## Next Steps

- [Add new frameworks](./frameworks.md)
- [Configure evaluation metrics](./evaluation.md)
- [Set up production testing](./deployment.md)
