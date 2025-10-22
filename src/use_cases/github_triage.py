"""
GitHub triage use case definition.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class GitHubTriageInput:
    """Input for GitHub triage use case."""
    issue_id: int
    title: str
    body: str
    author: str
    created_at: Optional[str] = None
    labels_current: Optional[List[str]] = None


@dataclass
class GitHubTriageOutput:
    """Output for GitHub triage use case."""
    classification: Dict[str, Any]  # type (bug/feature/question/docs), confidence
    priority: Dict[str, Any]  # level (P0-P3), reasoning, confidence
    routing: Optional[Dict[str, Any]] = None  # team assignment and reasoning
    labels: List[str] = None  # Suggested labels


@dataclass
class GitHubTriageGroundTruth:
    """Ground truth for GitHub triage evaluation."""
    expected_classification: str  # bug, feature, question, or docs
    expected_priority: str  # P0, P1, P2, or P3
    expected_team: Optional[str] = None
    expected_labels: List[str] = None


# Expected agents for this use case
EXPECTED_AGENTS = [
    "Issue Parser",
    "Classification Agent",
    "Priority Scorer"
]

# Expected tools for this use case
EXPECTED_TOOLS = [
    "parse_text",
    "classify_issue",
    "score_priority"
]
