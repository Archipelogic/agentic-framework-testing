"""
Email automation use case definition.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class EmailAutomationInput:
    """Input for email automation use case."""
    emails: List[Dict[str, Any]]  # List of email objects with subject, body, sender, timestamp
    templates: List[str]  # Available response templates
    user_context: Optional[Dict[str, Any]] = None  # User preferences, OOO status, etc.
    auto_respond: bool = False  # Whether to auto-send responses
    classification_rules: Optional[Dict[str, Any]] = None  # Custom classification rules


@dataclass
class EmailAutomationOutput:
    """Output for email automation use case."""
    classifications: List[Dict[str, Any]]  # Category, priority, action required for each email
    responses: List[Dict[str, Any]]  # Generated response for each email
    template_selected: Optional[str] = None  # Which template was used
    actions_taken: List[str] = None  # Archive, flag, forward, etc.
    summary: Optional[str] = None  # Overall summary of email batch


@dataclass
class EmailAutomationGroundTruth:
    """Ground truth for email automation evaluation."""
    expected_classifications: List[str]  # Expected category for each email
    expected_priorities: List[str]  # Expected priority for each email
    expected_template: Optional[str] = None  # Expected template selection
    response_quality_scores: Optional[List[float]] = None  # Quality score for each response


# Expected agents for this use case
EXPECTED_AGENTS = [
    "Email Classifier",
    "Template Selector",
    "Response Generator",
    "Tone Adjuster"
]

# Expected tools for this use case
EXPECTED_TOOLS = [
    "classify_email",
    "select_template",
    "generate_response",
    "adjust_tone"
]
