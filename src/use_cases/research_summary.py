"""
Research summary use case definition.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ResearchSummaryInput:
    """Input for research summary use case."""
    papers: List[Dict[str, Any]]  # List of paper objects with id, title, abstract, year
    review_focus: str  # What aspect to focus the review on
    num_papers: Optional[int] = None  # Limit number of papers to include
    word_count: Optional[int] = 1000  # Target word count for summary
    include_sections: Optional[List[str]] = None  # Introduction, Methods, Results, etc.


@dataclass
class ResearchSummaryOutput:
    """Output for research summary use case."""
    summary: str  # Main literature review text
    key_themes: List[Dict[str, Any]]  # Themes with associated papers
    citation_network: Optional[Dict[str, Any]] = None  # How papers relate to each other
    paper_summaries: Optional[List[Dict[str, Any]]] = None  # Individual paper summaries
    research_gaps: Optional[List[str]] = None  # Identified gaps in the research


@dataclass
class ResearchSummaryGroundTruth:
    """Ground truth for research summary evaluation."""
    expected_themes: List[str]  # Key themes that should be identified
    key_papers: List[str]  # Paper IDs that are most important
    expected_citations: List[tuple]  # Expected citation relationships (paper_a, paper_b)
    summary_quality_score: Optional[float] = None  # Human-rated quality 1-5


# Expected agents for this use case
EXPECTED_AGENTS = [
    "Document Parser",
    "Citation Network Builder",
    "Theme Extractor",
    "Synthesis Agent"
]

# Expected tools for this use case
EXPECTED_TOOLS = [
    "parse_papers",
    "build_citation_network",
    "extract_themes",
    "synthesize_summary"
]
