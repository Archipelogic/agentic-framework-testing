"""
Movie recommendation use case definition.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class MovieRecommendationInput:
    """Input for movie recommendation use case."""
    user_id: int
    watch_history: List[Dict[str, Any]]  # List of movie objects with id, title, rating
    constraints: Optional[Dict[str, Any]] = None  # Optional genre exclusions, year range, etc.
    num_recommendations: int = 5


@dataclass
class MovieRecommendationOutput:
    """Output for movie recommendation use case."""
    recommendations: List[Dict[str, Any]]  # List of recommended movies with id, title, reason
    user_profile_summary: Optional[Dict[str, Any]] = None


@dataclass
class MovieRecommendationGroundTruth:
    """Ground truth for movie recommendation evaluation."""
    held_out_movies: List[int]  # Movie IDs that should be recommended
    expected_genres: List[str]
    user_preferences: Dict[str, Any]


# Expected agents for this use case
EXPECTED_AGENTS = [
    "User Profile Analyzer",
    "Content Retriever", 
    "Collaborative Filtering Agent",
    "Ranking & Personalization Agent"
]

# Expected tools for this use case
EXPECTED_TOOLS = [
    "get_user_history",
    "search_movies",
    "apply_collaborative_filtering",
    "rank_recommendations"
]
