"""
Use case definitions for the agentic framework testing harness.
"""

from .movie_recommendation import (
    MovieRecommendationInput,
    MovieRecommendationOutput,
    MovieRecommendationGroundTruth,
    EXPECTED_AGENTS as MOVIE_AGENTS,
    EXPECTED_TOOLS as MOVIE_TOOLS
)

from .github_triage import (
    GitHubTriageInput,
    GitHubTriageOutput,
    GitHubTriageGroundTruth,
    EXPECTED_AGENTS as GITHUB_AGENTS,
    EXPECTED_TOOLS as GITHUB_TOOLS
)

from .recipe_generation import (
    RecipeGenerationInput,
    RecipeGenerationOutput,
    RecipeGenerationGroundTruth,
    EXPECTED_AGENTS as RECIPE_AGENTS,
    EXPECTED_TOOLS as RECIPE_TOOLS
)

from .research_summary import (
    ResearchSummaryInput,
    ResearchSummaryOutput,
    ResearchSummaryGroundTruth,
    EXPECTED_AGENTS as RESEARCH_AGENTS,
    EXPECTED_TOOLS as RESEARCH_TOOLS
)

from .email_automation import (
    EmailAutomationInput,
    EmailAutomationOutput,
    EmailAutomationGroundTruth,
    EXPECTED_AGENTS as EMAIL_AGENTS,
    EXPECTED_TOOLS as EMAIL_TOOLS
)

__all__ = [
    # Movie Recommendation
    'MovieRecommendationInput',
    'MovieRecommendationOutput',
    'MovieRecommendationGroundTruth',
    'MOVIE_AGENTS',
    'MOVIE_TOOLS',
    
    # GitHub Triage
    'GitHubTriageInput',
    'GitHubTriageOutput',
    'GitHubTriageGroundTruth',
    'GITHUB_AGENTS',
    'GITHUB_TOOLS',
    
    # Recipe Generation
    'RecipeGenerationInput',
    'RecipeGenerationOutput',
    'RecipeGenerationGroundTruth',
    'RECIPE_AGENTS',
    'RECIPE_TOOLS',
    
    # Research Summary
    'ResearchSummaryInput',
    'ResearchSummaryOutput',
    'ResearchSummaryGroundTruth',
    'RESEARCH_AGENTS',
    'RESEARCH_TOOLS',
    
    # Email Automation
    'EmailAutomationInput',
    'EmailAutomationOutput',
    'EmailAutomationGroundTruth',
    'EMAIL_AGENTS',
    'EMAIL_TOOLS'
]