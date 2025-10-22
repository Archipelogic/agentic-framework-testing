"""
Recipe generation use case definition.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class RecipeGenerationInput:
    """Input for recipe generation use case."""
    ingredients: List[str]  # Available ingredients
    dietary_constraints: List[str]  # vegetarian, vegan, gluten-free, etc.
    cuisine: Optional[str] = None  # Italian, Mexican, Chinese, etc.
    difficulty_level: Optional[str] = "medium"  # easy, medium, hard
    max_prep_time_mins: Optional[int] = 60


@dataclass
class RecipeGenerationOutput:
    """Output for recipe generation use case."""
    recipe: Dict[str, Any]  # Full recipe with name, instructions, prep time, etc.
    ingredients_used: List[str]  # Which input ingredients were used
    additional_ingredients: List[str]  # Any ingredients needed beyond input
    nutritional_info: Optional[Dict[str, Any]] = None  # Calories, protein, etc.
    difficulty_score: Optional[float] = None


@dataclass
class RecipeGenerationGroundTruth:
    """Ground truth for recipe generation evaluation."""
    expected_recipe_type: str  # Type/category of recipe
    must_use_ingredients: List[str]  # Ingredients that should be used
    constraints_satisfied: List[str]  # Which constraints should be satisfied
    nutritional_targets: Optional[Dict[str, Any]] = None


# Expected agents for this use case
EXPECTED_AGENTS = [
    "Ingredient Analyzer",
    "Recipe Database Agent",
    "Recipe Generator",
    "Nutrition Calculator"
]

# Expected tools for this use case
EXPECTED_TOOLS = [
    "analyze_ingredients",
    "search_recipes",
    "generate_recipe",
    "calculate_nutrition"
]
