"""
Mock adapters for all frameworks - allows testing without API keys.
"""

import time
import random
from typing import Dict, Any, List, Optional
from dataclasses import field

from src.core.types import (
    UseCaseType,
    FrameworkType,
    FrameworkResult,
    FrameworkAdapter,
    AgentAction,
    AgentActionType,
    TokenUsage,
    CostBreakdown
)


class MockFrameworkAdapter(FrameworkAdapter):
    """
    Base mock adapter that simulates framework behavior without API calls.
    """
    
    def __init__(self, framework_type: FrameworkType, version: str, config: Optional[Dict[str, Any]] = None):
        self.framework_type = framework_type
        self.version = version
        self.config = config or {}
        self.supported_use_cases = self._get_supported_use_cases()
        
    def get_framework_name(self) -> FrameworkType:
        return self.framework_type
    
    def get_framework_version(self) -> str:
        return self.version
    
    def supports_use_case(self, use_case: UseCaseType) -> bool:
        return use_case in self.supported_use_cases
    
    def _get_supported_use_cases(self) -> set:
        """Define which use cases each framework supports."""
        all_use_cases = {
            UseCaseType.MOVIE_RECOMMENDATION,
            UseCaseType.GITHUB_TRIAGE,
            UseCaseType.RECIPE_GENERATION,
            UseCaseType.RESEARCH_SUMMARY,
            UseCaseType.EMAIL_AUTOMATION
        }
        
        # Some frameworks have limitations
        if self.framework_type == FrameworkType.DSPY:
            # DSPy is best for single-agent optimization
            return {UseCaseType.GITHUB_TRIAGE, UseCaseType.RECIPE_GENERATION}
        elif self.framework_type == FrameworkType.SMOLAGENTS:
            # Smolagents is minimalist, best for simple tasks
            return {UseCaseType.GITHUB_TRIAGE, UseCaseType.RECIPE_GENERATION}
        elif self.framework_type == FrameworkType.ATOMIC_AGENTS:
            # Atomic Agents good for structured tasks
            return {UseCaseType.GITHUB_TRIAGE, UseCaseType.RECIPE_GENERATION, UseCaseType.EMAIL_AUTOMATION}
        elif self.framework_type == FrameworkType.HAYSTACK:
            # Haystack excellent for document-heavy tasks
            return {UseCaseType.RESEARCH_SUMMARY, UseCaseType.MOVIE_RECOMMENDATION, UseCaseType.RECIPE_GENERATION}
        else:
            # Most frameworks support all use cases
            return all_use_cases
    
    def run(self, use_case: UseCaseType, input_data: Dict[str, Any]) -> FrameworkResult:
        """Simulate framework execution with realistic metrics."""
        
        if not self.supports_use_case(use_case):
            return FrameworkResult(
                result={},
                succeeded=False,
                errors=[f"{self.framework_type.value} does not support {use_case.value}"]
            )
        
        # Simulate processing time (varies by framework)
        base_latency = self._get_base_latency()
        latency = base_latency + random.uniform(0, 0.5)
        time.sleep(latency / 100)  # Small actual delay for realism
        
        # Generate trajectory based on framework style
        trajectory = self._generate_trajectory(use_case)
        
        # Generate result based on use case
        result = self._generate_result(use_case, input_data)
        
        # Calculate tokens based on framework efficiency
        tokens = self._calculate_tokens(use_case, input_data, result)
        
        # Calculate cost
        cost = self._calculate_cost(tokens)
        
        # Simulate success/failure based on framework characteristics
        succeeded = self._should_succeed(use_case)
        
        return FrameworkResult(
            result=result,
            succeeded=succeeded,
            trajectory=trajectory,
            latency_seconds=latency,
            tokens=tokens,
            cost=cost,
            metadata={
                'framework': self.framework_type.value,
                'version': self.version,
                'mock': True
            }
        )
    
    def _should_succeed(self, use_case: UseCaseType) -> bool:
        """Simulate success/failure based on framework characteristics."""
        # Most frameworks have high success rates
        success_rates = {
            FrameworkType.LANGGRAPH: 0.95,
            FrameworkType.CREWAI: 0.92,
            FrameworkType.AUTOGEN: 0.90,
            FrameworkType.PYDANTIC_AI: 0.93,
            FrameworkType.HAYSTACK: 0.85,  # Complex pipelines can fail
            FrameworkType.LLAMAINDEX: 0.88,
            FrameworkType.DSPY: 0.70,  # Experimental framework
            FrameworkType.AGNO: 0.91,
            FrameworkType.BEEAI: 0.89,
            FrameworkType.STRANDS_AGENTS: 0.87,
            FrameworkType.BEDROCK_AGENTCORE: 0.88,
            FrameworkType.SMOLAGENTS: 0.65,  # Minimalist = more failures
            FrameworkType.ATOMIC_AGENTS: 0.85
        }
        
        # Some use cases are harder
        use_case_difficulty = {
            UseCaseType.RESEARCH_SUMMARY: 0.9,  # Harder task
            UseCaseType.EMAIL_AUTOMATION: 0.95,
            UseCaseType.GITHUB_TRIAGE: 1.0,
            UseCaseType.MOVIE_RECOMMENDATION: 1.0,
            UseCaseType.RECIPE_GENERATION: 0.98
        }
        
        base_rate = success_rates.get(self.framework_type, 0.9)
        difficulty = use_case_difficulty.get(use_case, 1.0)
        
        # Calculate final success probability
        success_probability = base_rate * difficulty
        
        # Random decision based on probability
        return random.random() < success_probability
    
    def _get_base_latency(self) -> float:
        """Get base latency for framework (seconds)."""
        latencies = {
            FrameworkType.LANGGRAPH: 1.2,  # Graph traversal overhead
            FrameworkType.CREWAI: 0.8,     # Fast role-based
            FrameworkType.AUTOGEN: 1.5,    # Conversational overhead
            FrameworkType.PYDANTIC_AI: 0.6,  # Type validation is fast
            FrameworkType.HAYSTACK: 1.0,   # Pipeline execution
            FrameworkType.LLAMAINDEX: 1.1,  # Document processing
            FrameworkType.DSPY: 0.5,       # Optimized prompts
            FrameworkType.AGNO: 0.3,       # Performance optimized
            FrameworkType.BEEAI: 0.9,      # Enterprise grade
            FrameworkType.STRANDS_AGENTS: 1.0,  # AWS overhead
            FrameworkType.BEDROCK_AGENTCORE: 1.3,  # Infrastructure layer
            FrameworkType.SMOLAGENTS: 0.4,  # Minimalist
            FrameworkType.ATOMIC_AGENTS: 0.7  # Modular
        }
        return latencies.get(self.framework_type, 1.0)
    
    def _generate_trajectory(self, use_case: UseCaseType) -> List[AgentAction]:
        """Generate a mock trajectory."""
        trajectories = {
            UseCaseType.MOVIE_RECOMMENDATION: [
                AgentAction(
                    action_type=AgentActionType.DECISION,
                    agent_name="Planner",
                    decision="Planning: First analyze user preferences, then search for matching movies, finally rank and filter results"
                ),
                AgentAction(
                    action_type=AgentActionType.LLM_CALL,
                    agent_name="PreferenceAnalyzer",
                    llm_input="Analyze user preferences: sci-fi, thriller",
                    llm_output="Analysis: User prefers highly-rated science fiction and thriller movies"
                ),
                AgentAction(
                    action_type=AgentActionType.TOOL_CALL,
                    agent_name="MovieSearcher",
                    tool_name="search_movies",
                    tool_input={"query": "top rated sci-fi thriller movies"},
                    tool_output="Found 15 movies matching criteria with ratings above 4.0"
                ),
                AgentAction(
                    action_type=AgentActionType.DECISION,
                    agent_name="Ranker",
                    decision="Selected top 5 movies based on ratings and relevance"
                ),
            ],
            UseCaseType.GITHUB_TRIAGE: [
                AgentAction(
                    action_type=AgentActionType.DECISION,
                    agent_name="Planner",
                    decision="Planning: First analyze issue description, then classify and prioritize, finally assign to team"
                ),
                AgentAction(
                    action_type=AgentActionType.LLM_CALL,
                    agent_name="IssueAnalyzer",
                    llm_input="Analyze issue description",
                    llm_output="Analysis: Issue is a bug with high severity"
                ),
                AgentAction(
                    action_type=AgentActionType.DECISION,
                    agent_name="Classifier",
                    decision="Classification: Bug with high severity"
                ),
                AgentAction(
                    action_type=AgentActionType.DECISION,
                    agent_name="Prioritizer",
                    decision="Prioritization: High priority due to severity"
                ),
                AgentAction(
                    action_type=AgentActionType.HANDOFF,
                    agent_name="Assigner",
                    decision="Assignment: Assigned to backend team"
                ),
            ],
            UseCaseType.RECIPE_GENERATION: [
                AgentAction(
                    action_type=AgentActionType.DECISION,
                    agent_name="Planner",
                    decision="Planning: First analyze ingredients, then generate recipe, finally provide nutritional information"
                ),
                AgentAction(
                    action_type=AgentActionType.LLM_CALL,
                    agent_name="IngredientAnalyzer",
                    llm_input="Analyze ingredients: chicken, rice",
                    llm_output="Analysis: Ingredients are chicken and rice"
                ),
                AgentAction(
                    action_type=AgentActionType.LLM_CALL,
                    agent_name="RecipeGenerator",
                    llm_input="Generate chicken and rice recipe",
                    llm_output="Generation: Recipe generated with cooking instructions"
                ),
                AgentAction(
                    action_type=AgentActionType.TOOL_CALL,
                    agent_name="NutritionCalculator",
                    tool_name="calculate_nutrition",
                    tool_input={"recipe": "chicken and rice"},
                    tool_output="Nutrition: Provided nutritional information for recipe"
                ),
            ],
            UseCaseType.RESEARCH_SUMMARY: [
                AgentAction(
                    action_type=AgentActionType.DECISION,
                    agent_name="Planner",
                    decision="Planning: First analyze papers, then summarize and provide key themes"
                ),
                AgentAction(
                    action_type=AgentActionType.LLM_CALL,
                    agent_name="PaperAnalyzer",
                    llm_input="Analyze papers about transformer architectures",
                    llm_output="Analysis: Papers are about transformer architectures"
                ),
                AgentAction(
                    action_type=AgentActionType.LLM_CALL,
                    agent_name="Summarizer",
                    llm_input="Summarize papers",
                    llm_output="Summary: Provided summary of papers"
                ),
                AgentAction(
                    action_type=AgentActionType.DECISION,
                    agent_name="ThemeExtractor",
                    decision="Key Themes: self-attention mechanisms, scaling laws"
                ),
            ],
            UseCaseType.EMAIL_AUTOMATION: [
                AgentAction(
                    action_type=AgentActionType.DECISION,
                    agent_name="Planner",
                    decision="Planning: First classify emails, then respond and provide template"
                ),
                AgentAction(
                    action_type=AgentActionType.LLM_CALL,
                    agent_name="EmailClassifier",
                    llm_input="Classify emails",
                    llm_output="Classification: Emails are urgent"
                ),
                AgentAction(
                    action_type=AgentActionType.LLM_CALL,
                    agent_name="EmailResponder",
                    llm_input="Generate response to urgent emails",
                    llm_output="Response: Responded to emails with template"
                ),
                AgentAction(
                    action_type=AgentActionType.DECISION,
                    agent_name="TemplateProvider",
                    decision="Template: Provided template for response"
                ),
            ],
        }
        
        if self.framework_type == FrameworkType.LANGGRAPH:
            # Graph-based: explicit nodes
            trajectory = [
                AgentAction(
                    action_type=AgentActionType.DECISION,
                    agent_name="InputAnalyzer",
                    decision="Parsing input requirements"
                ),
                AgentAction(
                    action_type=AgentActionType.TOOL_CALL,
                    agent_name="DataRetriever",
                    tool_name="search_database",
                    tool_input={"query": "relevant_data"}
                ),
                AgentAction(
                    action_type=AgentActionType.DECISION,
                    agent_name="ResultProcessor",
                    decision="Formatting output"
                )
            ]
        elif self.framework_type == FrameworkType.CREWAI:
            # Role-based: manager and workers
            trajectory = [
                AgentAction(
                    action_type=AgentActionType.DECISION,
                    agent_name="Manager",
                    decision="Assigning tasks to crew"
                ),
                AgentAction(
                    action_type=AgentActionType.LLM_CALL,
                    agent_name="Specialist1",
                    llm_input="Analyze requirements",
                    llm_output="Analysis complete"
                ),
                AgentAction(
                    action_type=AgentActionType.HANDOFF,
                    agent_name="Manager",
                    decision="Task completed"
                )
            ]
        elif self.framework_type == FrameworkType.AUTOGEN:
            # Conversational: back-and-forth
            trajectory = [
                AgentAction(
                    action_type=AgentActionType.LLM_CALL,
                    agent_name="AssistantAgent",
                    llm_input="Processing request",
                    llm_output="Initial analysis"
                ),
                AgentAction(
                    action_type=AgentActionType.LLM_CALL,
                    agent_name="UserProxyAgent",
                    llm_input="Verification needed",
                    llm_output="Confirmed"
                ),
                AgentAction(
                    action_type=AgentActionType.DECISION,
                    agent_name="AssistantAgent",
                    decision="Final response prepared"
                )
            ]
        else:
            # Generic trajectory
            trajectory = [
                AgentAction(
                    action_type=AgentActionType.LLM_CALL,
                    agent_name="MainAgent",
                    llm_input="Processing input",
                    llm_output="Generated output"
                )
            ]
        
        return trajectory
    
    def _generate_result(self, use_case: UseCaseType, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate realistic result based on use case."""
        
        if use_case == UseCaseType.MOVIE_RECOMMENDATION:
            # Add framework-specific variations
            base_movies = [
                "Blade Runner 2049", "Ex Machina", "The Martian",
                "Arrival", "Dune", "Tenet", "The Prestige"
            ]
            
            # Shuffle based on framework for variety
            random.seed(hash(self.framework_type.value))
            random.shuffle(base_movies)
            
            return {
                'recommendations': [
                    {
                        'movie_id': 100 + i,
                        'title': movie,
                        'genres': ['Sci-Fi', 'Drama'] if i % 2 == 0 else ['Sci-Fi', 'Thriller'],
                        'reasoning': f"Based on your love of {input_data.get('watch_history', [{}])[0].get('title', 'sci-fi')}",
                        'confidence': 0.85 + (i * 0.02)
                    }
                    for i, movie in enumerate(base_movies[:5])
                ],
                'user_profile_summary': {
                    'favorite_genres': ['Sci-Fi', 'Action'],
                    'avg_rating': 4.5
                }
            }
        
        elif use_case == UseCaseType.GITHUB_TRIAGE:
            severity = 'high' if 'crash' in str(input_data).lower() else 'medium'
            return {
                'classification': {
                    'type': 'bug' if 'bug' in str(input_data).lower() else 'feature',
                    'confidence': 0.85 + random.uniform(0, 0.1)
                },
                'priority': {
                    'level': 'P0' if severity == 'high' else 'P1',
                    'reasoning': f"{'Critical issue affecting stability' if severity == 'high' else 'Important feature request'}",
                    'confidence': 0.9
                },
                'routing': {
                    'team': 'backend' if 'database' in str(input_data).lower() else 'frontend',
                    'reasoning': 'Based on issue description',
                    'confidence': 0.85
                },
                'labels': ['bug', 'needs-triage', severity]
            }
        
        elif use_case == UseCaseType.RECIPE_GENERATION:
            ingredients = input_data.get('ingredients', ['chicken', 'rice'])
            return {
                'recipe': {
                    'name': f"{ingredients[0].title()} Delight",
                    'prep_time_mins': 30,
                    'cook_time_mins': 25,
                    'servings': 4,
                    'instructions': [
                        f"Prepare {ingredients[0]}",
                        "Season with spices",
                        "Cook until golden",
                        "Serve hot"
                    ]
                },
                'ingredients_used': ingredients[:3],
                'additional_ingredients': ['salt', 'pepper', 'olive oil'],
                'nutritional_info': {
                    'calories': 350,
                    'protein_g': 25,
                    'carbs_g': 30,
                    'fat_g': 15
                }
            }
        
        elif use_case == UseCaseType.RESEARCH_SUMMARY:
            return {
                'summary': 'This comprehensive analysis examines the evolution of transformer architectures...',
                'key_themes': [
                    {
                        'theme': 'Self-Attention Mechanisms',
                        'papers': ['paper1', 'paper2'],
                        'summary': 'Revolutionary approach to sequence modeling'
                    },
                    {
                        'theme': 'Scaling Laws',
                        'papers': ['paper3'],
                        'summary': 'Bigger models show emergent capabilities'
                    }
                ],
                'citation_network': {
                    'nodes': ['paper1', 'paper2', 'paper3'],
                    'edges': [('paper1', 'paper2'), ('paper2', 'paper3')],
                    'central_papers': ['paper1']
                }
            }
        
        elif use_case == UseCaseType.EMAIL_AUTOMATION:
            emails = input_data.get('emails', [{}])
            return {
                'classifications': [
                    {
                        'email_id': i,
                        'category': 'urgent' if i == 0 else 'normal',
                        'priority': 'high' if i == 0 else 'medium'
                    }
                    for i in range(len(emails))
                ],
                'responses': [
                    {
                        'email_id': i,
                        'response': 'Thank you for your email. I will review and respond shortly.'
                    }
                    for i in range(len(emails))
                ],
                'template_selected': 'professional_acknowledgment'
            }
        
        return {}
    
    def _calculate_tokens(self, use_case: UseCaseType, input_data: Dict[str, Any], result: Dict[str, Any]) -> TokenUsage:
        """Calculate realistic token usage based on framework efficiency."""
        
        # Base tokens from input/output size
        input_text = str(input_data)
        output_text = str(result)
        base_input = len(input_text) // 4
        base_output = len(output_text) // 4
        
        # Framework efficiency multipliers
        efficiency = {
            FrameworkType.LANGGRAPH: 1.1,      # Some overhead from graph
            FrameworkType.CREWAI: 1.2,         # Multiple agents
            FrameworkType.AUTOGEN: 1.3,        # Conversational overhead
            FrameworkType.PYDANTIC_AI: 0.9,    # Efficient validation
            FrameworkType.HAYSTACK: 1.0,       # Standard
            FrameworkType.LLAMAINDEX: 1.1,     # Document processing
            FrameworkType.DSPY: 0.8,           # Optimized prompts
            FrameworkType.AGNO: 0.85,          # Performance optimized
            FrameworkType.BEEAI: 1.0,          # Standard
            FrameworkType.STRANDS_AGENTS: 1.05,  # Slight AWS overhead
            FrameworkType.BEDROCK_AGENTCORE: 1.15,  # Infrastructure overhead
            FrameworkType.SMOLAGENTS: 0.7,     # Minimalist
            FrameworkType.ATOMIC_AGENTS: 0.95  # Efficient modular
        }
        
        multiplier = efficiency.get(self.framework_type, 1.0)
        
        input_tokens = int(base_input * multiplier)
        output_tokens = int(base_output * multiplier)
        
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )
    
    def _calculate_cost(self, tokens: TokenUsage) -> CostBreakdown:
        """Calculate cost based on framework's typical model usage."""
        
        # Model selection by framework
        model_map = {
            FrameworkType.LANGGRAPH: "gpt-4o-mini",
            FrameworkType.CREWAI: "gpt-4o-mini",
            FrameworkType.AUTOGEN: "gpt-4",
            FrameworkType.PYDANTIC_AI: "gpt-3.5-turbo",
            FrameworkType.HAYSTACK: "gpt-4o-mini",
            FrameworkType.LLAMAINDEX: "gpt-4o-mini",
            FrameworkType.DSPY: "gpt-3.5-turbo",
            FrameworkType.AGNO: "gpt-3.5-turbo",
            FrameworkType.BEEAI: "gpt-4",
            FrameworkType.STRANDS_AGENTS: "claude-3-5-haiku",
            FrameworkType.BEDROCK_AGENTCORE: "claude-3-5-sonnet",
            FrameworkType.SMOLAGENTS: "gpt-3.5-turbo",
            FrameworkType.ATOMIC_AGENTS: "gpt-3.5-turbo"
        }
        
        model = model_map.get(self.framework_type, "gpt-4o-mini")
        
        # Pricing per 1M tokens
        pricing = {
            "gpt-4o-mini": {"input": 0.150, "output": 0.600},
            "gpt-4": {"input": 30.00, "output": 60.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
            "claude-3-5-haiku": {"input": 0.80, "output": 4.00}
        }
        
        model_pricing = pricing.get(model, pricing["gpt-4o-mini"])
        
        input_cost = tokens.input_tokens * model_pricing["input"] / 1_000_000
        output_cost = tokens.output_tokens * model_pricing["output"] / 1_000_000
        
        return CostBreakdown(
            total_usd=input_cost + output_cost,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            model_name=model
        )


# Create mock adapters for all frameworks
def create_mock_adapter(framework: FrameworkType, config: Optional[Dict[str, Any]] = None) -> MockFrameworkAdapter:
    """Factory function to create mock adapter for any framework."""
    
    versions = {
        FrameworkType.LANGGRAPH: "0.2.0",
        FrameworkType.CREWAI: "0.80.0",
        FrameworkType.AUTOGEN: "0.2.35",
        FrameworkType.PYDANTIC_AI: "0.0.14",
        FrameworkType.HAYSTACK: "2.0.0",
        FrameworkType.LLAMAINDEX: "0.11.0",
        FrameworkType.DSPY: "2.5.0",
        FrameworkType.AGNO: "2.0.0",
        FrameworkType.BEEAI: "1.0.0",
        FrameworkType.STRANDS_AGENTS: "1.5.0",
        FrameworkType.BEDROCK_AGENTCORE: "1.0.0",
        FrameworkType.SMOLAGENTS: "0.1.0",
        FrameworkType.ATOMIC_AGENTS: "0.5.0"
    }
    
    return MockFrameworkAdapter(
        framework_type=framework,
        version=versions.get(framework, "1.0.0"),
        config=config
    )


def create_all_mock_adapters() -> Dict[str, MockFrameworkAdapter]:
    """Create mock adapters for all frameworks."""
    adapters = {}
    
    for framework in FrameworkType:
        adapters[framework.value] = create_mock_adapter(framework)
    
    return adapters
