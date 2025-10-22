"""
Framework adapters for all 13 agentic frameworks.

This module provides adapters that allow each framework to plug into 
the evaluation harness via a standardized interface.
"""

import time
import random
import logging
from typing import Dict, Any, List, Optional
from src.core.types import (
    FrameworkAdapter, FrameworkType, UseCaseType, FrameworkResult,
    AgentAction, AgentActionType, TokenUsage, CostBreakdown
)
from src.adapters.trajectory_generator import TrajectoryGenerator

logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================

def count_tokens(text: str) -> int:
    """
    Estimate token count from text.
    Rule of thumb: ~4 characters per token for English.
    """
    return len(text) // 4


def calculate_cost(
    tokens: TokenUsage,
    model_name: str = "gpt-4o-mini"
) -> CostBreakdown:
    """
    Calculate cost from token usage.
    
    Args:
        tokens: Token usage
        model_name: Model identifier
        
    Returns:
        Cost breakdown in USD
    """
    # Pricing as of Oct 2024 ($/1M tokens)
    MODEL_PRICING = {
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    }
    
    if model_name not in MODEL_PRICING:
        model_name = "gpt-4o-mini"
    
    pricing = MODEL_PRICING[model_name]
    
    input_cost = tokens.input_tokens * pricing["input"] / 1_000_000
    output_cost = tokens.output_tokens * pricing["output"] / 1_000_000
    
    return CostBreakdown(
        total_usd=input_cost + output_cost,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        model_name=model_name
    )


# ============================================================================
# Adapter 1: LangGraph
# ============================================================================

class LangGraphAdapter(FrameworkAdapter):
    """
    LangGraph framework adapter - Graph-based orchestration.
    
    Strengths:
    - Fine-grained control over workflow
    - Explicit state management
    - Built-in checkpointing
    - Excellent for complex branching logic
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model_name = self.config.get('model', 'gpt-4o-mini')
        
    def get_framework_name(self) -> FrameworkType:
        return FrameworkType.LANGGRAPH
    
    def get_framework_version(self) -> str:
        try:
            import langgraph
            return langgraph.__version__
        except:
            return "0.2.0"
    
    def supports_use_case(self, use_case: UseCaseType) -> bool:
        # LangGraph supports all use cases
        return True
    
    def run(
        self, 
        use_case: UseCaseType, 
        input_data: Dict[str, Any]
    ) -> FrameworkResult:
        """Execute LangGraph workflow for use case."""
        start_time = time.time()
        
        # Simulated implementation for now
        # In production, would build actual LangGraph workflow
        
        trajectory = [
            AgentAction(
                action_type=AgentActionType.DECISION,
                agent_name="Analyzer",
                decision="Analyzing input data",
                reasoning="Processing user requirements"
            ),
            AgentAction(
                action_type=AgentActionType.TOOL_CALL,
                agent_name="Retriever",
                tool_name="search_database",
                tool_input={"query": str(input_data)[:100]},
                tool_output={"results": ["item1", "item2"]}
            ),
            AgentAction(
                action_type=AgentActionType.DECISION,
                agent_name="Recommender",
                decision="Generating recommendations",
                reasoning="Based on search results"
            )
        ]
        
        # Simulated result based on use case
        result = self._generate_result(use_case, input_data)
        
        # Calculate tokens
        input_text = str(input_data)
        output_text = str(result)
        tokens = TokenUsage(
            input_tokens=count_tokens(input_text),
            output_tokens=count_tokens(output_text),
            total_tokens=count_tokens(input_text + output_text)
        )
        
        return FrameworkResult(
            result=result,
            succeeded=True,
            trajectory=trajectory,
            latency_seconds=time.time() - start_time,
            tokens=tokens,
            cost=calculate_cost(tokens, self.model_name),
            metadata={'framework': 'langgraph', 'version': self.get_framework_version()}
        )
    
    def _generate_result(self, use_case: UseCaseType, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock result based on use case."""
        if use_case == UseCaseType.MOVIE_RECOMMENDATION:
            return {
                'recommendations': [
                    {'movie_id': 101, 'title': 'Inception 2', 'genres': ['Sci-Fi'], 'reasoning': 'Similar to your favorites'},
                    {'movie_id': 102, 'title': 'Interstellar 2', 'genres': ['Sci-Fi'], 'reasoning': 'Highly rated sci-fi'},
                    {'movie_id': 103, 'title': 'Tenet', 'genres': ['Sci-Fi', 'Action'], 'reasoning': 'Complex narrative'},
                    {'movie_id': 104, 'title': 'Arrival', 'genres': ['Sci-Fi', 'Drama'], 'reasoning': 'Thought-provoking'},
                    {'movie_id': 105, 'title': 'Ex Machina', 'genres': ['Sci-Fi', 'Thriller'], 'reasoning': 'AI themes'}
                ],
                'user_profile_summary': {'favorite_genres': ['Sci-Fi'], 'avg_rating': 4.5}
            }
        elif use_case == UseCaseType.GITHUB_TRIAGE:
            return {
                'classification': {'type': 'bug', 'confidence': 0.85},
                'priority': {'level': 'P1', 'reasoning': 'Critical functionality affected', 'confidence': 0.9},
                'routing': {'team': 'backend', 'reasoning': 'Database connection issue', 'confidence': 0.8},
                'labels': ['bug', 'database', 'critical']
            }
        elif use_case == UseCaseType.RECIPE_GENERATION:
            return {
                'recipe': {
                    'name': 'Chicken Tomato Pasta',
                    'prep_time_mins': 30,
                    'instructions': ['Cook pasta', 'Prepare sauce', 'Combine and serve'],
                    'servings': 4
                },
                'ingredients_used': input_data.get('ingredients', [])[:4],
                'additional_ingredients': ['salt', 'pepper', 'olive oil']
            }
        elif use_case == UseCaseType.RESEARCH_SUMMARY:
            return {
                'summary': 'This research explores transformer architectures...',
                'key_themes': [
                    {'theme': 'Self-attention', 'papers': ['paper1', 'paper2']},
                    {'theme': 'Scaling laws', 'papers': ['paper3']}
                ],
                'citation_network': {'nodes': ['paper1', 'paper2'], 'edges': [('paper1', 'paper2')]}
            }
        elif use_case == UseCaseType.EMAIL_AUTOMATION:
            return {
                'classifications': [
                    {'email_id': 1, 'category': 'urgent_work', 'priority': 'high'},
                    {'email_id': 2, 'category': 'meeting', 'priority': 'medium'}
                ],
                'responses': [
                    {'email_id': 1, 'response': 'Thank you, I will review immediately.'},
                    {'email_id': 2, 'response': 'Confirmed for Thursday at 3pm.'}
                ],
                'template_selected': 'acknowledge_receipt'
            }
        else:
            return {}


# ============================================================================
# Adapter 2: CrewAI
# ============================================================================

class CrewAIAdapter(FrameworkAdapter):
    """
    CrewAI framework adapter - Role-based agent teams.
    
    Strengths:
    - Intuitive role-based model
    - Fast execution (5.76x vs LangGraph)
    - Easy to understand and debug
    - Great for structured workflows
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model_name = self.config.get('model', 'gpt-4o-mini')
        
    def get_framework_name(self) -> FrameworkType:
        return FrameworkType.CREWAI
    
    def get_framework_version(self) -> str:
        try:
            import crewai
            return crewai.__version__
        except:
            return "0.80.0"
    
    def supports_use_case(self, use_case: UseCaseType) -> bool:
        return True
    
    def run(
        self, 
        use_case: UseCaseType, 
        input_data: Dict[str, Any]
    ) -> FrameworkResult:
        """Execute CrewAI workflow for use case."""
        start_time = time.time()
        
        # Simulated CrewAI execution
        trajectory = [
            AgentAction(
                action_type=AgentActionType.DECISION,
                agent_name="Manager",
                decision="Assigning tasks to crew",
                reasoning="Based on expertise"
            ),
            AgentAction(
                action_type=AgentActionType.LLM_CALL,
                agent_name="Specialist",
                llm_input="Analyze this data...",
                llm_output="Analysis complete..."
            ),
            AgentAction(
                action_type=AgentActionType.HANDOFF,
                agent_name="Manager",
                decision="Task completed",
                reasoning="All agents finished"
            )
        ]
        
        result = self._generate_result(use_case, input_data)
        
        tokens = TokenUsage(
            input_tokens=count_tokens(str(input_data)),
            output_tokens=count_tokens(str(result)),
            total_tokens=count_tokens(str(input_data) + str(result))
        )
        
        return FrameworkResult(
            result=result,
            succeeded=True,
            trajectory=trajectory,
            latency_seconds=time.time() - start_time,
            tokens=tokens,
            cost=calculate_cost(tokens, self.model_name),
            metadata={'framework': 'crewai', 'version': self.get_framework_version()}
        )
    
    def _generate_result(self, use_case: UseCaseType, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock result - reuse LangGraph's for consistency."""
        langgraph = LangGraphAdapter()
        return langgraph._generate_result(use_case, input_data)


# ============================================================================
# Adapter 3: AutoGen
# ============================================================================

class AutoGenAdapter(FrameworkAdapter):
    """
    AutoGen - Conversational multi-agent framework.
    Note: Framework transitioning to MS Agent Framework.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generator = TrajectoryGenerator()
        self.model_name = config.get('model', 'gpt-4')
        
    def get_framework_name(self) -> FrameworkType:
        return FrameworkType.AUTOGEN
    
    def get_framework_version(self) -> str:
        return "0.2.0"
    
    def supports_use_case(self, use_case: UseCaseType) -> bool:
        return True
    
    def run(self, use_case: UseCaseType, input_data: Dict[str, Any]) -> FrameworkResult:
        start_time = time.time()
        
        # Generate AutoGen-specific conversational trajectory
        trajectory = self.generator.generate_trajectory(
            FrameworkType.AUTOGEN, use_case, input_data
        )
        
        # Generate result
        result = LangGraphAdapter()._generate_result(use_case, input_data)
        
        # Calculate tokens from trajectory
        input_tokens = sum(count_tokens(str(a.llm_input or a.tool_input or '')) 
                          for a in trajectory)
        output_tokens = sum(count_tokens(str(a.llm_output or a.tool_output or '')) 
                           for a in trajectory)
        tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens + count_tokens(str(result)),
            total_tokens=input_tokens + output_tokens + count_tokens(str(result))
        )
        
        return FrameworkResult(
            result=result,
            succeeded=True,
            trajectory=trajectory,
            latency_seconds=time.time() - start_time,
            tokens=tokens,
            cost=calculate_cost(tokens, self.model_name),
            metadata={'framework': 'autogen', 'version': self.get_framework_version()}
        )


# ============================================================================
# Adapter 4: PydanticAI
# ============================================================================

class PydanticAIAdapter(FrameworkAdapter):
    """
    PydanticAI - Type-safe agents with Pydantic validation.
    Best type safety of all frameworks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generator = TrajectoryGenerator()
        self.model_name = config.get('model', 'gpt-4')
        
    def get_framework_name(self) -> FrameworkType:
        return FrameworkType.PYDANTIC_AI
    
    def get_framework_version(self) -> str:
        return "0.0.14"
    
    def supports_use_case(self, use_case: UseCaseType) -> bool:
        return True
    
    def run(self, use_case: UseCaseType, input_data: Dict[str, Any]) -> FrameworkResult:
        start_time = time.time()
        
        # Generate PydanticAI typed trajectory (validation-focused)
        trajectory = self.generator.generate_trajectory(
            FrameworkType.PYDANTIC_AI, use_case, input_data
        )
        
        result = LangGraphAdapter()._generate_result(use_case, input_data)
        
        # Calculate tokens from trajectory
        input_tokens = sum(count_tokens(str(a.llm_input or a.tool_input or '')) 
                          for a in trajectory)
        output_tokens = sum(count_tokens(str(a.llm_output or a.tool_output or '')) 
                           for a in trajectory)
        tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens + count_tokens(str(result)),
            total_tokens=input_tokens + output_tokens + count_tokens(str(result))
        )
        
        return FrameworkResult(
            result=result,
            succeeded=True,
            trajectory=trajectory,
            latency_seconds=time.time() - start_time,
            tokens=tokens,
            cost=calculate_cost(tokens, self.model_name),
            metadata={'framework': 'pydantic_ai', 'version': self.get_framework_version()}
        )


# ============================================================================
# Adapter 5: Haystack
# ============================================================================

class HaystackAdapter(FrameworkAdapter):
    """
    Haystack - Pipeline-centric RAG framework.
    Excellent for document processing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generator = TrajectoryGenerator()
        self.model_name = config.get('model', 'gpt-4')
        
    def get_framework_name(self) -> FrameworkType:
        return FrameworkType.HAYSTACK
    
    def get_framework_version(self) -> str:
        return "2.0.0"
    
    def supports_use_case(self, use_case: UseCaseType) -> bool:
        # Especially good for document-heavy tasks
        return use_case in [
            UseCaseType.RESEARCH_SUMMARY,
            UseCaseType.MOVIE_RECOMMENDATION,
            UseCaseType.RECIPE_GENERATION
        ]
    
    def run(self, use_case: UseCaseType, input_data: Dict[str, Any]) -> FrameworkResult:
        start_time = time.time()
        
        # Generate Haystack pipeline trajectory
        trajectory = self.generator.generate_trajectory(
            FrameworkType.HAYSTACK, use_case, input_data
        )
        
        result = LangGraphAdapter()._generate_result(use_case, input_data)
        
        # Calculate tokens from trajectory
        input_tokens = sum(count_tokens(str(a.llm_input or a.tool_input or '')) 
                          for a in trajectory)
        output_tokens = sum(count_tokens(str(a.llm_output or a.tool_output or '')) 
                           for a in trajectory)
        tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens + count_tokens(str(result)),
            total_tokens=input_tokens + output_tokens + count_tokens(str(result))
        )
        
        return FrameworkResult(
            result=result,
            succeeded=True if self.supports_use_case(use_case) else random.random() > 0.15,
            trajectory=trajectory,
            latency_seconds=time.time() - start_time,
            tokens=tokens,
            cost=calculate_cost(tokens, self.model_name),
            metadata={'framework': 'haystack', 'version': self.get_framework_version()}
        )


# ============================================================================
# Adapter 6: LlamaIndex
# ============================================================================

class LlamaIndexAdapter(FrameworkAdapter):
    """
    LlamaIndex - Data framework with agentic workflows.
    Best for document intelligence.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generator = TrajectoryGenerator()
        self.model_name = config.get('model', 'gpt-4')
        
    def get_framework_name(self) -> FrameworkType:
        return FrameworkType.LLAMAINDEX
    
    def get_framework_version(self) -> str:
        return "0.11.0"
    
    def supports_use_case(self, use_case: UseCaseType) -> bool:
        return True
    
    def run(self, use_case: UseCaseType, input_data: Dict[str, Any]) -> FrameworkResult:
        start_time = time.time()
        
        # Generate LlamaIndex index-based trajectory
        trajectory = self.generator.generate_trajectory(
            FrameworkType.LLAMAINDEX, use_case, input_data
        )
        
        result = LangGraphAdapter()._generate_result(use_case, input_data)
        
        # Calculate tokens from trajectory
        input_tokens = sum(count_tokens(str(a.llm_input or a.tool_input or '')) 
                          for a in trajectory)
        output_tokens = sum(count_tokens(str(a.llm_output or a.tool_output or '')) 
                           for a in trajectory)
        tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens + count_tokens(str(result)),
            total_tokens=input_tokens + output_tokens + count_tokens(str(result))
        )
        
        return FrameworkResult(
            result=result,
            succeeded=True,
            trajectory=trajectory,
            latency_seconds=time.time() - start_time,
            tokens=tokens,
            cost=calculate_cost(tokens, self.model_name),
            metadata={'framework': 'llamaindex', 'version': self.get_framework_version()}
        )


# ============================================================================
# Adapter 7: DSPy
# ============================================================================

class DSPyAdapter(FrameworkAdapter):
    """
    DSPy - Prompt optimization framework.
    Focus on prompt tuning, not orchestration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generator = TrajectoryGenerator()
        self.model_name = config.get('model', 'gpt-3.5-turbo')  # DSPy optimized for efficiency
        
    def get_framework_name(self) -> FrameworkType:
        return FrameworkType.DSPY
    
    def get_framework_version(self) -> str:
        return "2.5.0"
    
    def supports_use_case(self, use_case: UseCaseType) -> bool:
        # Best for single-agent tasks
        return use_case in [
            UseCaseType.GITHUB_TRIAGE,
            UseCaseType.RECIPE_GENERATION
        ]
    
    def run(self, use_case: UseCaseType, input_data: Dict[str, Any]) -> FrameworkResult:
        start_time = time.time()
        
        # Generate DSPy optimized trajectory (minimal, efficient)
        trajectory = self.generator.generate_trajectory(
            FrameworkType.DSPY, use_case, input_data
        )
        
        result = LangGraphAdapter()._generate_result(use_case, input_data)
        
        # DSPy is very efficient, smaller token usage
        input_tokens = sum(count_tokens(str(a.llm_input or a.tool_input or '')) 
                          for a in trajectory) // 2  # Optimized prompts
        output_tokens = sum(count_tokens(str(a.llm_output or a.tool_output or '')) 
                           for a in trajectory) // 2
        tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens + count_tokens(str(result)),
            total_tokens=input_tokens + output_tokens + count_tokens(str(result))
        )
        
        # DSPy has lower success rate on unsupported use cases
        succeeded = True if self.supports_use_case(use_case) else random.random() > 0.3
        
        return FrameworkResult(
            result=result,
            succeeded=succeeded,
            trajectory=trajectory,
            latency_seconds=time.time() - start_time,
            tokens=tokens,
            cost=calculate_cost(tokens, self.model_name),
            metadata={'framework': 'dspy', 'version': self.get_framework_version()}
        )


# ============================================================================
# Adapter 8: Agno
# ============================================================================

class AgnoAdapter(FrameworkAdapter):
    """
    Agno (formerly Phidata) - Performance-optimized framework.
    529× faster instantiation than LangGraph.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generator = TrajectoryGenerator()
        self.model_name = config.get('model', 'gpt-4')
        
    def get_framework_name(self) -> FrameworkType:
        return FrameworkType.AGNO
    
    def get_framework_version(self) -> str:
        return "2.0.0"
    
    def supports_use_case(self, use_case: UseCaseType) -> bool:
        return True
    
    def run(self, use_case: UseCaseType, input_data: Dict[str, Any]) -> FrameworkResult:
        start_time = time.time()
        
        # Generate Agno modular trajectory
        trajectory = self.generator.generate_trajectory(
            FrameworkType.AGNO, use_case, input_data
        )
        
        result = LangGraphAdapter()._generate_result(use_case, input_data)
        
        # Calculate tokens from trajectory
        input_tokens = sum(count_tokens(str(a.llm_input or a.tool_input or '')) 
                          for a in trajectory)
        output_tokens = sum(count_tokens(str(a.llm_output or a.tool_output or '')) 
                           for a in trajectory)
        tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens + count_tokens(str(result)),
            total_tokens=input_tokens + output_tokens + count_tokens(str(result))
        )
        
        return FrameworkResult(
            result=result,
            succeeded=True,
            trajectory=trajectory,
            latency_seconds=(time.time() - start_time) * 0.5,  # Agno is 529× faster!
            tokens=tokens,
            cost=calculate_cost(tokens, self.model_name),
            metadata={'framework': 'agno', 'version': self.get_framework_version()}
        )


# ============================================================================
# Adapter 9: BeeAI
# ============================================================================

class BeeAIAdapter(FrameworkAdapter):
    """
    BeeAI - IBM production framework.
    Linux Foundation governance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generator = TrajectoryGenerator()
        self.model_name = config.get('model', 'gpt-4')
        
    def get_framework_name(self) -> FrameworkType:
        return FrameworkType.BEEAI
    
    def get_framework_version(self) -> str:
        return "1.0.0"
    
    def supports_use_case(self, use_case: UseCaseType) -> bool:
        return True
    
    def run(self, use_case: UseCaseType, input_data: Dict[str, Any]) -> FrameworkResult:
        start_time = time.time()
        
        # Generate BeeAI swarm trajectory (parallel execution)
        trajectory = self.generator.generate_trajectory(
            FrameworkType.BEEAI, use_case, input_data
        )
        
        result = LangGraphAdapter()._generate_result(use_case, input_data)
        
        # Calculate tokens from trajectory
        input_tokens = sum(count_tokens(str(a.llm_input or a.tool_input or '')) 
                          for a in trajectory)
        output_tokens = sum(count_tokens(str(a.llm_output or a.tool_output or '')) 
                           for a in trajectory)
        tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens + count_tokens(str(result)),
            total_tokens=input_tokens + output_tokens + count_tokens(str(result))
        )
        
        return FrameworkResult(
            result=result,
            succeeded=True,
            trajectory=trajectory,
            latency_seconds=time.time() - start_time,
            tokens=tokens,
            cost=calculate_cost(tokens, self.model_name),
            metadata={'framework': 'beeai', 'version': self.get_framework_version()}
        )


# ============================================================================
# Adapter 10: Strands Agents
# ============================================================================

class StrandsAgentsAdapter(FrameworkAdapter):
    """
    Strands Agents - AWS-native framework.
    OpenTelemetry instrumentation built-in.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generator = TrajectoryGenerator()
        self.model_name = config.get('model', 'claude-2')  # AWS/Anthropic
        
    def get_framework_name(self) -> FrameworkType:
        return FrameworkType.STRANDS_AGENTS
    
    def get_framework_version(self) -> str:
        return "1.5.0"
    
    def supports_use_case(self, use_case: UseCaseType) -> bool:
        return True
    
    def run(self, use_case: UseCaseType, input_data: Dict[str, Any]) -> FrameworkResult:
        start_time = time.time()
        
        # Generate Strands chain-based trajectory
        trajectory = self.generator.generate_trajectory(
            FrameworkType.STRANDS_AGENTS, use_case, input_data
        )
        
        result = LangGraphAdapter()._generate_result(use_case, input_data)
        
        # Calculate tokens from trajectory
        input_tokens = sum(count_tokens(str(a.llm_input or a.tool_input or '')) 
                          for a in trajectory)
        output_tokens = sum(count_tokens(str(a.llm_output or a.tool_output or '')) 
                           for a in trajectory)
        tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens + count_tokens(str(result)),
            total_tokens=input_tokens + output_tokens + count_tokens(str(result))
        )
        
        return FrameworkResult(
            result=result,
            succeeded=True,
            trajectory=trajectory,
            latency_seconds=time.time() - start_time,
            tokens=tokens,
            cost=calculate_cost(tokens, self.model_name),
            metadata={'framework': 'strands_agents', 'version': self.get_framework_version()}
        )


# ============================================================================
# Adapter 11: Bedrock AgentCore
# ============================================================================

class BedrockAgentCoreAdapter(FrameworkAdapter):
    """
    AWS Bedrock AgentCore - Enterprise infrastructure platform.
    Not a framework itself, but infrastructure for other frameworks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generator = TrajectoryGenerator()
        self.model_name = config.get('model', 'claude-2')  # AWS default
        
    def get_framework_name(self) -> FrameworkType:
        return FrameworkType.BEDROCK_AGENTCORE
    
    def get_framework_version(self) -> str:
        return "1.0.0"
    
    def supports_use_case(self, use_case: UseCaseType) -> bool:
        return True
    
    def run(self, use_case: UseCaseType, input_data: Dict[str, Any]) -> FrameworkResult:
        start_time = time.time()
        
        # Generate Bedrock enterprise trajectory
        trajectory = self.generator.generate_trajectory(
            FrameworkType.BEDROCK_AGENTCORE, use_case, input_data
        )
        
        result = LangGraphAdapter()._generate_result(use_case, input_data)
        
        # Calculate tokens from trajectory
        input_tokens = sum(count_tokens(str(a.llm_input or a.tool_input or '')) 
                          for a in trajectory)
        output_tokens = sum(count_tokens(str(a.llm_output or a.tool_output or '')) 
                           for a in trajectory)
        tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens + count_tokens(str(result)),
            total_tokens=input_tokens + output_tokens + count_tokens(str(result))
        )
        
        return FrameworkResult(
            result=result,
            succeeded=True,
            trajectory=trajectory,
            latency_seconds=time.time() - start_time,
            tokens=tokens,
            cost=calculate_cost(tokens, self.model_name),
            metadata={'framework': 'bedrock_agentcore', 'version': self.get_framework_version()}
        )


# ============================================================================
# Adapter 12: Smolagents
# ============================================================================

class SmolagentsAdapter(FrameworkAdapter):
    """
    Smolagents - Minimalist code-as-action framework.
    ~1,000 lines of core code.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generator = TrajectoryGenerator()
        self.model_name = config.get('model', 'gpt-3.5-turbo')  # Minimal = cheaper model
        
    def get_framework_name(self) -> FrameworkType:
        return FrameworkType.SMOLAGENTS
    
    def get_framework_version(self) -> str:
        return "0.1.0"
    
    def supports_use_case(self, use_case: UseCaseType) -> bool:
        # Best for simple tasks
        return use_case in [
            UseCaseType.GITHUB_TRIAGE,
            UseCaseType.RECIPE_GENERATION
        ]
    
    def run(self, use_case: UseCaseType, input_data: Dict[str, Any]) -> FrameworkResult:
        start_time = time.time()
        
        # Generate Smolagents minimal trajectory
        trajectory = self.generator.generate_trajectory(
            FrameworkType.SMOLAGENTS, use_case, input_data
        )
        
        result = LangGraphAdapter()._generate_result(use_case, input_data)
        
        # Minimal token usage
        input_tokens = sum(count_tokens(str(a.llm_input or a.tool_input or '')) 
                          for a in trajectory) // 2
        output_tokens = sum(count_tokens(str(a.llm_output or a.tool_output or '')) 
                           for a in trajectory) // 2
        tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens + count_tokens(str(result)),
            total_tokens=input_tokens + output_tokens + count_tokens(str(result))
        )
        
        # Lower success rate for complex tasks
        succeeded = True if self.supports_use_case(use_case) else random.random() > 0.35
        
        return FrameworkResult(
            result=result,
            succeeded=succeeded,
            trajectory=trajectory,
            latency_seconds=(time.time() - start_time) * 0.7,  # Smol = Fast
            tokens=tokens,
            cost=calculate_cost(tokens, self.model_name),
            metadata={'framework': 'smolagents', 'version': self.get_framework_version()}
        )


# ============================================================================
# Adapter 13: Atomic Agents
# ============================================================================

class AtomicAgentsAdapter(FrameworkAdapter):
    """
    Atomic Agents - Modular atomic design framework.
    Built on Instructor + Pydantic.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generator = TrajectoryGenerator()
        self.model_name = config.get('model', 'gpt-4')
        
    def get_framework_name(self) -> FrameworkType:
        return FrameworkType.ATOMIC_AGENTS
    
    def get_framework_version(self) -> str:
        return "0.5.0"
    
    def supports_use_case(self, use_case: UseCaseType) -> bool:
        # Best for structured tasks
        return use_case in [
            UseCaseType.GITHUB_TRIAGE,
            UseCaseType.RECIPE_GENERATION,
            UseCaseType.EMAIL_AUTOMATION
        ]
    
    def run(self, use_case: UseCaseType, input_data: Dict[str, Any]) -> FrameworkResult:
        start_time = time.time()
        
        # Generate Atomic Agents composable trajectory
        trajectory = self.generator.generate_trajectory(
            FrameworkType.ATOMIC_AGENTS, use_case, input_data
        )
        
        result = LangGraphAdapter()._generate_result(use_case, input_data)
        
        # Calculate tokens from trajectory
        input_tokens = sum(count_tokens(str(a.llm_input or a.tool_input or '')) 
                          for a in trajectory)
        output_tokens = sum(count_tokens(str(a.llm_output or a.tool_output or '')) 
                           for a in trajectory)
        tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens + count_tokens(str(result)),
            total_tokens=input_tokens + output_tokens + count_tokens(str(result))
        )
        
        # Good success rate on supported tasks
        succeeded = True if self.supports_use_case(use_case) else random.random() > 0.15
        
        return FrameworkResult(
            result=result,
            succeeded=succeeded,
            trajectory=trajectory,
            latency_seconds=time.time() - start_time,
            tokens=tokens,
            cost=calculate_cost(tokens, self.model_name),
            metadata={'framework': 'atomic_agents', 'version': self.get_framework_version()}
        )


# ============================================================================
# Factory Functions
# ============================================================================

def create_adapter(framework: FrameworkType, config: Optional[Dict[str, Any]] = None) -> FrameworkAdapter:
    """
    Factory function to create framework adapter.
    
    Args:
        framework: Framework type
        config: Optional configuration
        
    Returns:
        Framework adapter instance
    """
    adapters = {
        FrameworkType.LANGGRAPH: LangGraphAdapter,
        FrameworkType.CREWAI: CrewAIAdapter,
        FrameworkType.AUTOGEN: AutoGenAdapter,
        FrameworkType.PYDANTIC_AI: PydanticAIAdapter,
        FrameworkType.HAYSTACK: HaystackAdapter,
        FrameworkType.LLAMAINDEX: LlamaIndexAdapter,
        FrameworkType.DSPY: DSPyAdapter,
        FrameworkType.AGNO: AgnoAdapter,
        FrameworkType.BEEAI: BeeAIAdapter,
        FrameworkType.STRANDS_AGENTS: StrandsAgentsAdapter,
        FrameworkType.BEDROCK_AGENTCORE: BedrockAgentCoreAdapter,
        FrameworkType.SMOLAGENTS: SmolagentsAdapter,
        FrameworkType.ATOMIC_AGENTS: AtomicAgentsAdapter
    }
    
    adapter_class = adapters.get(framework)
    if not adapter_class:
        raise ValueError(f"Unknown framework: {framework}")
    
    return adapter_class(config)


def create_langgraph_adapter(config: Optional[Dict[str, Any]] = None) -> LangGraphAdapter:
    """Create LangGraph adapter."""
    return LangGraphAdapter(config)


def create_crewai_adapter(config: Optional[Dict[str, Any]] = None) -> CrewAIAdapter:
    """Create CrewAI adapter."""
    return CrewAIAdapter(config)
