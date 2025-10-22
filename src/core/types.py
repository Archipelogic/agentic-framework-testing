"""
Type definitions and data models for the agentic framework testing harness.

This module defines all core types used across the framework, ensuring
type safety and clear contracts between components.
"""

from typing import Dict, List, Any, Optional, Union, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod


# ============================================================================
# Enums
# ============================================================================

class UseCaseType(str, Enum):
    """Supported use case types."""
    MOVIE_RECOMMENDATION = "movie_recommendation"
    GITHUB_TRIAGE = "github_triage"
    RECIPE_GENERATION = "recipe_generation"
    RESEARCH_SUMMARY = "research_summary"
    EMAIL_AUTOMATION = "email_automation"


class FrameworkType(str, Enum):
    """Supported framework types."""
    LANGGRAPH = "langgraph"
    CREWAI = "crewai"
    AUTOGEN = "autogen"
    PYDANTIC_AI = "pydantic_ai"
    HAYSTACK = "haystack"
    LLAMAINDEX = "llamaindex"
    DSPY = "dspy"
    AGNO = "agno"
    BEEAI = "beeai"
    STRANDS_AGENTS = "strands_agents"
    BEDROCK_AGENTCORE = "bedrock_agentcore"
    SMOLAGENTS = "smolagents"
    ATOMIC_AGENTS = "atomic_agents"


class AgentActionType(str, Enum):
    """Types of agent actions."""
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    DECISION = "decision"
    HANDOFF = "handoff"
    ERROR = "error"


# ============================================================================
# Core Data Models
# ============================================================================

@dataclass
class TokenUsage:
    """Token usage tracking."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )


@dataclass
class CostBreakdown:
    """Cost tracking with breakdown."""
    total_usd: float
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    model_name: Optional[str] = None
    
    def __add__(self, other: 'CostBreakdown') -> 'CostBreakdown':
        return CostBreakdown(
            total_usd=self.total_usd + other.total_usd,
            input_cost_usd=self.input_cost_usd + other.input_cost_usd,
            output_cost_usd=self.output_cost_usd + other.output_cost_usd,
            model_name=self.model_name or other.model_name
        )


@dataclass
class AgentAction:
    """Represents a single agent action in the trajectory."""
    action_type: AgentActionType
    agent_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # For tool calls
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Any] = None
    
    # For LLM calls
    llm_input: Optional[str] = None
    llm_output: Optional[str] = None
    
    # For decisions
    decision: Optional[str] = None
    reasoning: Optional[str] = None
    
    # Metadata
    duration_seconds: Optional[float] = None
    tokens: Optional[TokenUsage] = None
    cost: Optional[CostBreakdown] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'action_type': self.action_type.value,
            'agent_name': self.agent_name,
            'timestamp': self.timestamp.isoformat(),
            'tool_name': self.tool_name,
            'tool_input': self.tool_input,
            'tool_output': self.tool_output,
            'llm_input': self.llm_input,
            'llm_output': self.llm_output,
            'decision': self.decision,
            'reasoning': self.reasoning,
            'duration_seconds': self.duration_seconds,
            'tokens': self.tokens.__dict__ if self.tokens else None,
            'cost': self.cost.__dict__ if self.cost else None,
            'error': self.error
        }


@dataclass
class FrameworkResult:
    """Standardized result from any framework execution."""
    result: Dict[str, Any]
    succeeded: bool
    trajectory: List[AgentAction] = field(default_factory=list)
    latency_seconds: float = 0.0
    tokens: TokenUsage = field(default_factory=TokenUsage)
    cost: CostBreakdown = field(default_factory=lambda: CostBreakdown(total_usd=0.0))
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'result': self.result,
            'succeeded': self.succeeded,
            'trajectory': [action.to_dict() for action in self.trajectory],
            'latency_seconds': self.latency_seconds,
            'tokens': self.tokens.__dict__,
            'cost': self.cost.__dict__,
            'errors': self.errors,
            'warnings': self.warnings,
            'metadata': self.metadata
        }


@dataclass
class TestCase:
    """Represents a single test case for evaluation."""
    id: str
    use_case: UseCaseType
    input_data: Dict[str, Any]
    ground_truth: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'use_case': self.use_case.value,
            'input_data': self.input_data,
            'ground_truth': self.ground_truth,
            'metadata': self.metadata
        }


# ============================================================================
# Framework Adapter Interface
# ============================================================================

class FrameworkAdapter(ABC):
    """
    Base interface that all framework adapters must implement.
    
    This is the core contract that allows any agentic framework to plug
    into the testing harness.
    """
    
    @abstractmethod
    def get_framework_name(self) -> FrameworkType:
        """Return the framework identifier."""
        pass
    
    @abstractmethod
    def get_framework_version(self) -> str:
        """Return the framework version string."""
        pass
    
    @abstractmethod
    def supports_use_case(self, use_case: UseCaseType) -> bool:
        """Check if this framework supports a given use case."""
        pass
    
    @abstractmethod
    def run(
        self, 
        use_case: UseCaseType, 
        input_data: Dict[str, Any]
    ) -> FrameworkResult:
        """
        Execute the agent workflow for a given use case.
        
        Args:
            use_case: Which use case to execute
            input_data: Use-case-specific input dict
            
        Returns:
            FrameworkResult with result, trajectory, metrics
        """
        pass
    
    def cleanup(self) -> None:
        """Optional cleanup after evaluation run."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Return framework configuration."""
        return {}
