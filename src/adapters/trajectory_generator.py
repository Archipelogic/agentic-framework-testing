"""
Trajectory generator for creating realistic agent action sequences.
This module helps create proper trajectories for framework adapters.
"""

from typing import List, Dict, Any
import random
from datetime import datetime

from src.core.types import (
    AgentAction, AgentActionType, FrameworkType, UseCaseType,
    TokenUsage
)


class TrajectoryGenerator:
    """Generate realistic trajectories for different frameworks and use cases."""
    
    def __init__(self):
        self.framework_patterns = self._define_framework_patterns()
        self.use_case_patterns = self._define_use_case_patterns()
    
    def _define_framework_patterns(self) -> Dict[FrameworkType, Dict[str, Any]]:
        """Define characteristic patterns for each framework."""
        return {
            FrameworkType.LANGGRAPH: {
                'style': 'graph_based',
                'agents': ['GraphNavigator', 'NodeProcessor', 'StateManager'],
                'emphasis': 'state_transitions',
                'typical_length': (5, 8)
            },
            FrameworkType.CREWAI: {
                'style': 'role_based', 
                'agents': ['Manager', 'Specialist', 'Reviewer', 'Executor'],
                'emphasis': 'delegation',
                'typical_length': (6, 10)
            },
            FrameworkType.AUTOGEN: {
                'style': 'conversational',
                'agents': ['Assistant', 'UserProxy', 'Critic'],
                'emphasis': 'dialogue',
                'typical_length': (8, 12)
            },
            FrameworkType.PYDANTIC_AI: {
                'style': 'typed',
                'agents': ['TypedAgent', 'Validator'],
                'emphasis': 'validation',
                'typical_length': (4, 6)
            },
            FrameworkType.HAYSTACK: {
                'style': 'pipeline',
                'agents': ['Retriever', 'Reader', 'Generator'],
                'emphasis': 'document_processing',
                'typical_length': (5, 7)
            },
            FrameworkType.LLAMAINDEX: {
                'style': 'index_based',
                'agents': ['Indexer', 'QueryEngine', 'Synthesizer'],
                'emphasis': 'retrieval',
                'typical_length': (5, 8)
            },
            FrameworkType.DSPY: {
                'style': 'optimized',
                'agents': ['Optimizer', 'Executor'],
                'emphasis': 'efficiency',
                'typical_length': (3, 5)
            },
            FrameworkType.AGNO: {
                'style': 'modular',
                'agents': ['CoreAgent', 'PluginAgent'],
                'emphasis': 'flexibility',
                'typical_length': (5, 7)
            },
            FrameworkType.BEEAI: {
                'style': 'swarm',
                'agents': ['Queen', 'Worker1', 'Worker2', 'Worker3'],
                'emphasis': 'parallel',
                'typical_length': (7, 10)
            },
            FrameworkType.STRANDS_AGENTS: {
                'style': 'chain',
                'agents': ['ChainLink1', 'ChainLink2', 'ChainLink3'],
                'emphasis': 'sequential',
                'typical_length': (4, 6)
            },
            FrameworkType.BEDROCK_AGENTCORE: {
                'style': 'enterprise',
                'agents': ['Orchestrator', 'ServiceAgent', 'Monitor'],
                'emphasis': 'reliability',
                'typical_length': (5, 8)
            },
            FrameworkType.SMOLAGENTS: {
                'style': 'minimal',
                'agents': ['SmolAgent'],
                'emphasis': 'simplicity',
                'typical_length': (2, 4)
            },
            FrameworkType.ATOMIC_AGENTS: {
                'style': 'atomic',
                'agents': ['AtomicUnit1', 'AtomicUnit2'],
                'emphasis': 'composability',
                'typical_length': (4, 6)
            }
        }
    
    def _define_use_case_patterns(self) -> Dict[UseCaseType, List[str]]:
        """Define typical action sequences for each use case."""
        return {
            UseCaseType.MOVIE_RECOMMENDATION: [
                'analyze_preferences',
                'fetch_user_history',
                'query_movie_database',
                'apply_filters',
                'rank_results',
                'generate_explanations'
            ],
            UseCaseType.GITHUB_TRIAGE: [
                'parse_issue',
                'extract_keywords',
                'search_similar',
                'classify_type',
                'assign_labels',
                'suggest_assignee'
            ],
            UseCaseType.RECIPE_GENERATION: [
                'parse_ingredients',
                'check_dietary_restrictions',
                'search_recipes',
                'adapt_recipe',
                'calculate_nutrition',
                'format_output'
            ],
            UseCaseType.RESEARCH_SUMMARY: [
                'parse_papers',
                'extract_abstracts',
                'identify_themes',
                'analyze_methodologies',
                'synthesize_findings',
                'generate_summary'
            ],
            UseCaseType.EMAIL_AUTOMATION: [
                'parse_email',
                'classify_intent',
                'extract_entities',
                'generate_response',
                'apply_template',
                'schedule_send'
            ]
        }
    
    def generate_trajectory(
        self,
        framework: FrameworkType,
        use_case: UseCaseType,
        input_data: Dict[str, Any]
    ) -> List[AgentAction]:
        """Generate a realistic trajectory for the given framework and use case."""
        
        pattern = self.framework_patterns.get(framework, {})
        use_case_steps = self.use_case_patterns.get(use_case, [])
        
        trajectory = []
        agents = pattern.get('agents', ['Agent'])
        min_len, max_len = pattern.get('typical_length', (4, 8))
        num_steps = random.randint(min_len, max_len)
        
        # Start with planning
        trajectory.append(self._create_planning_action(agents[0], use_case))
        
        # Add framework-specific actions based on style
        style = pattern.get('style', 'generic')
        
        if style == 'graph_based':
            trajectory.extend(self._generate_graph_actions(agents, use_case_steps))
        elif style == 'role_based':
            trajectory.extend(self._generate_delegation_actions(agents, use_case_steps))
        elif style == 'conversational':
            trajectory.extend(self._generate_dialogue_actions(agents, use_case_steps))
        elif style == 'pipeline':
            trajectory.extend(self._generate_pipeline_actions(agents, use_case_steps))
        elif style == 'swarm':
            trajectory.extend(self._generate_parallel_actions(agents, use_case_steps))
        else:
            trajectory.extend(self._generate_generic_actions(agents, use_case_steps, num_steps))
        
        # Add completion action
        trajectory.append(self._create_completion_action(agents[0]))
        
        return trajectory
    
    def _create_planning_action(self, agent: str, use_case: UseCaseType) -> AgentAction:
        """Create initial planning action."""
        return AgentAction(
            action_type=AgentActionType.DECISION,
            agent_name=agent,
            decision=f"Planning approach for {use_case.value}",
            reasoning="Analyzing requirements and available resources",
            duration_seconds=random.uniform(0.1, 0.3)
        )
    
    def _create_completion_action(self, agent: str) -> AgentAction:
        """Create completion action."""
        return AgentAction(
            action_type=AgentActionType.DECISION,
            agent_name=agent,
            decision="Task completed successfully",
            reasoning="All subtasks finished and results validated",
            duration_seconds=random.uniform(0.05, 0.15)
        )
    
    def _generate_graph_actions(self, agents: List[str], steps: List[str]) -> List[AgentAction]:
        """Generate graph-based traversal actions."""
        actions = []
        for i, step in enumerate(steps[:4]):
            # State transition
            actions.append(AgentAction(
                action_type=AgentActionType.DECISION,
                agent_name=agents[0],
                decision=f"Transitioning to node: {step}",
                reasoning="Following graph edges",
                duration_seconds=random.uniform(0.1, 0.2)
            ))
            # Node processing
            actions.append(AgentAction(
                action_type=AgentActionType.LLM_CALL,
                agent_name=agents[1],
                llm_input=f"Process {step}",
                llm_output=f"Completed {step} with results",
                duration_seconds=random.uniform(0.5, 1.0)
            ))
        return actions
    
    def _generate_delegation_actions(self, agents: List[str], steps: List[str]) -> List[AgentAction]:
        """Generate role-based delegation actions."""
        actions = []
        manager = agents[0]
        specialists = agents[1:]
        
        # Manager assigns tasks
        actions.append(AgentAction(
            action_type=AgentActionType.DECISION,
            agent_name=manager,
            decision="Assigning tasks to crew",
            reasoning="Based on agent expertise",
            duration_seconds=random.uniform(0.1, 0.3)
        ))
        
        # Specialists work
        for i, step in enumerate(steps[:3]):
            specialist = specialists[i % len(specialists)]
            actions.append(AgentAction(
                action_type=AgentActionType.LLM_CALL,
                agent_name=specialist,
                llm_input=f"Execute {step}",
                llm_output=f"Results for {step}",
                duration_seconds=random.uniform(0.4, 0.8)
            ))
            # Handoff
            if i < len(steps[:3]) - 1:
                actions.append(AgentAction(
                    action_type=AgentActionType.HANDOFF,
                    agent_name=specialist,
                    decision=f"Handing off to next specialist",
                    reasoning="Task segment complete"
                ))
        
        return actions
    
    def _generate_dialogue_actions(self, agents: List[str], steps: List[str]) -> List[AgentAction]:
        """Generate conversational pattern actions."""
        actions = []
        for i, step in enumerate(steps[:4]):
            # Agent proposes
            actions.append(AgentAction(
                action_type=AgentActionType.LLM_CALL,
                agent_name=agents[0],
                llm_input=f"How should we handle {step}?",
                llm_output=f"I suggest we {step} by...",
                duration_seconds=random.uniform(0.3, 0.6)
            ))
            # Other agent responds
            if len(agents) > 1:
                actions.append(AgentAction(
                    action_type=AgentActionType.LLM_CALL,
                    agent_name=agents[1],
                    llm_input=f"Review suggestion for {step}",
                    llm_output=f"Good approach, let me add...",
                    duration_seconds=random.uniform(0.3, 0.6)
                ))
        return actions
    
    def _generate_pipeline_actions(self, agents: List[str], steps: List[str]) -> List[AgentAction]:
        """Generate pipeline processing actions."""
        actions = []
        for i, (agent, step) in enumerate(zip(agents * 2, steps[:5])):
            # Each stage of pipeline
            if 'search' in step or 'fetch' in step or 'query' in step:
                actions.append(AgentAction(
                    action_type=AgentActionType.TOOL_CALL,
                    agent_name=agent,
                    tool_name="retriever",
                    tool_input={"query": step},
                    tool_output={"results": f"Data for {step}"},
                    duration_seconds=random.uniform(0.2, 0.5)
                ))
            else:
                actions.append(AgentAction(
                    action_type=AgentActionType.LLM_CALL,
                    agent_name=agent,
                    llm_input=f"Process {step}",
                    llm_output=f"Processed {step}",
                    duration_seconds=random.uniform(0.3, 0.7)
                ))
        return actions
    
    def _generate_parallel_actions(self, agents: List[str], steps: List[str]) -> List[AgentAction]:
        """Generate parallel swarm actions."""
        actions = []
        queen = agents[0]
        workers = agents[1:] if len(agents) > 1 else ['Worker']
        
        # Queen coordinates
        actions.append(AgentAction(
            action_type=AgentActionType.DECISION,
            agent_name=queen,
            decision="Dispatching parallel tasks",
            reasoning="Maximize throughput",
            duration_seconds=random.uniform(0.1, 0.2)
        ))
        
        # Workers execute in parallel
        for step in steps[:3]:
            for worker in workers[:2]:
                actions.append(AgentAction(
                    action_type=AgentActionType.LLM_CALL,
                    agent_name=worker,
                    llm_input=f"Parallel execution of {step}",
                    llm_output=f"Worker result for {step}",
                    duration_seconds=random.uniform(0.2, 0.4)
                ))
        
        # Queen aggregates
        actions.append(AgentAction(
            action_type=AgentActionType.DECISION,
            agent_name=queen,
            decision="Aggregating parallel results",
            reasoning="Combining worker outputs",
            duration_seconds=random.uniform(0.1, 0.3)
        ))
        
        return actions
    
    def _generate_generic_actions(
        self, 
        agents: List[str], 
        steps: List[str], 
        num_steps: int
    ) -> List[AgentAction]:
        """Generate generic action sequence."""
        actions = []
        for i in range(min(num_steps, len(steps))):
            agent = agents[i % len(agents)]
            step = steps[i]
            
            # Mix of action types
            if i % 3 == 0 and ('search' in step or 'fetch' in step):
                actions.append(AgentAction(
                    action_type=AgentActionType.TOOL_CALL,
                    agent_name=agent,
                    tool_name="generic_tool",
                    tool_input={"action": step},
                    tool_output={"result": f"Data from {step}"},
                    duration_seconds=random.uniform(0.2, 0.5)
                ))
            elif i % 3 == 1:
                actions.append(AgentAction(
                    action_type=AgentActionType.DECISION,
                    agent_name=agent,
                    decision=f"Executing {step}",
                    reasoning="Next logical step",
                    duration_seconds=random.uniform(0.1, 0.3)
                ))
            else:
                actions.append(AgentAction(
                    action_type=AgentActionType.LLM_CALL,
                    agent_name=agent,
                    llm_input=f"Perform {step}",
                    llm_output=f"Completed {step}",
                    duration_seconds=random.uniform(0.3, 0.7)
                ))
        
        return actions


# Convenience function for adapter implementations
def generate_trajectory_for_framework(
    framework: FrameworkType,
    use_case: UseCaseType, 
    input_data: Dict[str, Any]
) -> List[AgentAction]:
    """Generate a trajectory for the specified framework and use case."""
    generator = TrajectoryGenerator()
    return generator.generate_trajectory(framework, use_case, input_data)
