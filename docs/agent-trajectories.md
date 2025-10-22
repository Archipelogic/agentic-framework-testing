# Agent Trajectory Construction Guide

## Overview

Agent trajectories are sequences of `AgentAction` objects that represent the step-by-step execution flow of an agentic framework. They provide visibility into how frameworks solve problems and are essential for evaluating capabilities like multi-agent coordination, tool usage, and decision-making.

## AgentAction Types

The system supports five types of agent actions:

### 1. TOOL_CALL
Used when an agent invokes an external tool or function.

```python
AgentAction(
    action_type=AgentActionType.TOOL_CALL,
    agent_name="DataAnalyzer",
    tool_name="web_search",
    tool_input={"query": "latest AI trends"},
    tool_output={"results": [...]}
)
```

### 2. LLM_CALL
Used when an agent makes a call to a language model.

```python
AgentAction(
    action_type=AgentActionType.LLM_CALL,
    agent_name="Planner",
    llm_input="Generate a plan for movie recommendations",
    llm_output="1. Analyze user preferences\n2. Search database\n3. Rank results"
)
```

### 3. DECISION
Used when an agent makes a strategic decision or choice.

```python
AgentAction(
    action_type=AgentActionType.DECISION,
    agent_name="Coordinator",
    decision="Delegate to specialist agent",
    reasoning="Task requires domain expertise"
)
```

### 4. HANDOFF
Used when one agent hands off work to another agent.

```python
AgentAction(
    action_type=AgentActionType.HANDOFF,
    agent_name="Manager",
    decision="Transfer to QA agent",
    reasoning="Implementation complete, needs review"
)
```

### 5. ERROR
Used when an error occurs during execution.

```python
AgentAction(
    action_type=AgentActionType.ERROR,
    agent_name="Executor",
    error="API rate limit exceeded",
    decision="Retry with backoff"
)
```

## Use Case Specific Patterns

### Movie Recommendation
```python
trajectory = [
    # 1. Initial planning
    AgentAction(
        action_type=AgentActionType.DECISION,
        agent_name="Recommender",
        decision="Analyze user profile",
        reasoning="Need to understand preferences"
    ),
    # 2. Fetch user data
    AgentAction(
        action_type=AgentActionType.TOOL_CALL,
        agent_name="Recommender",
        tool_name="database_query",
        tool_input={"user_id": 123, "table": "preferences"},
        tool_output={"genres": ["sci-fi", "action"], "rating_avg": 4.2}
    ),
    # 3. Generate recommendations
    AgentAction(
        action_type=AgentActionType.LLM_CALL,
        agent_name="Recommender",
        llm_input="Based on sci-fi and action preferences, suggest 5 movies",
        llm_output="1. Inception 2. The Matrix 3. Interstellar..."
    ),
    # 4. Rank and filter
    AgentAction(
        action_type=AgentActionType.TOOL_CALL,
        agent_name="Recommender",
        tool_name="ranking_algorithm",
        tool_input={"movies": [...], "user_profile": {...}},
        tool_output={"ranked_movies": [...]}
    )
]
```

### GitHub Issue Triage
```python
trajectory = [
    # 1. Parse issue
    AgentAction(
        action_type=AgentActionType.LLM_CALL,
        agent_name="Parser",
        llm_input="Extract key information from issue: [issue text]",
        llm_output="Type: bug, Component: authentication, Priority: high"
    ),
    # 2. Check similar issues
    AgentAction(
        action_type=AgentActionType.TOOL_CALL,
        agent_name="Analyzer",
        tool_name="search_issues",
        tool_input={"keywords": ["auth", "bug"], "state": "open"},
        tool_output={"similar_issues": [234, 567]}
    ),
    # 3. Assign labels
    AgentAction(
        action_type=AgentActionType.DECISION,
        agent_name="Triager",
        decision="Apply labels: bug, auth, high-priority",
        reasoning="Based on content analysis and similar issues"
    )
]
```

### Research Summary (Multi-Agent)
```python
trajectory = [
    # 1. Coordinator delegates
    AgentAction(
        action_type=AgentActionType.DECISION,
        agent_name="Coordinator",
        decision="Assign papers to specialist agents",
        reasoning="Parallel processing for efficiency"
    ),
    # 2. First specialist analyzes
    AgentAction(
        action_type=AgentActionType.LLM_CALL,
        agent_name="Specialist_1",
        llm_input="Summarize paper: [paper 1]",
        llm_output="Key findings: ..."
    ),
    # 3. Second specialist analyzes
    AgentAction(
        action_type=AgentActionType.LLM_CALL,
        agent_name="Specialist_2",
        llm_input="Summarize paper: [paper 2]",
        llm_output="Key findings: ..."
    ),
    # 4. Handoff to synthesizer
    AgentAction(
        action_type=AgentActionType.HANDOFF,
        agent_name="Specialist_1",
        decision="Send to Synthesizer",
        reasoning="Individual analysis complete"
    ),
    # 5. Synthesize results
    AgentAction(
        action_type=AgentActionType.LLM_CALL,
        agent_name="Synthesizer",
        llm_input="Combine summaries into cohesive research summary",
        llm_output="Comprehensive summary: ..."
    )
]
```

## Framework-Specific Patterns

### LangGraph
- Emphasizes graph traversal and state transitions
- Each node transition should be a DECISION action
- Tool calls are explicit nodes in the graph

### CrewAI
- Focus on role-based agents with specific expertise
- Heavy use of HANDOFF actions between crew members
- Manager agent coordinates with DECISION actions

### AutoGen
- Conversational patterns with back-and-forth LLM_CALL actions
- Agents negotiate and reach consensus through dialogue
- Error recovery through conversation

### DSPy
- Optimized prompt chains with LLM_CALL actions
- Minimal trajectory due to optimization
- Focus on efficiency over explicit steps

## Best Practices

1. **Be Realistic**: Match the trajectory to how the framework actually works
2. **Show Coordination**: For multi-agent frameworks, show agent interactions
3. **Include Failures**: Add ERROR actions and recovery for realism
4. **Vary by Use Case**: Different use cases should have different patterns
5. **Time Matters**: Include duration_seconds for performance analysis
6. **Resource Tracking**: Add tokens and cost for resource metrics

## Implementation Example

```python
def generate_trajectory(framework_type: FrameworkType, use_case: UseCaseType) -> List[AgentAction]:
    """Generate realistic trajectory for framework and use case."""
    
    trajectory = []
    
    # Start with planning/decision
    trajectory.append(AgentAction(
        action_type=AgentActionType.DECISION,
        agent_name=f"{framework_type.value}_planner",
        decision=f"Planning approach for {use_case.value}",
        reasoning="Analyzing requirements and available resources"
    ))
    
    # Add framework-specific patterns
    if framework_type == FrameworkType.CREWAI:
        # Add crew coordination
        trajectory.append(AgentAction(
            action_type=AgentActionType.DECISION,
            agent_name="Manager",
            decision="Assigning tasks to crew members",
            reasoning="Based on agent expertise"
        ))
    
    # Add use-case specific actions
    if use_case == UseCaseType.MOVIE_RECOMMENDATION:
        trajectory.append(AgentAction(
            action_type=AgentActionType.TOOL_CALL,
            agent_name="Recommender",
            tool_name="user_profile_fetch",
            tool_input={"user_id": "..."},
            tool_output={"preferences": "..."}
        ))
    
    # Continue building trajectory...
    
    return trajectory
```

## Testing Trajectories

When implementing trajectories, test that they:

1. **Are Coherent**: Actions flow logically from one to next
2. **Match Framework Style**: Reflect the framework's architecture
3. **Complete the Task**: Lead to successful use case completion
4. **Include Metadata**: Have timing, tokens, and cost data
5. **Handle Edge Cases**: Include error scenarios

## Future Enhancements

For production implementations, consider:

- **Real execution tracking**: Capture actual framework execution
- **Trajectory replay**: Ability to replay and debug trajectories  
- **Visualization**: Graph/timeline views of trajectories
- **Comparison tools**: Side-by-side trajectory comparison
- **Pattern mining**: Identify common patterns across frameworks
