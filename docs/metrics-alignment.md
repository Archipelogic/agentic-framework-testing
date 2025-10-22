# Metrics Alignment Documentation

## Complete Table Alignment

Both the **Comprehensive Evaluation Table** and **Per-Use-Case Tables** now display the **exact same 22 metrics**, ensuring perfect consistency in framework evaluation.

### Table Structure

#### Comprehensive Table (Framework-Level)
Shows aggregated scores across ALL use cases for each framework:
- **22 metrics total** with explicit weights (summing to 1.0)
- Capability scores are **averaged across all use cases**
- Single overall score per framework

#### Per-Use-Case Tables
Shows scores for each framework on **individual use cases**:
- **Same 22 metrics** as comprehensive table
- Capability scores are **calculated specifically for that use case**
- Individual overall score per framework-use case combination

### Metrics Breakdown (Both Tables)

| Category | Metrics | Weight | Description |
|----------|---------|--------|-------------|
| **Core Capabilities** | 8 metrics | 60% | Framework's fundamental abilities |
| | Multi-Agent | 9% | Orchestrating multiple AI agents |
| | Tool Usage | 7% | External tool integration effectiveness |
| | Error Handling | 7% | Robustness and recovery from errors |
| | Context Retention | 6% | Memory and state management |
| | Adaptability | 6% | Flexibility to changing requirements |
| | Scalability | 7% | Performance under load |
| | Observability | 5% | Logging, debugging, transparency |
| | RAG | 13% | Document retrieval and grounding |
| **Enhanced Metrics** | 7 metrics | 25% | Advanced behavioral analysis |
| | Reasoning Depth | 3% | Number of reasoning steps taken |
| | Planning Score | 4% | Quality of upfront planning |
| | Decision Confidence | 3% | Certainty in decisions made |
| | Backtrack Rate ↓ | 2% | How often decisions are reversed |
| | Grounding Score | 4% | % of claims supported by context |
| | Factual Consistency | 4% | Absence of contradictions |
| | Tool Efficiency | 5% | Optimal tool usage without redundancy |
| **Resource Usage** | 3 metrics | 5% | System resource consumption |
| | Memory ↓ | 2% | RAM usage (lower is better) |
| | CPU ↓ | 2% | Processor utilization (lower is better) |
| | Cache Hit Rate | 1% | Data reuse efficiency |
| **Performance** | 3 metrics | 10% | Core performance indicators |
| | Success Rate | 7% | % of tests passing |
| | Latency ↓ | 2% | Speed (lower is better) |
| | Cost ↓ | 1% | Token costs (lower is better) |

**Note:** Metrics marked with ↓ are inverse - lower values result in higher scores.

### Capability Score Calculation

For **per-use-case tables**, capability scores are calculated from that specific test's metrics:

```python
capabilities = {
    'multi_agent': success_rate + decision_confidence / 2,
    'tool_usage': tool_efficiency_score,
    'error_handling': factual_consistency,
    'context_retention': grounding_score,  # min 10% for mock
    'adaptability': planning_score - backtrack_rate,
    'scalability': (memory_efficiency + cpu_efficiency) / 2,
    'observability': reasoning_depth * 5,  # scaled
    'rag_capability': grounding_score  # min 10% for mock
}
```

### Visual Features

Both tables include:
- **Color coding**: Green (≥80), Blue (50-79), Red (<50)
- **Sticky framework column**: Stays visible while scrolling
- **Grouped headers**: Clear metric categorization
- **Weighted percentages**: Shown in headers
- **Tooltips**: Hover for metric descriptions

### Data Flow

1. **Tests execute** → Enhanced metrics calculated
2. **Per use case** → Capability scores derived from test metrics
3. **Framework level** → Capability scores averaged across use cases
4. **Overall score** → Weighted sum of all 22 metrics

This ensures complete transparency and consistency in how frameworks are evaluated, whether viewed at the framework level or individual use case level.
