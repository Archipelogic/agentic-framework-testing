# 🚀 Agentic AI Framework Testing Harness

**Comprehensive evaluation system for comparing 13+ agentic AI frameworks across standardized use cases with 22 evaluation metrics.**

---

## 🎯 Overview

This testing harness provides **systematic, data-driven comparison** of agentic AI frameworks including LangGraph, CrewAI, AutoGen, and 10+ others. Features comprehensive metrics (capabilities, performance, resources), beautiful HTML reports, and both mock and live testing modes.

### ✨ Key Features

- **13 Frameworks** - LangGraph, CrewAI, AutoGen, Pydantic AI, Haystack, LlamaIndex, DSPy, and more
- **5 Use Cases** - Movie recommendations, GitHub triage, recipe generation, research summary, email automation  
- **22 Metrics** - Capabilities (8), Enhanced metrics (7), Resources (3), Performance (4)
- **Beautiful Reports** - Interactive HTML dashboards with sortable tables and visualizations
- **Mock & Live Modes** - Test without API keys using mock data, or with real APIs
- **Unified Runner** - Single script for all operations with clean CLI

---

## 📦 Clean Project Structure

```
agentic-framework-testing/
├── src/
│   ├── core/                    # Core types and config
│   ├── adapters/                # Framework adapters (mock + live)
│   ├── use_cases/               # 5 evaluation scenarios
│   ├── metrics/                 # Evaluation & enhanced metrics
│   ├── reporting/               # HTML report generation
│   ├── utils/                   # Data generation utilities
│   └── cli/                     # Command-line helpers
├── data/                        # Test data & ground truth
├── docs/                        # Comprehensive documentation
├── tests/                       # Integration tests
├── scripts/                     # Data generation scripts
├── benchmark_results/           # Output reports
├── run_evaluation.py            # Main evaluation runner
├── run.sh                       # Unified shell script
└── requirements.txt             # Dependencies
```

---

## 🚀 Quick Start

### Setup and Run

```bash
# Clone and enter directory
git clone https://github.com/your-org/agentic-framework-testing
cd agentic-framework-testing

# One-time setup
./run.sh setup

# Run evaluation (mock mode - no API keys needed)
./run.sh test

# Run with real APIs (requires .env file with API keys)
./run.sh test --live

# Open latest report
./run.sh report
```

### Available Commands

| Command | Description |
|---------|-------------|
| `./run.sh setup` | Install dependencies and prepare environment |
| `./run.sh test` | Run mock evaluation (default) |
| `./run.sh test --live` | Run with real API calls |
| `./run.sh test --quick` | Quick mode: 3 frameworks, 2 use cases |
| `./run.sh test --samples 50` | Run with 50 test samples per use case |
| `./run.sh report` | Open latest report in browser |
| `./run.sh clean` | Clean all benchmark results |
| `./run.sh help` | Show help message |

---

## 🌐 Evaluation Metrics

### Capability Scores (60% weight)
- **Multi-Agent** - Orchestrating multiple AI agents
- **Tool Usage** - External tool integration
- **Error Handling** - Robustness and recovery
- **Context Retention** - Memory management
- **Adaptability** - Flexibility to requirements
- **Scalability** - Performance under load
- **Observability** - Logging and debugging
- **RAG** - Document retrieval and grounding

### Enhanced Metrics (25% weight)
- **Reasoning Analysis** - Depth, planning quality, decision confidence
- **Hallucination Detection** - Grounding score, factual consistency
- **Tool Efficiency** - Optimal tool usage without redundancy

### Performance & Resources (15% weight)
- **Success Rate** - Test passing percentage
- **Latency** - Processing speed
- **Cost** - Token usage costs
- **Memory/CPU** - Resource consumption

---

## 📋 Use Cases

1. **Movie Recommendation** - Recommend 5 movies based on user preferences
2. **GitHub Issue Triage** - Classify and prioritize GitHub issues
3. **Recipe Generation** - Create recipes from ingredients and constraints
4. **Research Summary** - Summarize academic papers and extract themes
5. **Email Automation** - Classify and generate email responses

---

## 🤝 Contributing

Contributions are welcome! Please check out our [contributing guide](docs/contributing.md) for details.

## 📋 License

MIT License - See LICENSE file for details.

## 📂 Documentation

Comprehensive documentation available in the `docs/` folder:

### Getting Started
- [README](docs/README.md) - Documentation hub with navigation
- [Setup Guide](docs/setup.md) - Installation and environment setup
- [Quick Start](docs/quickstart.md) - 5-minute quick start guide
- [Configuration](docs/configuration.md) - API keys and model configuration

### Usage Guides
- [Use Cases](docs/use-cases.md) - Understanding evaluation scenarios
- [Frameworks](docs/frameworks.md) - Supported frameworks and features
- [Evaluation Metrics](docs/evaluation.md) - Metrics, scoring, and fairness
- [Agent Trajectories](docs/agent-trajectories.md) - Understanding agent execution traces

### Development
- [API Reference](docs/api-reference.md) - Complete API documentation
- [Best Practices](docs/best-practices.md) - Ensuring fair evaluations
- [Deployment](docs/deployment.md) - Running benchmarks in production
- [Contributing](docs/contributing.md) - How to contribute
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## 🚀 Ready to Start?

```bash
# Setup and run your first evaluation
./run.sh setup
./run.sh test
```

Happy benchmarking!
