# Agentic AI Framework Evaluation Documentation

## Welcome to the Agentic AI Framework Testing Harness

This comprehensive testing platform enables standardized evaluation and comparison of 13+ agentic AI frameworks across multiple real-world use cases.

```bash
# Clone the repository
git clone https://github.com/your-org/agentic-framework-testing
cd agentic-framework-testing

# Setup and run your first evaluation
./run.sh setup   # One-time setup
./run.sh test    # Run mock evaluation
./run.sh report  # View results
```

## 📋 Complete Documentation Index

### 🎯 Getting Started
| Document | Description |
|----------|-------------|
| [**Setup Guide**](setup.md) | Complete installation and environment setup |
| [**Quick Start**](quickstart.md) | Get running in 5 minutes with examples |
| [**Configuration**](configuration.md) | API keys, AWS Bedrock, model settings |

### 📊 Core Concepts
| Document | Description |
|----------|-------------|
| [**Use Cases**](use-cases.md) | Understanding the 5 evaluation scenarios |
| [**Frameworks**](frameworks.md) | All 13 supported agentic AI frameworks |
| [**Evaluation Metrics**](evaluation.md) | 22 metrics: capabilities, performance, quality |
| [**Agent Trajectories**](agent-trajectories.md) | Understanding agent execution traces |

### 🛠️ Advanced Topics
| Document | Description |
|----------|-------------|
| [**Deployment**](deployment.md) | Production deployment and scaling |
| [**Best Practices**](best-practices.md) | Ensuring fair and accurate evaluations |
| [**API Reference**](api-reference.md) | Complete code documentation |
| [**Hugging Face Datasets**](huggingface-datasets.md) | Using real-world data for testing |
| [**Metrics Alignment**](metrics-alignment.md) | How metrics flow through the system |

### 🤝 Support & Contributing
| Document | Description |
|----------|-------------|
| [**Troubleshooting**](troubleshooting.md) | Common issues and solutions |
| [**Contributing**](contributing.md) | How to contribute to the project |

## 🎨 What Can This Do?

### Evaluate 13 Frameworks
- **Graph-Based**: LangGraph, Haystack
- **Multi-Agent**: CrewAI, AutoGen, BeeAI
- **Type-Safe**: Pydantic AI, Atomic Agents
- **Optimized**: DSPy, Agno, Smolagents
- **Cloud-Native**: AWS Bedrock AgentCore, Strands
- **RAG-Focused**: LlamaIndex

### Across 5 Real-World Use Cases
1. **Movie Recommendation** - Personalized recommendations
2. **GitHub Issue Triage** - Issue classification and routing
3. **Recipe Generation** - Creative recipe creation
4. **Research Summary** - Academic paper analysis
5. **Email Automation** - Smart email handling

### With 22 Comprehensive Metrics
- **8 Capability Scores** (60% weight)
- **7 Enhanced Metrics** (25% weight)  
- **3 Resource Metrics** (5% weight)
- **4 Performance Metrics** (10% weight)

## 💡 Key Commands

| Command | Description |
|---------|-------------|
| `./run.sh setup` | Install dependencies and prepare environment |
| `./run.sh test` | Run mock evaluation (no API keys needed) |
| `./run.sh test --live` | Run with real API calls |
| `./run.sh test --quick` | Quick test: 3 frameworks, 2 use cases |
| `./run.sh test --samples 50` | Test with 50 samples per use case |
| `./run.sh report` | Open latest report in browser |
| `./run.sh clean` | Clean all benchmark results |

## 📖 Recommended Reading Path

### For New Users
1. [Quick Start Guide](quickstart.md) - Get running fast
2. [Use Cases](use-cases.md) - Understand what we're testing
3. [Frameworks](frameworks.md) - Learn about each framework
4. [Evaluation Metrics](evaluation.md) - How we score and compare

### For Developers
1. [API Reference](api-reference.md) - Code structure and APIs
2. [Agent Trajectories](agent-trajectories.md) - How agents execute
3. [Contributing](contributing.md) - Add frameworks or features
4. [Best Practices](best-practices.md) - Write quality code

### For Production Use
1. [Configuration](configuration.md) - Set up API keys and AWS
2. [Deployment](deployment.md) - Scale to production
3. [Metrics Alignment](metrics-alignment.md) - Understand data flow
4. [Troubleshooting](troubleshooting.md) - Fix common issues

## 🏗️ Architecture Overview

```
agentic-framework-testing/
├── run.sh                    # Unified runner script
├── run_evaluation.py         # Main evaluation engine
├── src/
│   ├── adapters/            # Framework adapters
│   ├── core/                # Core types and config
│   ├── metrics/             # Evaluation metrics
│   ├── reporting/           # Report generation
│   └── use_cases/           # Test scenarios
├── data/                    # Auto-generated test data
├── benchmark_results/       # Evaluation outputs
└── docs/                    # This documentation
```

## 🤝 Contributing

We welcome contributions! See [Contributing Guide](contributing.md) for:
- Adding new frameworks
- Creating use cases
- Improving metrics
- Fixing bugs
- Writing documentation

## 📝 License

MIT License - See the main repository LICENSE file.
- **Contributing**: See [Contributing Guide](./contributing.md)

---

*Last Updated: October 2024*
