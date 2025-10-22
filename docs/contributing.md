# Contributing Guide

## Welcome Contributors!

We welcome contributions to the Agentic AI Framework Testing Harness! This guide will help you get started.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Process](#development-process)
3. [Code Standards](#code-standards)
4. [Testing Requirements](#testing-requirements)
5. [Pull Request Process](#pull-request-process)
6. [Adding New Features](#adding-new-features)

## Getting Started

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/your-username/agentic-framework-testing.git
cd agentic-framework-testing
```

3. Add upstream remote:

```bash
git remote add upstream https://github.com/original/agentic-framework-testing.git
```

### Development Environment

1. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install in development mode:

```bash
pip install -e .
pip install -r requirements-dev.txt
```

3. Install pre-commit hooks:

```bash
pre-commit install
```

### Running Tests

Before making changes, ensure all tests pass:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_integration.py -v
```

## Development Process

### Branch Strategy

We follow the Git Flow branching model:

- `main` - Stable, production-ready code
- `develop` - Integration branch for features
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `hotfix/*` - Emergency fixes for production

### Creating a Feature Branch

```bash
# Update develop branch
git checkout develop
git pull upstream develop

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add new feature"

# Push to your fork
git push origin feature/your-feature-name
```

### Commit Message Format

We follow the Conventional Commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

**Examples:**

```bash
# Feature
git commit -m "feat(adapters): add support for new framework"

# Bug fix
git commit -m "fix(evaluator): correct accuracy calculation"

# Documentation
git commit -m "docs(readme): update installation instructions"

# With body
git commit -m "feat(benchmark): add parallel execution

- Implement ThreadPoolExecutor for parallel runs
- Add max_workers configuration
- Update progress tracking for parallel execution

Closes #123"
```

## Code Standards

### Python Style Guide

We follow PEP 8 with these additions:

1. **Line Length**: Maximum 100 characters (exceptions for URLs, strings)
2. **Imports**: Grouped and sorted using `isort`
3. **Type Hints**: Required for all public functions
4. **Docstrings**: Google style for all public classes and functions

### Code Formatting

Use Black for automatic formatting:

```bash
# Format single file
black src/benchmark/runner.py

# Format entire project
black src/ tests/

# Check without modifying
black --check src/
```

### Type Checking

Use mypy for static type checking:

```bash
# Check types
mypy src/

# Strict mode
mypy --strict src/
```

### Linting

Use flake8 for linting:

```bash
# Lint code
flake8 src/ tests/

# With specific rules
flake8 --select=E9,F63,F7,F82 src/
```

### Example Code Style

```python
"""Module docstring describing purpose."""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging

# Constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3

logger = logging.getLogger(__name__)


@dataclass
class ExampleClass:
    """
    Brief description of class.
    
    Longer description if needed, explaining purpose,
    usage, and any important details.
    
    Attributes:
        name: Description of name attribute.
        value: Description of value attribute.
    
    Example:
        >>> example = ExampleClass("test", 42)
        >>> example.process()
        "Processed: test with value 42"
    """
    
    name: str
    value: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def process(self) -> str:
        """
        Process the example data.
        
        Returns:
            Processed string representation.
        
        Raises:
            ValueError: If value is negative.
        """
        if self.value < 0:
            raise ValueError("Value must be non-negative")
        
        return f"Processed: {self.name} with value {self.value}"
    
    def _internal_method(self) -> None:
        """Internal method (note the single underscore)."""
        pass


def public_function(
    param1: str,
    param2: Optional[int] = None,
    *,  # Force keyword-only arguments after this
    param3: bool = False
) -> Dict[str, Union[str, int]]:
    """
    Brief description of function.
    
    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to None.
        param3: Description of param3. Defaults to False.
    
    Returns:
        Dictionary containing processed results with keys:
        - 'status': Processing status
        - 'value': Processed value
    
    Raises:
        ValueError: If param1 is empty.
        TypeError: If param2 is not an integer.
    
    Example:
        >>> result = public_function("test", param2=5)
        >>> print(result['status'])
        'success'
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    
    result: Dict[str, Union[str, int]] = {
        'status': 'success',
        'value': param2 or 0
    }
    
    if param3:
        logger.info(f"Processing with param3 enabled: {param1}")
    
    return result
```

## Testing Requirements

### Test Structure

```
tests/
├── unit/                 # Unit tests
│   ├── test_adapters.py
│   ├── test_evaluator.py
│   └── test_config.py
├── integration/         # Integration tests
│   ├── test_end_to_end.py
│   └── test_frameworks.py
├── performance/        # Performance tests
│   └── test_benchmarks.py
└── fixtures/          # Test fixtures
    └── test_data.json
```

### Writing Tests

```python
"""Test module for example functionality."""

import pytest
from unittest.mock import Mock, patch
from src.example import ExampleClass


class TestExampleClass:
    """Test cases for ExampleClass."""
    
    @pytest.fixture
    def example_instance(self):
        """Create example instance for testing."""
        return ExampleClass("test", 42)
    
    def test_initialization(self):
        """Test class initialization."""
        instance = ExampleClass("test", 42)
        assert instance.name == "test"
        assert instance.value == 42
        assert instance.metadata == {}
    
    def test_process_success(self, example_instance):
        """Test successful processing."""
        result = example_instance.process()
        assert result == "Processed: test with value 42"
    
    def test_process_negative_value_raises(self):
        """Test that negative values raise ValueError."""
        instance = ExampleClass("test", -1)
        with pytest.raises(ValueError, match="non-negative"):
            instance.process()
    
    @pytest.mark.parametrize("name,value,expected", [
        ("test1", 1, "Processed: test1 with value 1"),
        ("test2", 100, "Processed: test2 with value 100"),
        ("", 0, "Processed:  with value 0"),
    ])
    def test_process_various_inputs(self, name, value, expected):
        """Test processing with various inputs."""
        instance = ExampleClass(name, value)
        assert instance.process() == expected
    
    @patch('src.example.external_api_call')
    def test_with_mock(self, mock_api, example_instance):
        """Test with mocked external dependency."""
        mock_api.return_value = {"status": "ok"}
        
        # Test code that uses external_api_call
        result = example_instance.method_using_api()
        
        mock_api.assert_called_once()
        assert result is not None


@pytest.mark.integration
class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.slow
    def test_full_workflow(self):
        """Test complete workflow (marked as slow)."""
        # Integration test code
        pass
```

### Test Coverage Requirements

- Minimum coverage: 80%
- New features must include tests
- Bug fixes should include regression tests

## Pull Request Process

### Before Submitting

1. **Update from upstream:**
```bash
git fetch upstream
git rebase upstream/develop
```

2. **Run all checks:**
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Run linting
flake8 src/ tests/

# Type checking
mypy src/

# Run tests
pytest tests/ --cov=src
```

3. **Update documentation:**
- Update relevant docs in `docs/`
- Update docstrings
- Update README if needed

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change
- [ ] Documentation update

## Changes Made
- Change 1
- Change 2

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Added new tests for changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings

## Related Issues
Closes #123
```

### Review Process

1. PR must pass all CI checks
2. Requires at least one reviewer approval
3. Maintainer will merge after approval

## Adding New Features

### Adding a New Framework

1. **Create adapter class:**

```python
# src/adapters/newframework_adapter.py
from src.adapters.base import BaseFrameworkAdapter

class NewFrameworkAdapter(BaseFrameworkAdapter):
    """Adapter for NewFramework."""
    # Implementation
```

2. **Add to FrameworkType enum:**

```python
# src/core/types.py
class FrameworkType(str, Enum):
    # ... existing frameworks
    NEW_FRAMEWORK = "new_framework"
```

3. **Update factory method:**

```python
# src/adapters/__init__.py
def create_adapter(framework: FrameworkType, config=None):
    # ... existing code
    elif framework == FrameworkType.NEW_FRAMEWORK:
        return NewFrameworkAdapter(config)
```

4. **Add tests:**

```python
# tests/unit/test_newframework_adapter.py
def test_newframework_adapter():
    # Test implementation
```

5. **Update documentation:**
- Add to `docs/frameworks.md`
- Update README
- Add configuration examples

### Adding a New Use Case

1. **Define use case type:**

```python
# src/core/types.py
class UseCaseType(str, Enum):
    # ... existing use cases
    NEW_USE_CASE = "new_use_case"
```

2. **Create evaluator:**

```python
# src/use_cases/new_use_case.py
class NewUseCaseEvaluator:
    """Evaluator for new use case."""
    # Implementation
```

3. **Add test data:**

```bash
# Create data files
data/new_use_case/test_cases.json
data/new_use_case/ground_truth.json
```

4. **Update adapters to support the use case**

5. **Add tests and documentation**

## Documentation

### Documentation Standards

1. **Docstrings**: All public APIs must have docstrings
2. **Examples**: Include usage examples in docstrings
3. **Type Hints**: Use type hints for clarity
4. **README**: Keep README up to date
5. **Guides**: Update relevant guides in `docs/`

### Building Documentation

```bash
# Build Sphinx documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## Community

### Code of Conduct

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Please read and follow it.

### Getting Help

- **Discord**: Join our Discord server
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and discussions
- **Email**: maintainers@example.com

### Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

## Release Process

### Version Numbering

We use Semantic Versioning (SemVer):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Run full test suite
4. Build and test package
5. Tag release
6. Create GitHub release
7. Publish to PyPI

## License

By contributing, you agree that your contributions will be licensed under the project's license.

## Thank You!

Thank you for contributing to the Agentic AI Framework Testing Harness! Your contributions help make this tool better for everyone.

## Quick Links

- [Issue Tracker](https://github.com/project/issues)
- [Project Board](https://github.com/project/projects)
- [Discord Server](https://discord.gg/example)
- [Documentation](./README.md)
