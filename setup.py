from setuptools import setup, find_packages

setup(
    name="agentic-testing-harness",
    version="1.0.0",
    description="Comprehensive testing harness for evaluating agentic AI frameworks",
    author="Your Team",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.5.0",
        "rich>=13.0.0",
        "tabulate>=0.9.0",
        "jinja2>=3.1.0",
        "tqdm>=4.65.0",
        "click>=8.1.0",
        "pyyaml>=6.0",
        "jsonschema>=4.19.0",
    ],
    extras_require={
        "frameworks": [
            "langgraph>=0.1.0",
            "crewai>=0.28.0",
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "tiktoken>=0.5.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.11.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ]
    },
    entry_points={
        'console_scripts': [
            'agentic-test=src.main:main',
        ],
    },
    python_requires=">=3.8",
)
