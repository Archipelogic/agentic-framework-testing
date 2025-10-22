#!/usr/bin/env python3
"""
COMPLETE data generation for GitHub Triage use case.
Generates BOTH test cases AND ground truth with all required fields.
Self-contained script that addresses ALL agent requirements.
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from datasets import load_dataset


class GitHubTriageDataGenerator:
    """Generate complete, realistic data for GitHub triage use case."""
    
    def __init__(self):
        self.test_cases = []
        self.ground_truth = []
        self.dataset = None
        
        # Components and features
        self.components = [
            "authentication", "database", "API", "frontend", "backend",
            "cache", "queue", "scheduler", "logger", "router",
            "middleware", "validator", "parser", "serializer", "encoder"
        ]
        
        self.errors = [
            "NullPointerException", "IndexOutOfBoundsException", "TypeError",
            "ValueError", "KeyError", "AttributeError", "ConnectionError",
            "TimeoutError", "PermissionError", "FileNotFoundError"
        ]
        
        self.actions = [
            "uploading files", "processing data", "connecting to server",
            "saving configuration", "loading resources", "parsing JSON",
            "validating input", "executing query", "rendering page"
        ]
        
        # GitHub usernames (varied)
        self.usernames = self.generate_usernames()
        
        # Teams for routing
        self.teams = [
            "core", "platform", "infrastructure", "security", "frontend",
            "backend", "mobile", "devops", "qa", "documentation",
            "community", "api", "data", "ml", "performance"
        ]
        
        # Common labels
        self.label_pool = [
            "bug", "enhancement", "documentation", "question", "help wanted",
            "good first issue", "wontfix", "duplicate", "invalid", "confirmed",
            "needs-repro", "needs-info", "ready", "in-progress", "blocked",
            "high-priority", "low-priority", "breaking-change", "regression",
            "security", "performance", "accessibility", "i18n", "testing"
        ]
    
    def generate_usernames(self) -> List[str]:
        """Generate diverse GitHub usernames."""
        prefixes = ["dev", "code", "tech", "bug", "feature", "open", "pro", "new", "senior", "junior"]
        suffixes = ["master", "wizard", "ninja", "guru", "expert", "fan", "contributor", "hacker", "coder"]
        numbers = ["", "123", "42", "007", "99", "2024", "88", "11"]
        
        usernames = []
        # Real-looking names
        first_names = ["alex", "sarah", "mike", "jenn", "david", "lisa", "robert", "maria", "james", "patricia",
                      "william", "linda", "richard", "barbara", "joseph", "susan", "thomas", "jessica"]
        last_initials = list("abcdefghijklmnopqrstuvwxyz")
        
        for first in first_names:
            for initial in random.sample(last_initials, 3):
                usernames.append(f"{first}{initial}")
                usernames.append(f"{first}_{initial}")
                usernames.append(f"{first}{initial}{random.choice(numbers)}")
        
        # Tech usernames
        for _ in range(20):
            usernames.append(f"{random.choice(prefixes)}_{random.choice(suffixes)}{random.choice(numbers)}")
        
        return usernames
    
    def load_github_issues(self):
        """Try to load real GitHub issues dataset."""
        print("🐙 Loading GitHub issues dataset...")
        try:
            self.dataset = load_dataset(
                "lewtun/github-issues", 
                split="train[:2000]",
                cache_dir=".cache"
            )
            print(f"✅ Loaded {len(self.dataset)} real issues")
        except Exception as e:
            print(f"⚠️ Could not load dataset: {e}")
            print("   Using synthetic generation instead")
            self.dataset = None
    
    def generate_issue_title(self, issue_type: str) -> str:
        """Generate realistic issue title."""
        templates = {
            "bug": [
                f"{random.choice(self.components)} throws {random.choice(self.errors)} when {random.choice(self.actions)}",
                f"Unable to {random.choice(self.actions).replace('ing', '')} in {random.choice(self.components)}",
                f"{random.choice(self.components)} not working after update",
                f"Crash when {random.choice(self.actions)} with large files",
                f"Memory leak in {random.choice(self.components)}",
                f"{random.choice(self.components)} returns incorrect results",
                f"Regression: {random.choice(self.components)} broken in v3.0"
            ],
            "feature": [
                f"Add support for {random.choice(['async', 'batch', 'parallel'])} {random.choice(self.actions)}",
                f"Feature request: Better {random.choice(self.components)} configuration",
                f"Enhancement: Add {random.choice(['caching', 'logging', 'metrics'])} to {random.choice(self.components)}",
                f"Support for {random.choice(['WebSocket', 'gRPC', 'GraphQL'])} in API",
                f"Add option to customize {random.choice(self.components)}",
                f"Implement {random.choice(['retry logic', 'rate limiting', 'circuit breaker'])}"
            ],
            "question": [
                f"How to configure {random.choice(self.components)}?",
                f"Best practice for {random.choice(self.actions)}?",
                f"Is it possible to {random.choice(self.actions).replace('ing', '')} with custom settings?",
                f"Documentation unclear about {random.choice(self.components)}",
                f"What is the recommended way to handle {random.choice(['errors', 'timeouts', 'retries'])}?"
            ],
            "docs": [
                f"Update documentation for {random.choice(self.components)}",
                f"Fix typo in {random.choice(['README', 'API docs', 'guide'])}",
                f"Missing documentation for {random.choice(self.components)} configuration",
                f"Outdated example in {random.choice(['quickstart', 'tutorial', 'reference'])}",
                f"Add code examples for {random.choice(self.components)}"
            ]
        }
        return random.choice(templates[issue_type])
    
    def generate_issue_body(self, issue_type: str) -> str:
        """Generate realistic issue body."""
        if issue_type == "bug":
            body = f"""## Description
When attempting to {random.choice(self.actions)}, the {random.choice(self.components)} component fails unexpectedly.

## Steps to Reproduce
1. Start the application
2. Navigate to the {random.choice(['dashboard', 'settings', 'profile'])} page  
3. Attempt to {random.choice(self.actions).replace('ing', '')}
4. Observe the error

## Expected Behavior
The operation should complete successfully without errors.

## Actual Behavior
The system throws a {random.choice(self.errors)}.

## Environment
- OS: {random.choice(['Ubuntu 22.04', 'Windows 11', 'macOS 14.0'])}
- Version: {random.choice(['3.2.1', '2.8.0', '4.0.0-beta'])}
- Python: {random.choice(['3.8', '3.9', '3.10', '3.11'])}"""
            
        elif issue_type == "feature":
            body = f"""## Problem Statement
Currently, there's no way to {random.choice(self.actions)} efficiently when dealing with large datasets.

## Proposed Solution
Add a new option to {random.choice(self.components)} that allows users to configure batch processing with customizable chunk sizes.

## Alternatives Considered
- Using external tools (not integrated)
- Workaround with current functionality (too slow)
- Manual processing (error-prone)

## Use Cases
- Processing large CSV files
- Bulk API operations
- Data migration tasks"""
            
        elif issue_type == "question":
            body = f"""## Question
I'm trying to {random.choice(self.actions)} but I'm not sure about the best approach.

## Context
Working on a project that requires {random.choice(['high performance', 'scalability', 'reliability'])}.

## What I've Tried
- Checked the documentation
- Searched existing issues
- Tried different configurations

## Code Sample
```python
# Current approach
config = {{
    'option1': '?',
    'option2': '?'
}}
```"""
        else:  # docs
            body = f"""## Documentation Issue
The current documentation for {random.choice(self.components)} is missing important details about configuration options.

## Suggested Change
Add a complete example showing how to configure all available options with explanations.

## Location
File: `docs/api/{random.choice(self.components)}.md`"""
        
        return body
    
    def determine_priority(self, issue_type: str, body: str) -> str:
        """Determine priority based on content."""
        body_lower = body.lower()
        
        # P0 - Critical (5%)
        if any(word in body_lower for word in ["production", "critical", "security", "data loss"]):
            return "P0"
        
        # P1 - High (20%)
        if issue_type == "bug" and any(word in body_lower for word in ["error", "broken", "failing"]):
            return random.choices(["P1", "P2"], weights=[70, 30])[0]
        
        # P3 - Low (25%)
        if issue_type in ["docs", "question"]:
            return random.choices(["P2", "P3"], weights=[40, 60])[0]
        
        # P2 - Medium (50% default)
        return random.choices(["P1", "P2", "P3"], weights=[15, 60, 25])[0]
    
    def determine_team(self, title: str, body: str) -> str:
        """Determine team assignment."""
        text = (title + " " + body).lower()
        
        if any(word in text for word in ["api", "endpoint", "rest"]):
            return "api"
        elif any(word in text for word in ["ui", "frontend", "react"]):
            return "frontend"
        elif any(word in text for word in ["backend", "server", "database"]):
            return "backend"
        elif any(word in text for word in ["security", "auth", "permission"]):
            return "security"
        elif any(word in text for word in ["performance", "slow", "memory"]):
            return "performance"
        elif any(word in text for word in ["docs", "documentation"]):
            return "documentation"
        else:
            return random.choice(self.teams)
    
    def select_labels(self, issue_type: str, priority: str, team: str) -> List[str]:
        """Select appropriate labels."""
        labels = [issue_type]
        
        if priority in ["P0", "P1"]:
            labels.append("high-priority")
        elif priority == "P3":
            labels.append("low-priority")
        
        if team in ["frontend", "backend", "api", "security"]:
            labels.append(team)
        
        # Add 0-3 additional labels
        additional = random.sample(
            ["needs-review", "confirmed", "help wanted", "good first issue"],
            random.randint(0, 3)
        )
        labels.extend(additional)
        
        return list(set(labels))
    
    def generate(self, num_test_cases: int = 200):
        """Generate test cases and ground truth."""
        self.load_github_issues()
        
        print(f"\n🔧 Generating {num_test_cases} test cases...")
        
        # Issue type distribution
        type_weights = {"bug": 40, "feature": 30, "question": 20, "docs": 10}
        
        for test_idx in range(num_test_cases):
            # Try to use real data if available
            if self.dataset and test_idx < len(self.dataset):
                issue_data = self.dataset[test_idx]
                if isinstance(issue_data, dict):
                    title = issue_data.get('title', '') or ''
                    title = title[:200] if title else self.generate_issue_title('bug')
                    body = issue_data.get('body', '') or ''
                    body = body[:2000] if body else self.generate_issue_body('bug')
                    # Determine type from labels or content
                    labels = issue_data.get('labels', [])
                    if 'bug' in str(labels).lower():
                        issue_type = 'bug'
                    elif 'feature' in str(labels).lower() or 'enhancement' in str(labels).lower():
                        issue_type = 'feature'
                    elif 'question' in str(labels).lower():
                        issue_type = 'question'
                    elif 'doc' in str(labels).lower():
                        issue_type = 'docs'
                    else:
                        issue_type = random.choices(list(type_weights.keys()), weights=list(type_weights.values()))[0]
                else:
                    issue_type = random.choices(list(type_weights.keys()), weights=list(type_weights.values()))[0]
                    title = self.generate_issue_title(issue_type)
                    body = self.generate_issue_body(issue_type)
            else:
                # Generate synthetic
                issue_type = random.choices(list(type_weights.keys()), weights=list(type_weights.values()))[0]
                title = self.generate_issue_title(issue_type)
                body = self.generate_issue_body(issue_type)
            
            # Metadata
            author = random.choice(self.usernames)
            created_at = (datetime.now() - timedelta(days=random.randint(0, 90))).isoformat() + "Z"
            
            current_labels = []
            if random.random() < 0.3:
                current_labels = random.sample(self.label_pool, random.randint(1, 3))
            
            # Create test case
            test_case = {
                "test_id": f"github_{test_idx:03d}",
                "issue_id": 1000 + test_idx,
                "title": title,
                "body": body,
                "author": author,
                "created_at": created_at,
                "labels_current": current_labels
            }
            self.test_cases.append(test_case)
            
            # Generate ground truth
            priority = self.determine_priority(issue_type, body)
            team = self.determine_team(title, body)
            labels = self.select_labels(issue_type, priority, team)
            
            ground_truth = {
                "test_id": f"github_{test_idx:03d}",
                "expected_classification": issue_type,
                "expected_priority": priority,
                "expected_team": team,
                "expected_labels": labels
            }
            self.ground_truth.append(ground_truth)
            
            if (test_idx + 1) % 50 == 0:
                print(f"  Generated {test_idx + 1}/{num_test_cases} test cases...")
        
        print(f"✅ Generated {len(self.test_cases)} test cases")
    
    def validate(self):
        """Validate generated data."""
        print("\n🔍 Validating data...")
        
        issues = []
        for i, (test, truth) in enumerate(zip(self.test_cases, self.ground_truth)):
            if not test.get('title'):
                issues.append(f"Test {i}: No title")
            if not test.get('body'):
                issues.append(f"Test {i}: No body")
            if truth.get('expected_classification') not in ['bug', 'feature', 'question', 'docs']:
                issues.append(f"Truth {i}: Invalid classification")
        
        if issues:
            print(f"⚠️ Found {len(issues)} issues")
        else:
            print("✅ All validation checks passed!")
        
        # Statistics
        print("\n📊 Data Statistics:")
        types = [gt['expected_classification'] for gt in self.ground_truth]
        priorities = [gt['expected_priority'] for gt in self.ground_truth]
        
        print(f"  Types: bug={types.count('bug')}, feature={types.count('feature')}, "
              f"question={types.count('question')}, docs={types.count('docs')}")
        print(f"  Priorities: P0={priorities.count('P0')}, P1={priorities.count('P1')}, "
              f"P2={priorities.count('P2')}, P3={priorities.count('P3')}")
    
    def save(self, output_dir: str = "data/github_triage"):
        """Save test cases and ground truth."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save files
        with open(f"{output_dir}/test_cases.json", 'w') as f:
            json.dump(self.test_cases, f, indent=2)
        print(f"📁 Saved test cases to {output_dir}/test_cases.json")
        
        with open(f"{output_dir}/ground_truth.json", 'w') as f:
            json.dump(self.ground_truth, f, indent=2)
        print(f"📁 Saved ground truth to {output_dir}/ground_truth.json")
        
        # Show sample
        if self.test_cases:
            print("\n📋 Sample Test Case:")
            sample = self.test_cases[0]
            print(f"  ID: {sample['test_id']}")
            print(f"  Title: {sample['title'][:60]}...")
            print(f"  Author: {sample['author']}")


def main():
    """Generate GitHub triage data."""
    print("=" * 60)
    print("🐙 GITHUB TRIAGE DATA GENERATOR")
    print("=" * 60)
    print("Generating COMPLETE data for GitHub triage use case")
    print("This includes:")
    print("  ✓ Realistic issue titles and bodies")
    print("  ✓ Bug reports, feature requests, questions, docs")
    print("  ✓ Priority levels (P0-P3)")
    print("  ✓ Team routing assignments")
    print("  ✓ Label suggestions")
    print("=" * 60)
    
    generator = GitHubTriageDataGenerator()
    generator.generate(num_test_cases=200)
    generator.validate()
    generator.save()
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE DATA GENERATION FINISHED")
    print("=" * 60)
    print("The GitHub triage use case now has:")
    print("  • Realistic test cases with varied issue types")
    print("  • Comprehensive ground truth for evaluation")
    print("  • All fields required by the agents")
    print("\nReady for testing with all frameworks!")


if __name__ == "__main__":
    main()
