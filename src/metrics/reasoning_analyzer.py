"""
Reasoning analysis for framework execution traces.
Analyzes decision depth, planning quality, and problem decomposition.
"""

import re
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ReasoningStep:
    """Represents a single reasoning step."""
    content: str
    depth: int
    type: str  # 'decision', 'analysis', 'planning', 'execution'
    is_backtrack: bool = False


class ReasoningAnalyzer:
    """Analyze reasoning patterns in execution traces."""
    
    def __init__(self):
        """Initialize reasoning analyzer."""
        self.reasoning_indicators = {
            'decision': ['decided', 'choosing', 'selected', 'picked'],
            'analysis': ['analyzing', 'considering', 'evaluating', 'examining'],
            'planning': ['planning', 'will', 'going to', 'next step'],
            'execution': ['executing', 'running', 'calling', 'performing']
        }
        
        self.backtrack_indicators = [
            'retry', 'failed', 'error', 'incorrect', 'wrong',
            'back to', 'reconsider', 'alternative', 'instead'
        ]
    
    def analyze_trace(self, trace: Any) -> Dict[str, Any]:
        """Analyze reasoning patterns in execution trace."""
        # Handle different trace formats
        if isinstance(trace, str):
            trace_text = trace
        elif isinstance(trace, list):
            trace_text = '\n'.join(str(item) for item in trace)
        else:
            trace_text = str(trace)
        
        steps = self._parse_steps(trace_text)
        
        if not steps:
            return self._empty_metrics()
        
        return {
            'reasoning_depth': len(steps),
            'max_depth': max(s.depth for s in steps) if steps else 0,
            'decision_points': sum(1 for s in steps if s.type == 'decision'),
            'analysis_steps': sum(1 for s in steps if s.type == 'analysis'),
            'planning_steps': sum(1 for s in steps if s.type == 'planning'),
            'execution_steps': sum(1 for s in steps if s.type == 'execution'),
            'backtrack_rate': round(sum(1 for s in steps if s.is_backtrack) / len(steps) * 100, 1),
            'planning_score': self._calculate_planning_score([s.__dict__ for s in steps]),
            'problem_decomposition_score': self._analyze_decomposition(steps),
            'decision_confidence': self._calculate_decision_confidence(steps)
        }
    
    def _parse_steps(self, trace: str) -> List[ReasoningStep]:
        """Parse trace into reasoning steps."""
        steps = []
        lines = trace.split('\n')
        
        for line in lines:
            if not line.strip():
                continue
            
            # Determine depth by indentation
            depth = (len(line) - len(line.lstrip())) // 2
            
            # Determine step type
            step_type = self._classify_step(line.lower())
            
            # Check if it's a backtrack
            is_backtrack = any(indicator in line.lower() for indicator in self.backtrack_indicators)
            
            steps.append(ReasoningStep(
                content=line.strip(),
                depth=depth,
                type=step_type,
                is_backtrack=is_backtrack
            ))
        
        return steps
    
    def _classify_step(self, line: str) -> str:
        """Classify the type of reasoning step."""
        for step_type, indicators in self.reasoning_indicators.items():
            if any(indicator in line for indicator in indicators):
                return step_type
        return 'execution'  # Default
    
    def _calculate_planning_score(self, steps: List[Dict]) -> int:
        """Calculate planning score based on planning steps."""
        if not steps:
            # In mock mode, simulate some planning based on trace content
            return random.randint(20, 40)  # Simulate modest planning in mock mode
        
        planning_steps = [s for s in steps if s['type'] == 'planning']
        if not planning_steps:
            # If we have steps but no explicit planning, give partial credit
            return min(25, len(steps) * 5)
        
        execution_steps = [s for s in steps if s['type'] == 'execution']
        
        if not execution_steps:
            return 50  # All planning, no execution
        
        # Good planning has planning before execution
        plan_before_exec = 0
        for i, step in enumerate(steps):
            if step['type'] == 'execution' and i > 0:
                if any(s['type'] == 'planning' for s in steps[:i]):
                    plan_before_exec += 1
        
        planning_ratio = len(planning_steps) / len(steps)
        execution_success = plan_before_exec / len(execution_steps) if execution_steps else 0
        
        # Score based on balance and sequence
        score = (planning_ratio * 40 + execution_success * 60)
        return min(100, int(score))
    
    def _analyze_decomposition(self, steps: List[ReasoningStep]) -> int:
        """Analyze how well problems are decomposed (0-100)."""
        if not steps:
            return 0
        
        # Check for hierarchical structure (depth changes)
        depth_changes = 0
        for i in range(1, len(steps)):
            if steps[i].depth != steps[i-1].depth:
                depth_changes += 1
        
        # Check for systematic approach (type variety)
        type_variety = len(set(s.type for s in steps))
        
        # Score based on structure and variety
        structure_score = min(100, depth_changes * 10)
        variety_score = min(100, type_variety * 25)
        
        return int((structure_score + variety_score) / 2)
    
    def _calculate_decision_confidence(self, steps: List[ReasoningStep]) -> int:
        """Calculate confidence in decisions (0-100)."""
        decision_steps = [s for s in steps if s.type == 'decision']
        if not decision_steps:
            return 50  # No decisions made
        
        # Decisions without backtracks are more confident
        confident_decisions = [s for s in decision_steps if not s.is_backtrack]
        
        confidence = (len(confident_decisions) / len(decision_steps)) * 100
        return int(confidence)
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics when no trace available."""
        return {
            'reasoning_depth': 0,
            'max_depth': 0,
            'decision_points': 0,
            'analysis_steps': 0,
            'planning_steps': 0,
            'execution_steps': 0,
            'backtrack_rate': 0,
            'planning_score': 0,
            'problem_decomposition_score': 0,
            'decision_confidence': 50
        }
