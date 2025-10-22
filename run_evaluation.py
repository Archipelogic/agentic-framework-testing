#!/usr/bin/env python3
"""
Unified Benchmark Runner for Agentic Framework Testing Harness
Supports both mock (no API) and live (with APIs) modes.
Always uses real test data and ground truth for evaluation.

Usage:
    python benchmark.py --mock          # Run with mock adapters (no APIs needed)
    python benchmark.py --live          # Run with real API calls
    python benchmark.py --mock --quick  # Quick test with top frameworks
    python benchmark.py --help          # Show all options
"""

import sys
import os
import json
import random
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import Config
from src.core.types import UseCaseType, FrameworkType
from src.reporting.unified_reporter import UnifiedReporter
from src.metrics.enhanced_evaluator import EnhancedEvaluator
from src.metrics.evaluator import Evaluator
from src.utils.data_generator import ensure_all_data_exists


class UnifiedBenchmarkRunner:
    """Unified benchmark runner that works with both mock and live adapters."""
    
    def __init__(self, mode: str = 'mock', quick: bool = False, verbose: bool = True):
        """
        Initialize benchmark runner.
        
        Args:
            mode: 'mock' for no APIs, 'live' for real API calls
            quick: If True, only test top frameworks on subset of use cases
            verbose: If True, print detailed progress
        """
        self.mode = mode
        self.quick = quick
        self.verbose = verbose
        self.config = Config() if mode == 'live' else None
        
        # Define use cases
        self.use_cases = self._get_use_cases()
        
        # Initialize evaluators
        self.evaluator = Evaluator()
        self.enhanced_evaluator = EnhancedEvaluator()
        
        # Load test data for all use cases
        self.test_data = {}
        self.ground_truth = {}
        self._load_test_data()
        
        # Get framework adapters
        self.adapters = self._get_adapters()
    
    def _get_use_cases(self) -> List[UseCaseType]:
        """Get use cases to test based on mode."""
        if self.quick:
            # Quick mode: only 2 use cases
            return [
                UseCaseType.MOVIE_RECOMMENDATION,
                UseCaseType.GITHUB_TRIAGE
            ]
        else:
            # Full mode: all use cases
            return [
                UseCaseType.MOVIE_RECOMMENDATION,
                UseCaseType.GITHUB_TRIAGE,
                UseCaseType.RECIPE_GENERATION,
                UseCaseType.RESEARCH_SUMMARY,
                UseCaseType.EMAIL_AUTOMATION
            ]
    
    def _load_test_data(self):
        """Load actual test cases and ground truth from JSON files."""
        if self.verbose:
            print("\n📚 Loading test data and ground truth...")
        
        for use_case in self.use_cases:
            use_case_name = use_case.value
            
            # Load test cases
            test_file = Path(f'data/{use_case_name}/test_cases.json')
            if test_file.exists():
                with open(test_file) as f:
                    test_cases = json.load(f)
                    self.test_data[use_case] = test_cases
                    if self.verbose:
                        print(f"  ✅ Loaded {len(test_cases)} test cases for {use_case_name}")
            else:
                print(f"  ❌ Missing test cases for {use_case_name}")
                self.test_data[use_case] = []
            
            # Load ground truth
            truth_file = Path(f'data/{use_case_name}/ground_truth.json')
            if truth_file.exists():
                with open(truth_file) as f:
                    ground_truth = json.load(f)
                    self.ground_truth[use_case] = ground_truth
                    if self.verbose:
                        print(f"  ✅ Loaded {len(ground_truth)} ground truth entries for {use_case_name}")
            else:
                print(f"  ❌ Missing ground truth for {use_case_name}")
                self.ground_truth[use_case] = []
    
    def _get_adapters(self) -> Dict[str, Any]:
        """Get framework adapters based on mode."""
        if self.mode == 'mock':
            from src.adapters.mock_adapters import create_all_mock_adapters
            adapters = create_all_mock_adapters()
            
            if self.quick:
                # Quick mode: only top 3 frameworks
                top_frameworks = ['langgraph', 'crewai', 'autogen']
                adapters = {k: v for k, v in adapters.items() if k in top_frameworks}
            
            if self.verbose:
                print(f"\n✅ Created {len(adapters)} mock framework adapters")
            return adapters
        else:
            # Live mode: use real framework adapters
            from src.adapters.framework_adapters import create_adapter
            adapters = {}
            
            # Get all available frameworks
            frameworks = list(FrameworkType)
            if self.quick:
                frameworks = [FrameworkType.LANGGRAPH, FrameworkType.CREWAI, FrameworkType.AUTOGEN]
            
            for framework in frameworks:
                try:
                    adapter = create_adapter(framework, self.config)
                    adapters[framework.value] = adapter
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠️ Could not load {framework.value}: {e}")
            
            if self.verbose:
                print(f"\n✅ Loaded {len(adapters)} live framework adapters")
            return adapters
    
    def run_benchmark(self, 
                      num_samples_per_test: int = 20,
                      parallel: bool = False,
                      save_outputs: bool = True) -> Dict[str, Any]:
        """
        Run the benchmark with real test data.
        
        Args:
            num_samples_per_test: Number of test cases to run per use case (randomly sampled)
            parallel: If True, run tests in parallel (live mode only)
            save_outputs: If True, save all outputs to files
            
        Returns:
            Complete results dictionary
        """
        if self.verbose:
            print("\n" + "="*60)
            print("🏃 Running benchmark tests...")
            print(f"  Mode: {self.mode.upper()}")
            print(f"  Frameworks: {len(self.adapters)}")
            print(f"  Use cases: {len(self.use_cases)}")
            if num_samples_per_test is not None:
                print(f"  Samples per test: {num_samples_per_test}")
            else:
                print(f"  Using ALL test data (200 per use case)")
            print("="*60 + "\n")
        
        # Try to use progress bar if available
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
        
        results = {}
        enhanced_metrics = {}
        
        # Iterate through frameworks
        iterator = tqdm(self.adapters.items(), desc="Testing frameworks") if use_tqdm else self.adapters.items()
        
        for framework_name, adapter in iterator:
            framework_results = {}
            framework_enhanced = {}
            passed = 0
            failed = 0
            total_latency = 0
            total_accuracy = 0
            
            for use_case in self.use_cases:
                if adapter.supports_use_case(use_case):
                    # Get test cases for this use case
                    test_cases = self.test_data.get(use_case, [])
                    ground_truths = self.ground_truth.get(use_case, [])
                    
                    if not test_cases or not ground_truths:
                        if self.verbose:
                            print(f"  ⚠️ No test data for {use_case.value}, skipping...")
                        continue
                    
                    # Use all test cases unless num_samples_per_test is specified
                    if num_samples_per_test is not None:
                        # Sample test cases if requested
                        if len(test_cases) > num_samples_per_test:
                            # Randomly sample test cases
                            indices = random.sample(range(len(test_cases)), num_samples_per_test)
                            sampled_tests = [test_cases[i] for i in indices]
                            sampled_truths = [ground_truths[i] for i in indices]
                        else:
                            sampled_tests = test_cases[:num_samples_per_test]
                            sampled_truths = ground_truths[:num_samples_per_test]
                    else:
                        # Use ALL test data - no sampling
                        sampled_tests = test_cases
                        sampled_truths = ground_truths
                    
                    # Run tests on sampled data
                    use_case_results = []
                    use_case_enhanced = []
                    
                    for test_input, ground_truth in zip(sampled_tests, sampled_truths):
                        # Run the framework on real test input
                        try:
                            result = adapter.run(use_case, test_input)
                            
                            # Evaluate against real ground truth
                            accuracy = self.evaluator.evaluate(
                                use_case=use_case.value,
                                framework=framework_name,
                                test_id=test_input.get('test_id', 'unknown'),
                                framework_result=result,
                                ground_truth=ground_truth
                            )
                            
                            # Enhanced metrics if succeeded
                            if result.succeeded:
                                enhanced = self.enhanced_evaluator.evaluate_comprehensive(
                                    output=result.result,
                                    ground_truth=ground_truth,
                                    context=test_input,
                                    trace=f"Test {test_input.get('test_id', 'unknown')} for {framework_name}"
                                )
                                use_case_enhanced.append(enhanced)
                            else:
                                enhanced = None
                            
                            # Store detailed result
                            use_case_results.append({
                                'test_id': test_input.get('test_id', 'unknown'),
                                'success': result.succeeded,
                                'accuracy': accuracy.overall_accuracy if accuracy else 0,
                                'latency': result.latency_seconds,
                                'tokens': result.tokens.total_tokens if result.tokens else 0,
                                'cost': result.cost.total_usd if result.cost else 0,
                                'output': result.result,
                                'trajectory': [action.to_dict() for action in result.trajectory] if result.trajectory else [],
                                'errors': result.errors,
                                'metadata': result.metadata,
                                'enhanced_metrics': enhanced
                            })
                            
                            if result.succeeded:
                                passed += 1
                                total_latency += result.latency_seconds
                                total_accuracy += accuracy.overall_accuracy if accuracy else 0
                            else:
                                failed += 1
                                
                        except Exception as e:
                            # Handle framework errors gracefully
                            use_case_results.append({
                                'test_id': test_input.get('test_id', 'unknown'),
                                'success': False,
                                'accuracy': 0,
                                'error': str(e)
                            })
                            failed += 1
                    
                    # Aggregate use case results
                    if use_case_results:
                        avg_accuracy = sum(r.get('accuracy', 0) for r in use_case_results) / len(use_case_results)
                        avg_latency = sum(r.get('latency', 0) for r in use_case_results) / len(use_case_results)
                        
                        framework_results[use_case.value] = {
                            'tests_run': len(use_case_results),
                            'tests_passed': sum(1 for r in use_case_results if r.get('success', False)),
                            'average_accuracy': avg_accuracy,
                            'average_latency': avg_latency,
                            'detailed_results': use_case_results
                        }
                        
                        # Aggregate enhanced metrics
                        if use_case_enhanced:
                            framework_enhanced[use_case.value] = self._aggregate_enhanced_metrics(use_case_enhanced)
                
                else:
                    framework_results[use_case.value] = {
                        'supported': False
                    }
                    failed += 1
            
            # Print summary for this framework
            if self.verbose and not use_tqdm:
                avg_latency = total_latency / passed if passed > 0 else 0
                avg_accuracy = total_accuracy / passed if passed > 0 else 0
                status_icon = "✅" if passed > failed else "⚠️" if passed > 0 else "❌"
                print(f"{status_icon} {framework_name}: {passed}/{passed+failed} tests passed" +
                      f" (avg accuracy: {avg_accuracy:.1%}, avg latency: {avg_latency:.2f}s)")
            
            results[framework_name] = framework_results
            if framework_enhanced:
                enhanced_metrics[framework_name] = framework_enhanced
        
        # Calculate capability scores based on REAL results
        capability_scores = self._calculate_capability_scores(results, enhanced_metrics)
        
        # Define capability weights
        capability_weights = {
            'multi_agent': 0.15,
            'tool_usage': 0.12,
            'error_handling': 0.10,
            'context_retention': 0.10,
            'adaptability': 0.12,
            'scalability': 0.08,
            'observability': 0.08,
            'rag_capability': 0.10,
            'overall_score': 0.15
        }
        
        # Package complete results
        complete_results = {
            'results': results,
            'capability_scores': capability_scores,
            'capability_weights': capability_weights,
            'enhanced_metrics': enhanced_metrics,
            'mode': self.mode,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'config': {
                'num_samples_per_test': num_samples_per_test if num_samples_per_test is not None else 'ALL',
                'frameworks_tested': list(self.adapters.keys()),
                'use_cases_tested': [uc.value for uc in self.use_cases]
            }
        }
        
        return complete_results
    
    def _aggregate_enhanced_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate multiple enhanced metrics into averages."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        # Average all numeric values
        for key in metrics_list[0].keys():
            if isinstance(metrics_list[0][key], dict):
                # Recurse for nested dicts
                nested_values = [m[key] for m in metrics_list if key in m]
                aggregated[key] = self._aggregate_enhanced_metrics(nested_values)
            elif isinstance(metrics_list[0][key], (int, float)):
                # Average numeric values
                values = [m[key] for m in metrics_list if key in m]
                aggregated[key] = sum(values) / len(values) if values else 0
            else:
                # Keep first value for non-numeric
                aggregated[key] = metrics_list[0][key]
        
        return aggregated
    
    def _calculate_capability_scores(self, results: Dict, enhanced_metrics: Dict) -> Dict:
        """Calculate capability scores based on real accuracy and metrics."""
        from src.metrics.capability_scorer import calculate_real_capability_scores
        return calculate_real_capability_scores(results, enhanced_metrics)
    
    def save_results(self, results: Dict, auto_open: bool = True) -> str:
        """Save results and generate reports with organized directory structure."""
        from src.reporting.results_saver import save_benchmark_results
        return save_benchmark_results(results, self.mode, auto_open, self.verbose)


# Import helper functions
from src.cli.helpers import preflight_check, main

if __name__ == "__main__":
    sys.exit(main())
