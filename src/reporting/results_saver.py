"""
Results saving functionality for benchmark runs.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from src.reporting.unified_reporter import UnifiedReporter


def save_benchmark_results(results: Dict, mode: str, auto_open: bool = True, verbose: bool = True) -> str:
    """
    Save benchmark results with organized directory structure.
    
    Args:
        results: Complete results dictionary
        mode: 'mock' or 'live'
        auto_open: Whether to auto-open the HTML report
        verbose: Whether to print detailed output
        
    Returns:
        Path to the run directory
    """
    # Create a subdirectory for this run
    timestamp = results['timestamp']
    base_dir = Path('benchmark_results')
    base_dir.mkdir(exist_ok=True)
    
    # Create run-specific directory
    run_dir = base_dir / f'{mode}_run_{timestamp}'
    run_dir.mkdir(exist_ok=True)
    
    # Create subdirectory for framework outputs
    outputs_dir = run_dir / 'framework_outputs'
    outputs_dir.mkdir(exist_ok=True)
    
    if verbose:
        print(f"\n📁 Created run directory: {run_dir}")
    
    # Save main results file
    results_file = run_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"✅ Results saved to {results_file.relative_to(base_dir.parent)}")
    
    # Save individual framework outputs with ALL data
    for framework_name, framework_data in results['results'].items():
        framework_file = outputs_dir / f'{framework_name}.json'
        
        # Create comprehensive output for this framework
        framework_output = {
            'framework': framework_name,
            'timestamp': timestamp,
            'mode': mode,
            'use_cases': framework_data
        }
        
        with open(framework_file, 'w') as f:
            json.dump(framework_output, f, indent=2)
        
        if verbose:
            print(f"  📄 {framework_name} output → {framework_file.relative_to(run_dir)}")
    
    # Save summary file
    summary = generate_summary(results)
    summary_file = run_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"📊 Summary saved to {summary_file.relative_to(base_dir.parent)}")
    
    # Generate HTML report (don't auto-open yet, we'll do it after rename)
    reporter = UnifiedReporter(output_dir=str(run_dir))
    report_path = reporter.generate_enhanced_report(results_file, auto_open=False)
    
    # Rename report to simple name
    final_report_path = run_dir / 'report.html'
    if Path(report_path).exists() and Path(report_path) != final_report_path:
        shutil.move(str(report_path), str(final_report_path))
        report_path = final_report_path
    
    # Now open the report with the correct path
    if auto_open:
        reporter._open_report(Path(report_path))
    
    if verbose:
        print(f"📈 HTML report saved to {report_path.relative_to(base_dir.parent) if isinstance(report_path, Path) else report_path}")
        
        print(f"\n✨ All outputs saved to: {run_dir.relative_to(base_dir.parent)}")
        print(f"   • results.json - Complete benchmark results")
        print(f"   • summary.json - Quick overview")
        print(f"   • report.html - Interactive HTML report")
        print(f"   • framework_outputs/ - Individual framework outputs")
    
    return str(run_dir)


def generate_summary(results: Dict) -> Dict:
    """Generate a summary of the benchmark results."""
    summary = {
        'run_id': f'{results["mode"]}_{results["timestamp"]}',
        'mode': results['mode'],
        'timestamp': results['timestamp'],
        'config': results.get('config', {}),
        'results_summary': {}
    }
    
    for framework_name, framework_data in results['results'].items():
        total_tests = 0
        total_passed = 0
        total_accuracy = 0
        
        for use_case_data in framework_data.values():
            if 'tests_run' in use_case_data:
                total_tests += use_case_data['tests_run']
                total_passed += use_case_data['tests_passed']
                total_accuracy += use_case_data.get('average_accuracy', 0) * use_case_data['tests_run']
        
        summary['results_summary'][framework_name] = {
            'tests_run': total_tests,
            'tests_passed': total_passed,
            'tests_failed': total_tests - total_passed,
            'success_rate': f"{(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "0%",
            'average_accuracy': f"{(total_accuracy/total_tests):.1%}" if total_tests > 0 else "0%"
        }
    
    return summary
