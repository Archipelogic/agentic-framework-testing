"""
CLI helper functions for the unified benchmark runner.
"""

import sys
import os
import socket
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def preflight_check():
    """Run pre-flight checks before benchmark."""
    import platform
    print("\n🔍 Running pre-flight checks...")
    print("-" * 40)
    
    checks_passed = True
    
    # Check Python version
    py_version = platform.python_version()
    if tuple(map(int, py_version.split('.')[:2])) >= (3, 8):
        print(f"✅ Python {py_version} detected")
    else:
        print(f"❌ Python {py_version} (3.8+ required)")
        checks_passed = False
    
    # Check port availability (for report server)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 8080))
    if result != 0:
        print("✅ Port 8080 available")
    else:
        print("⚠️ Port 8080 in use (will be cleared)")
        # Kill any existing process on port 8080
        try:
            result = subprocess.run(['lsof', '-ti:8080'], capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                subprocess.run(['kill', '-9', result.stdout.strip()], check=False)
        except:
            pass
    sock.close()
    
    # Check for data files
    data_exists = all(
        Path(f'data/{uc}/test_cases.json').exists()
        for uc in ['movie_recommendation', 'github_triage', 'recipe_generation', 'research_summary', 'email_automation']
    )
    
    if data_exists:
        print("✅ Test data files found")
    else:
        print("⚠️ Some test data missing (will be generated)")
    
    # Check for API keys (for live mode)
    has_keys = bool(os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY'))
    if has_keys:
        print("✅ API keys found (live mode available)")
    else:
        print("⚠️ No API keys found (mock mode only)")
    
    print("-" * 40)
    if checks_passed:
        print("✅ All checks passed!\n")
    return checks_passed


def main():
    """Main entry point."""
    from src.utils.data_generator import ensure_all_data_exists
    
    # Lazy import to avoid circular dependency
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from run_evaluation import UnifiedBenchmarkRunner
    
    parser = argparse.ArgumentParser(
        description='Unified Agentic Framework Testing Harness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py --mock              # Run with mock adapters (no APIs)
  python run_evaluation.py --live              # Run with real API calls
  python run_evaluation.py --mock --quick      # Quick test with top frameworks
  python run_evaluation.py --mock --samples 5  # Run 5 test samples per use case
  python run_evaluation.py --live --parallel   # Run tests in parallel (live mode)
  
Notes:
  - Mock mode requires no API keys and uses simulated responses
  - Live mode requires API keys configured in .env file
  - Test data is automatically generated if missing
  - Results are saved in benchmark_results/{mode}_run_{timestamp}/
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--mock', action='store_true', help='Run with mock adapters (no API calls)')
    mode_group.add_argument('--live', action='store_true', help='Run with real API calls')
    
    # Options
    parser.add_argument('--quick', action='store_true', help='Quick mode: test top 3 frameworks on 2 use cases')
    parser.add_argument('--samples', type=int, default=20, help='Number of test samples per use case (default: 20 for reasonable speed)')
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel (live mode only)')
    parser.add_argument('--no-open', action='store_true', help="Don't auto-open the HTML report")
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # Determine mode
    mode = 'mock' if args.mock else 'live'
    
    # Print header
    print("=" * 60)
    print("🚀 Agentic Framework Testing Harness")
    print("=" * 60)
    print()
    
    # Run preflight checks
    if not preflight_check():
        if mode == 'live':
            print("\n❌ Pre-flight checks failed. Cannot run in live mode.")
            sys.exit(1)
    
    # Ensure all test data exists
    success, failed = ensure_all_data_exists(verbose=not args.quiet)
    if not success:
        print(f"\n❌ Cannot proceed without test data for: {', '.join(failed)}")
        print("   Please check the data generation scripts.")
        sys.exit(1)
    
    # Create and run benchmark
    try:
        runner = UnifiedBenchmarkRunner(
            mode=mode,
            quick=args.quick,
            verbose=not args.quiet
        )
        
        # Run the benchmark with real test data
        results = runner.run_benchmark(
            num_samples_per_test=args.samples,
            parallel=args.parallel and mode == 'live'
        )
        
        # Save results and generate reports
        output_dir = runner.save_results(results, auto_open=not args.no_open)
        
        print("\n" + "="*60)
        print("✅ Benchmark completed successfully!")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Benchmark interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error running benchmark: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1
