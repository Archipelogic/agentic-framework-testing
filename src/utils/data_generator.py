#!/usr/bin/env python3
"""
Data generation utilities for the testing harness.
Automatically generates missing test data for use cases.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple


def check_data_exists(use_case: str) -> bool:
    """Check if data exists for a use case."""
    data_dir = Path(f"data/{use_case}")
    test_file = data_dir / "test_cases.json"
    ground_truth_file = data_dir / "ground_truth.json"
    
    return test_file.exists() and ground_truth_file.exists()


def get_generator_script(use_case: str) -> str:
    """Get the generator script path for a use case."""
    script_map = {
        "research_summary": "generate_research_summary_data.py",
        "email_automation": "generate_email_automation_data.py", 
        "github_triage": "generate_github_triage_data.py",
        "movie_recommendation": "generate_movie_recommendation_data.py",
        "recipe_generation": "generate_recipe_generation_data.py"
    }
    
    script_name = script_map.get(use_case)
    if not script_name:
        return None
        
    script_path = Path("scripts") / script_name
    return str(script_path) if script_path.exists() else None


def generate_data_for_use_case(use_case: str, verbose: bool = True) -> bool:
    """Generate data for a specific use case if missing."""
    if check_data_exists(use_case):
        if verbose:
            print(f"✅ Data already exists for {use_case}")
        return True
    
    script_path = get_generator_script(use_case)
    if not script_path:
        print(f"⚠️  No generator script found for {use_case}")
        return False
    
    print(f"🔧 Generating data for {use_case}...")
    try:
        # Run the generator script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ Successfully generated data for {use_case}")
            return True
        else:
            print(f"❌ Failed to generate data for {use_case}")
            if verbose:
                print(f"   Error: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Data generation timed out for {use_case}")
        return False
    except Exception as e:
        print(f"❌ Error generating data for {use_case}: {e}")
        return False


def ensure_all_data_exists(verbose: bool = True) -> Tuple[bool, List[str]]:
    """
    Ensure all use case data exists, generating if necessary.
    
    Returns:
        Tuple of (success, list_of_failed_use_cases)
    """
    use_cases = [
        "research_summary",
        "email_automation",
        "github_triage", 
        "movie_recommendation",
        "recipe_generation"
    ]
    
    print("\n" + "="*60)
    print("📊 Checking test data availability...")
    print("="*60)
    
    failed = []
    for use_case in use_cases:
        if not check_data_exists(use_case):
            print(f"⚠️  Missing data for {use_case}")
            # Try to generate it
            if not generate_data_for_use_case(use_case, verbose):
                failed.append(use_case)
        else:
            if verbose:
                print(f"✅ Data exists for {use_case}")
    
    if failed:
        print(f"\n❌ Failed to generate data for: {', '.join(failed)}")
        print("   Please check the generator scripts and try again.")
        return False, failed
    else:
        print("\n✅ All test data is ready!")
        return True, []


def cleanup_old_data(use_case: str = None):
    """Clean up old data files (hf versions)."""
    if use_case:
        use_cases = [use_case]
    else:
        use_cases = [
            "research_summary",
            "email_automation", 
            "github_triage",
            "movie_recommendation",
            "recipe_generation"
        ]
    
    for uc in use_cases:
        data_dir = Path(f"data/{uc}")
        if data_dir.exists():
            # Remove old hf files
            for pattern in ["*_hf.json", "*.json.bak"]:
                for file in data_dir.glob(pattern):
                    file.unlink()
                    print(f"🗑️  Cleaned up {file}")


if __name__ == "__main__":
    # Test the data generator
    success, failed = ensure_all_data_exists(verbose=True)
    if not success:
        sys.exit(1)
