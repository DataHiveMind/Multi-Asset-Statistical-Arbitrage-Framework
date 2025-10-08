#!/usr/bin/env python3
"""
Test runner script for the Multi-Asset Statistical Arbitrage Framework.
This script provides a convenient way to run tests with different configurations.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle the output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run tests for the Multi-Asset Statistical Arbitrage Framework")
    parser.add_argument("--framework", choices=["unittest", "pytest"], default="unittest",
                       help="Testing framework to use (default: unittest)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Run tests with verbose output")
    parser.add_argument("--coverage", action="store_true",
                       help="Run tests with coverage reporting (pytest only)")
    parser.add_argument("--specific", "-s", 
                       help="Run specific test file or test method")
    
    args = parser.parse_args()
    
    # Get the project root directory
    project_root = Path(__file__).parent
    venv_python = project_root / ".venv" / "bin" / "python"
    
    # Check if virtual environment exists
    if not venv_python.exists():
        print("❌ Virtual environment not found. Please run 'make install' first.")
        sys.exit(1)
    
    success = True
    
    if args.framework == "unittest":
        if args.specific:
            cmd = [str(venv_python), "-m", "unittest", args.specific]
        else:
            cmd = [str(venv_python), "-m", "unittest", "discover", "tests/"]
        
        if args.verbose:
            cmd.append("-v")
        
        success = run_command(cmd, "Unit Tests")
    
    elif args.framework == "pytest":
        cmd = [str(venv_python), "-m", "pytest", "tests/"]
        
        if args.verbose:
            cmd.append("-v")
        
        if args.coverage:
            cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
        
        if args.specific:
            cmd.append(args.specific)
        
        success = run_command(cmd, "Pytest Tests")
    
    if success:
        print(f"\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()