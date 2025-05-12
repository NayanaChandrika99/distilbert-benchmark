"""
Main entry point for the DistilBERT benchmarking suite.

This script provides a convenient CLI interface to run benchmarks.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.runner import main as runner_main


def main():
    """Entry point for the benchmarking suite."""
    runner_main()


if __name__ == "__main__":
    main()
