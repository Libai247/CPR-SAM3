"""Benchmark evaluation entry point."""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    tools = Path(__file__).resolve().parent / "tools"
    sys.path.insert(0, str(tools))
    runpy.run_path(str(tools / "eval.py"), run_name="__main__")
