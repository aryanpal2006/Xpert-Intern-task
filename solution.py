#!/usr/bin/env python3
"""Thin entrypoint for the Cold Start solution.

The real pipeline lives in the `coldstart` package (see coldstart/main.py). This wrapper lets you
run it as a single script as well:

    python3 solution.py <input_dir> <output.json>

`run.sh` / `run.bat` call `python3 -m coldstart.main` directly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coldstart.main import main  # noqa: E402

if __name__ == "__main__":
    main()
