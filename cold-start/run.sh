#!/usr/bin/env bash
# The one command we run: ./run.sh <input_dir> <output.json>
# Runs cleanly on a standard Linux machine in under 30 minutes.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: ./run.sh <input_dir> <output.json>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

python3 -m coldstart.main "$1" "$2"
