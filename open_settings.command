#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d ".venv" ]]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

if ! python -c "import tomli_w" >/dev/null 2>&1; then
  echo "Installing dependencies..."
  python -m pip install -r requirements.txt
fi

if ! python -c "import tkinter" >/dev/null 2>&1; then
  echo "tkinter is not available in this Python build."
  echo "Use a Python installation that includes tkinter, then run this script again."
  exit 1
fi

exec python main.py --config config.toml --settings
