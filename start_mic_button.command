#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d ".venv" ]]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

if ! python -c "import faster_whisper, numpy, pynput, pyperclip, sounddevice, tomli_w" >/dev/null 2>&1; then
  echo "Installing dependencies..."
  python -m pip install -r requirements.txt
fi

exec python main.py --config config.toml --mic-button
