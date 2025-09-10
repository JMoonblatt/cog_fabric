#!/usr/bin/env bash
set -e

# Make sure we're in the project root (this folder should contain 'cog_fabric/' and 'scripts/')
if [ ! -d "cog_fabric" ] || [ ! -d "scripts" ]; then
  echo "Run this from the project root containing 'cog_fabric/' and 'scripts/'"
  exit 1
fi

# Activate venv (relative path from here)
if [ -f ".venv/Scripts/activate" ]; then
  source .venv/Scripts/activate
fi

# Install/update requirements
pip install -r requirements.txt

# Run the demo (module mode ensures imports work)
python -m scripts.run_dot_tracking --config configs/default.yaml

