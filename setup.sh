#!/usr/bin/env bash
set -euo pipefail

# 1. Create & activate venv
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# 2. Upgrade pip & install deps
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Install pre-commit hooks
pre-commit install

# 4. Copy .env template if it doesn't exist
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "⚠️  .env created, please edit it with your own credentials."
else
  echo ".env already exists, skipping"
fi

# 5. Final sanity checks
echo "Running pytest..."
.venv/bin/pytest --maxfail=1 --disable-warnings -q

echo "Running pre-commit hooks on all files..."
pre-commit run --all-files

# 6. Activate the venv
echo "Activating virtual environment..."
source .venv/bin/activate

echo "✅ Setup complete! Activate your venv with:  source .venv/bin/activate"