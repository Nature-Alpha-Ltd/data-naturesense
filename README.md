# NatureSense Monthly Data

A Google Cloud Function–based pipeline to generate NatureSense monthly data from BigQuery, with local development, CI/CD, and automated deployments.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Clone & Shell Setup](#clone--shell-setup)
  - [Local Python Environment](#local-python-environment)
    - [Run Onboarding Script](#run-onboarding-script)
    - [Environment & Config](#environment--config)
- [Running Locally](#running-locally)
- [Linting & Formatting](#linting--formatting)
- [Pre‑commit Hooks](#precommit-hooks)
- [Testing & Coverage](#testing--coverage)
- [CI/CD (GitHub Actions)](#cicd-github-actions)
- [Directory Layout](#directory-layout)

---

## Features

- Loads ALD (with NatureSense metrics) and companies' assets guestimator data from BigQuery
- Aggregates ALD to company-level NatureSense metrics
- Implements Bayesian approach with country-level priors (i.e., create posteriors)
- Write results to BigQuery: 
  - `gen2_files.naturesense` (monthly data as to be distributed to clients)
  - `other_gen2_files.naturesense_enhanced` (including pre and pos enhancement columns)
- Deploys as HTTP‑triggered Cloud Function
- Local dev environment with Black, isort, pytest, pre‑commit hooks
- GitHub Actions for PR checks (lint, formatting, tests/coverage)
- Manual "Run workflow" button for on‑demand feature‑branch deploys
- Single `config.ini` for `dev`/`prod` profiles; only `ENVIRONMENT` is passed at runtime

---

## Prerequisites

- **Git** (2.20+)  
- **Python** 3.12

---

## Getting Started

### Clone & Shell Setup

```bash
git clone https://github.com/Nature-Alpha-Ltd/data-naturesense.git
cd data-naturesense
```

### Local Python Environment

#### Run Onboarding Script

```bash
chmod +x setup.sh
./setup.sh
```

NOTE: .sh scripts only run on MacOS

#### Environment & Config

We centralise non‑sensitive settings in config.ini

```ini
[dev]
PROJECT_ID = na-datalake
BQ_DATASET = gen2_files_dev
FILE_NAME = naturesense
BQ_DATASET_OTHER = other_gen2_files_dev
FILE_NAME_OTHER = naturesense_enhanced
ALD = na-datalake.production_ready_access_layer.naturesense_solved_assets
ASSET_COUNTS_GUESTIMATES = na-datalake.production_ready_access_layer.guestimator_latest
NATURESENSE_COUNTRY = na-datalake.production_ready_access_layer.naturesense_country_level
MASTER_TABLE = na-datalake.production_ready_access_layer.isin_master_table_latest
PRIMARY_SECTOR = na-datalake.production_ready_access_layer.isin_primary_gics_sector

[prod]
PROJECT_ID = na-datalake
BQ_DATASET = gen2_files
FILE_NAME = naturesense
BQ_DATASET_OTHER = other_gen2_files
FILE_NAME_OTHER = naturesense_enhanced
ALD = na-datalake.production_ready_access_layer.naturesense_solved_assets
ASSET_COUNTS_GUESTIMATES = na-datalake.production_ready_access_layer.guestimator_latest
NATURESENSE_COUNTRY = na-datalake.production_ready_access_layer.naturesense_country_level
MASTER_TABLE = na-datalake.production_ready_access_layer.isin_master_table_latest
PRIMARY_SECTOR = na-datalake.production_ready_access_layer.isin_primary_gics_sector
```

For local development, you need to add a Service Account JSON file to authenticate with Google Cloud Platform. Place the JSON file in the repository root directory. The file should be named according to your project's naming convention (e.g., `na-datalake-*.json`).

Keep a minimal local .env (not committed)

```bash
echo "ENVIRONMENT=dev" > .env
source .env
```

The code picks up ENVIRONMENT from the .env file and reads the rest from config.ini.

---

## Running Locally

With venv active and ENVIRONMENT=dev simply run the script as you would

```bash
python3 main.py
```

---

## Linting & Formatting

- Black

```bash
black .
```

- isort

```bash
isort . --profile=black
```

---

## Pre‑commit Hooks

We've configured:

- black
- isort --profile=black
- pytest

Install the Git hooks once (part of setup.sh script):

```bash
pre-commit install
```

On each git commit, Black, isort, and pytest will run automatically.

---

## Testing & Coverage

Run all test locally:

```bash
pytest --maxfail=1 --disable-warnings -q --cov=./ --cov-report=term
```

To generate an XML report:

```bash
pytest --cov=./ --cov-report=xml
```

---

## CI/CD (GitHub Actions)

We have two workflows:

1. **pr_check.yml** - on PR to dev or main, runs:
   - Black & isort (check only)
   - pytest + coverage (posts a comment under your PR)
2. **deploy-cloud-funcion.yml** - on push to dev/main or manual dispatch:
   - Authenticates to GCP via Workload Identity Federation
   - Deploys to the correct Cloud Function based on ENVIRONMENT

You can trigger a manual deployment from any branch by using the "Run workflow" button in the Actions tab, choosing the branch and environment.

---

## Directory Layout

```text
.
├── main.py                  # Cloud Function entrypoint
├── config.ini               # dev/prod profiles
├── requirements.txt         # requirements file used in CF
├── requirements-dev.txt     # requirements file used when developing
├── setup.sh                 # onboarding script
├── utils/
│   └── gcp_tools.py        # GCP utilities
├── tests/
│   └── test_ns_aggregation.py
│   └── test_gcp_tools.py
├── .pre-commit-config.yaml  # pre commit file with rules
├── .github/
│   └── workflows/
│       ├── pr_checks.yml
│       └── deploy-cloud-function.yml
└── .gcloudignore            # CF ignore file
```