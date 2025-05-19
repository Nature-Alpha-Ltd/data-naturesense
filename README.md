# Nature RevX Metrics

A Google Cloud Function–based pipeline to calculate Nature RevX metrics from BigQuery data, with local development, CI/CD, and automated deployments.

---

## Table of Contents

- [Nature RevX Metrics](#nature-revx-metrics)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Getting Started](#getting-started)
    - [Clone \& Shell Setup](#clone--shell-setup)
    - [Local Python Environment](#local-python-environment)
      - [Run Onboarding Script](#run-onboarding-script)
      - [Environment \& Config](#environment--config)
  - [Running Locally](#running-locally)
  - [Linting \& Formatting](#linting--formatting)
  - [Pre‑commit Hooks](#precommit-hooks)
  - [Testing \& Coverage](#testing--coverage)
  - [CI/CD (GitHub Actions)](#cicd-github-actions)
  - [Directory Layout](#directory-layout)

---

## Features

- Extracts and validates mapping tables in BigQuery  
- Merges revenue splits with impact & dependency scores  
- Calculates final Nature RevX metrics  
- Deploys as HTTP‑triggered Cloud Function  
- Local dev environment with Black, isort, pytest, pre‑commit hooks  
- GitHub Actions for PR checks (lint, formatting, tests/coverage)  
- Manual “Run workflow” button for on‑demand feature‑branch deploys  
- Single `config.ini` for `dev`/`prod` profiles; only `ENVIRONMENT` is passed at runtime

---

## Prerequisites

- **Git** (2.20+)  
- **Python** 3.12

---

## Getting Started

### Clone & Shell Setup

```bash
git clone https://github.com/Nature-Alpha-Ltd/data-nrevx.git
cd data-nrevx
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
BQ_DATASET = monthly_files_2_dev
ENTITIES_TABLE_ID = na-datalake.mappings.isin_master_table_latest
ISIN_MAPPING = na-datalake.processed_data.factsetid_to_isin_master_latest
MM_TABLE = na-datalake.raw_internal_data_ingestion.mm_dependencies_impacts_v2
GICS_MAPPING = na-datalake.mappings.rbics_to_gics_NEW_latest
RBICS_L6 = na-datalake.production_ready_access_layer.rbics_l6_revenue_scaled
MISSING_L6_MAP = na-datalake.processed_data.check_for_missing_l6_mapping

[prod]
PROJECT_ID = na-datalake
BQ_DATASET = monthly_files_2
ENTITIES_TABLE_ID = na-datalake.mappings.isin_master_table_latest
ISIN_MAPPING = na-datalake.processed_data.factsetid_to_isin_master_latest
MM_TABLE = na-datalake.raw_internal_data_ingestion.mm_dependencies_impacts_v2
GICS_MAPPING = na-datalake.mappings.rbics_to_gics_NEW_latest
RBICS_L6 = na-datalake.production_ready_access_layer.rbics_l6_revenue_scaled
MISSING_L6_MAP = na-datalake.processed_data.check_for_missing_l6_mapping
```

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

We’ve configured:

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
2. **deploy-cloud-funcion.yml** - on push to dev/main or manual dispatch: WORK IN PROGRESS
   - Authenticates to GCP via Workload Identity Federation
   - Deploys to the correct Cloud Function based on ENVIRONMENT

You can trigger a manual deployment from any branch by using the “Run workflow” button in the Actions tab, choosing the branch and environment.

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
│   ├── gcp_tools.py
│   └── nrevx_engine.py
├── tests/
│   ├── test_gcp_tools.py
│   └── test_nrevx_engine.py
├── .pre-commit-config.yaml  # pre commit file with rules
├── .github/
│   └── workflows/
│       ├── pr_checks.yml
│       └── deploy-cloud-function.yml
└── .gcloudignore            # CF ignore file
```
