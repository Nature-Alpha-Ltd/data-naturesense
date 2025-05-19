import logging
import os
import subprocess
from datetime import datetime, timedelta

import pandas as pd
import pandas_gbq
from dotenv import load_dotenv
from google.auth import default as google_auth_default
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery
from google.oauth2 import service_account

load_dotenv()  # Load variables from .env

# Set up logging
logging.basicConfig(level=logging.INFO)

# Module-level cache
_bq_client = None


def running_in_gcp() -> bool:
    """
    Detects if the code is running in a GCP environment (e.g., Cloud Functions or Cloud Run).
    """
    return os.getenv("RUNNING_IN_GCP") is not None


def get_bq_client() -> bigquery.Client:
    """
    Returns a BigQuery client. Enforces SA credentials locally.
    Uses ADC in GCP.
    """
    global _bq_client
    if _bq_client is not None:
        return _bq_client

    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not running_in_gcp():
        # Local environment — require SA
        if not sa_path or not os.path.exists(sa_path):
            raise RuntimeError(
                "Service account credentials required for local development. "
                "Set GOOGLE_APPLICATION_CREDENTIALS to a valid key path in .env file."
            )
        logging.info("Using local service account: %s", sa_path)
        credentials = service_account.Credentials.from_service_account_file(sa_path)
        _bq_client = bigquery.Client(
            credentials=credentials, project=credentials.project_id
        )
    else:
        # GCP environment — use ADC
        logging.info("Using ADC in GCP environment")
        try:
            creds, _ = google_auth_default()
            _bq_client = bigquery.Client(credentials=creds)
        except DefaultCredentialsError as e:
            raise RuntimeError(
                "Failed to load ADC credentials in GCP environment."
            ) from e

    return _bq_client


def run_query(query: str) -> pd.DataFrame:
    """
    Executes a SQL query using BigQuery client and returns the result as a DataFrame.
    """
    client = get_bq_client()
    return client.query(query).result().to_dataframe()


def write_df_to_bq(df: pd.DataFrame, dest_table_id: str) -> None:
    """
    Writes a DataFrame to a BigQuery table using pandas-gbq.
    """
    client = get_bq_client()
    project_id = client.project  # Get project ID from the client
    logging.info("Writing DataFrame to BigQuery table: %s", dest_table_id)
    pandas_gbq.to_gbq(
        df,
        dest_table_id,
        project_id=project_id,
        if_exists="replace",  # Equivalent to WRITE_TRUNCATE
    )


def last_day_of_month() -> str:
    """
    Determines the last day of the current month and returns it as a formatted string.

    Returns:
        str: The last day of the current month in 'YYYYMMDD' format.
    """
    today = datetime.now()
    next_month = today.replace(day=28) + timedelta(days=4)
    last_day = next_month - timedelta(days=next_month.day)
    return last_day.strftime("%Y%m%d")


def get_git_branch():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        suffix = result.stdout.strip() + "_" + datetime.now().strftime("%Y%m%d")
        return suffix
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning("Could not determine Git branch. Using default suffix.")
        return "develop" + "_" + datetime.now().strftime("%Y%m%d")


def save_results(
    nature_revx_df: pd.DataFrame, table_name: str, dataset: str, env: str, project: str
) -> None:
    """Write the results DataFrame to BigQuery."""
    if env != "dev":
        datestr = last_day_of_month()
    else:
        datestr = get_git_branch()
    output_table_id = f"{project}.{dataset}.{table_name}_{datestr}"
    write_df_to_bq(nature_revx_df, output_table_id)
    logging.info(f"{table_name} metrics saved to %s", output_table_id)
