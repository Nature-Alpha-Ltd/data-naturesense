import logging

import numpy as np
import pandas as pd

from utils.gcp_tools import run_query

# Configure logging
logging.basicConfig(level=logging.INFO)


def postprocess_dataframe(
    df: pd.DataFrame, date_table: str, data_dictionary_table: str
) -> pd.DataFrame:
    """
    Post-processes the input DataFrame by:
    - Rounding all numeric values to 3 decimal places.
    - Renaming specific columns for clarity.
    - Reordering columns to a preferred order.
    - Adding reference_date column

    Parameters:
    - df (pd.DataFrame): Input DataFrame to process.

    Returns:
    - pd.DataFrame: Processed DataFrame.
    """

    # Rename columns
    rename_map = {
        "agg_cultural_probability": "agg_cultural",
        "agg_supporting_probability": "agg_supporting",
        "agg_regulating_probability": "agg_regulating",
        "agg_provisioning_probability": "agg_provisioning",
    }

    # Step 1: Drop any existing target columns that would conflict with renaming
    df = df.drop(columns=[v for v in rename_map.values() if v in df.columns])

    # Step 2: Rename the correct probability columns
    df = df.rename(columns=rename_map)

    # Drop duplicate column names from numeric selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = list(pd.Index(numeric_cols).drop_duplicates())

    # Fix for duplicated columns in the main DataFrame
    df = df.loc[:, ~df.columns.duplicated()]

    # Round safely
    df.loc[:, numeric_cols] = df.loc[:, numeric_cols].round(3)

    # Add reference date column
    logging.info("Loading reference_date data from %s", date_table)
    date_df = run_query(
        f"SELECT na_entity_id, nrevx_year AS reference_date FROM {date_table}"
    )
    df = df.merge(date_df, on="na_entity_id", how="left")

    # Ensure desired column order
    logging.info("Loading column order data from %s", data_dictionary_table)
    columns_df = run_query(f"SELECT col_name FROM {data_dictionary_table}")
    desired_col_order = columns_df["col_name"].tolist()
    df = df[desired_col_order]

    return df
