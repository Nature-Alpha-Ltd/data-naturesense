"""
This module contains functions for calculating nature materiality scores,
aggregating impact and dependency measures, and computing revenue-based metrics.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Configure logging at the module level
logging.basicConfig(level=logging.INFO)

# Module-level dependency groups for different dependency categories
dependency_groups = {
    "provisioning": [
        "dependency_fibres_and_other_materials_metric",
        "dependency_ground_water_metric",
        "dependency_surface_water_metric",
    ],
    "regulating": [
        "dependency_pollination_metric",
        "dependency_water_quality_metric",
        "dependency_ventilation_metric",
        "dependency_climate_regulation_metric",
        "dependency_water_regulation_metric",
        "dependency_disease_control_metric",
        "dependency_flood_and_storm_protection_metric",
        "dependency_pest_control_metric",
        "dependency_mass_stabilisation_and_erosion_control_metric",
    ],
    "supporting": [
        "dependency_soil_quality_metric",
        "dependency_soil_formation_and_fertility_metric",
        "dependency_nutrient_cycling_metric",
        "dependency_maintain_nursery_habitats_metric",
    ],
    "cultural": [
        "dependency_recreation_and_tourism_metric",
        "dependency_traditional_knowledge_and_livelihoods_metric",
    ],
}

# Predefined impact columns
impact_columns = [
    "impact_biodiversity_disturbances_metric",
    "impact_land_use_change_metric",
    "impact_freshwater_ecosystem_use_metric",
    "impact_ghg_emissions_metric",
    "impact_marine_ecosystem_use_metric",
    "impact_non_ghg_emissions_metric",
    "impact_other_resource_use_metric",
    "impact_social_impact_on_indigenous_communities_metric",
    "impact_pollutants_metric",
    "impact_solid_waste_metric",
]


def nature_materiality_scores() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate predefined materiality scores for impacts and dependencies.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - Impact weights as a DataFrame.
            - Dependency weights as a DataFrame.
    """
    logging.info("Retrieving predefined nature materiality scores")

    impact_weights = pd.DataFrame(
        data=[
            [
                0.39219115,
                0.13324588,
                0.3725284,
                0.2920244,
                0.27300596,
                0.34515902,
                0.31929635,
                0.31247397,
                0.37236367,
                0.26919162,
            ]
        ],
        columns=impact_columns,
    )

    predefined_dependency_weights = {
        "provisioning": [0.3181, 0.5589, 0.7658],
        "regulating": [
            0.1734,
            0.6244,
            0.1176,
            0.226,
            0.5158,
            0.1628,
            0.1928,
            0.267,
            0.3379,
        ],
        "supporting": [0.56738625, 0.57101461, 0.55446907, 0.21113787],
        "cultural": [0.8454, 0.5341],
    }

    dependency_cols = [col for group in dependency_groups.values() for col in group]

    # Initialize a dataframe filled with zeros
    dependency_weights_df = pd.DataFrame(
        0, index=dependency_groups.keys(), columns=dependency_cols
    )

    # Assign weights to the respective columns for each category, so 4 rows
    # (for provisioning, regulating, supporting, cultural) and 18 columns (for the 18 dependencies).
    for category, columns in dependency_groups.items():
        dependency_weights_df = dependency_weights_df.astype(
            float
        )  # Ensure DataFrame is float type
        dependency_weights_df.loc[category, columns] = pd.Series(
            predefined_dependency_weights[category], index=columns
        ).astype(float)

    return impact_weights, dependency_weights_df


def nature_risk_calculation(
    df: pd.DataFrame,
    columns: list,
    zscore_scale: bool = False,
    custom_mean: float = None,
    custom_std: float = None,
) -> np.ndarray:
    """
    Process specified columns by z-scoring, combining them, and converting to a probability score.

    Parameters:
        df : pd.DataFrame
            DataFrame containing the columns to process.
        columns : list
            List of column names to process.
        custom_mean : float, optional
            Custom mean for z-scoring; if None, the column's mean is used.
        custom_std : float, optional
            Custom standard deviation for z-scoring; if None, the column's std is used.
        zscore_scale : bool, optional
            If True, returns z-scored values; otherwise returns probability scores (0 to 1).

    Returns:
        np.ndarray: The final probability scores (or z-scores if zscore_scale is True).
    Notes
    -----
    The function performs the following steps:
    1. Validates input columns exist in the DataFrame
    2. Z-scores each column individually (using custom or column-specific statistics)
    3. Combines z-scored columns
    4. Z-scores the combined score again
    5. Converts final z-scores to probabilities between 0 and 1
    """
    df = df.copy()

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if not isinstance(columns, list):
        raise TypeError("Columns must be provided as a list")
    if not all(col in df.columns for col in columns):
        missing_cols = [col for col in columns if col not in df.columns]
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    logging.info("Processing %d columns: %s", len(columns), ", ".join(columns))

    # Ensure columns are numeric and log any NaN values found
    for col in columns:
        df.loc[:, col] = pd.to_numeric(
            df[col], errors="coerce"
        )  # Use .loc to avoid warning
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            logging.warning(
                "Column '%s' contains %d NaN values which will be removed",
                col,
                nan_count,
            )

    df_clean = df.dropna(subset=columns)
    if df_clean.empty:
        raise ValueError("No valid data remains after removing NaN values")

    processed_columns = []
    for col in columns:
        series = df_clean[col].copy()
        logging.debug(
            "Column %s: mean=%.3f, std=%.3f", col, series.mean(), series.std()
        )
        processed_columns.append(series)

    if not processed_columns:
        raise ValueError("No valid columns to process after cleaning data")

    logging.info("Combining processed columns")
    combined_score = sum(processed_columns) / len(processed_columns)

    logging.info("Z-scoring combined score")
    mean_to_use = custom_mean if custom_mean is not None else combined_score.mean()
    std_to_use = (
        custom_std
        if custom_std is not None and custom_std != 0
        else combined_score.std()
    )
    final_zscore = (combined_score - mean_to_use) / std_to_use

    if not zscore_scale:
        logging.info("Converting combined z-scores to probability scores")
        probability_scores = stats.norm.cdf(final_zscore)
    else:
        probability_scores = final_zscore

    # Align result with original DataFrame index
    result = pd.Series(data=np.nan, index=df.index, dtype=float)
    result.loc[df_clean.index] = probability_scores

    logging.info(
        "Final score: min=%.3f, max=%.3f, mean=%.3f",
        probability_scores.min(),
        probability_scores.max(),
        probability_scores.mean(),
    )

    return result.to_numpy()


def log_nan_info(df: pd.DataFrame, column: str) -> None:
    """
    Log details about NaN values in a specific column.

    Parameters:
        df : pd.DataFrame
            DataFrame to check.
        column : str
            Name of the column to inspect.
    """
    nan_count = df[column].isna().sum()
    if nan_count > 0:
        nan_rows = df[df[column].isna()]
        logging.warning(
            "Column '%s' contains %d NaN values. GICS codes: %s; GICS names: %s",
            column,
            nan_count,
            nan_rows.get("gics_sub_industry_code", pd.Series([])).tolist(),
            nan_rows.get("gics_sub_industry_name", pd.Series([])).tolist(),
        )


def nrevx_aggregation_dependency(
    nrevx_df: pd.DataFrame, calculate_pca_weights: bool = False
) -> pd.DataFrame:
    """
    Aggregates dependency scores for the DataFrame based on materiality scores.

    Parameters:
        nrevx_df : pd.DataFrame
            Input DataFrame containing raw dependency data.
        calculate_pca_weights : bool
            Flag indicating whether to recalculate PCA weights (unused).

    Returns:
        pd.DataFrame: Updated DataFrame with aggregated dependency metrics.
    """
    df = nrevx_df.copy()
    _, dependency_weights = nature_materiality_scores()

    for category, cols in dependency_groups.items():
        # Remove the '_metric' suffix to match df columns
        expected_cols = [col.replace("_metric", "") for col in cols]
        matching_cols = [col for col in expected_cols if col in df.columns]
        missing_cols = [col for col in expected_cols if col not in df.columns]

        if not matching_cols:
            logging.warning(
                "No matching columns found for category '%s'. Expected: %s",
                category,
                ", ".join(expected_cols),
            )
            continue

        if missing_cols:
            logging.warning(
                "In category '%s', %d of %d expected columns are missing: %s",
                category,
                len(matching_cols),
                len(expected_cols),
                ", ".join(missing_cols),
            )

        # Prepare weights; add back '_metric' to match the dependency_weights DataFrame columns
        category_cols = [col + "_metric" for col in matching_cols]
        category_weights = dependency_weights.loc[category, category_cols]
        category_weights.index = matching_cols  # align indices

        agg_column_name = f"agg_{category.split('_')[-1].lower()}"
        df[agg_column_name] = df[matching_cols].dot(category_weights)
        log_nan_info(df, agg_column_name)

        # Define custom scaling parameters per aggregation category
        if agg_column_name == "agg_cultural":
            custom_mean, custom_std = 0.12606244992795387, 0.17369946355268806
        elif agg_column_name == "agg_supporting":
            custom_mean, custom_std = 0.19285089996912308, 0.3078010084200954
        elif agg_column_name == "agg_regulating":
            custom_mean, custom_std = 0.8071865223857555, 0.368678399723487
        elif agg_column_name == "agg_provisioning":
            custom_mean, custom_std = 0.48140205225401395, 0.3327861596590439
        else:
            custom_mean, custom_std = None, None

        if custom_mean is not None:
            df[f"{agg_column_name}_probability"] = nature_risk_calculation(
                df,
                [agg_column_name],
                zscore_scale=False,
                custom_mean=custom_mean,
                custom_std=custom_std,
            )
            df[agg_column_name] = nature_risk_calculation(
                df,
                [agg_column_name],
                zscore_scale=True,
                custom_mean=custom_mean,
                custom_std=custom_std,
            )

    if all(
        col in df.columns
        for col in [
            "agg_cultural",
            "agg_supporting",
            "agg_regulating",
            "agg_provisioning",
        ]
    ):
        valid_rows = (
            df[["agg_cultural", "agg_supporting", "agg_regulating", "agg_provisioning"]]
            .notna()
            .all(axis=1)
        )
        df["final_dependency_zscore"] = np.nan
        df.loc[valid_rows, "final_dependency_zscore"] = nature_risk_calculation(
            df[valid_rows],
            ["agg_cultural", "agg_supporting", "agg_regulating", "agg_provisioning"],
            zscore_scale=True,
            custom_mean=0.5,
            custom_std=1.1,
        )
        log_nan_info(df, "final_dependency_zscore")

        df["final_dependency"] = np.nan
        df.loc[valid_rows, "final_dependency"] = nature_risk_calculation(
            df[valid_rows],
            ["agg_cultural", "agg_supporting", "agg_regulating", "agg_provisioning"],
            zscore_scale=False,
            custom_mean=0.5,
            custom_std=1.1,
        )
        log_nan_info(df, "final_dependency")

    return df


def nrevx_aggregation_impact(nrevx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates impact scores for the DataFrame using predefined impact weights.

    Parameters:
        nrevx_df : pd.DataFrame
            Input DataFrame containing raw impact data.

    Returns:
        pd.DataFrame: Updated DataFrame with calculated final impact scores.
    """
    df = nrevx_df.copy()
    impact_weights, _ = nature_materiality_scores()
    # Adjust column names for matching
    impact_weights.columns = [
        col.replace("_metric", "") for col in impact_weights.columns
    ]

    impact_cols_existing = [col for col in impact_weights.columns if col in df.columns]
    missing_cols = [col for col in impact_weights.columns if col not in df.columns]

    if not impact_cols_existing:
        logging.warning(
            "No impact columns found in data - skipping final_impact calculation"
        )
        return df

    if len(impact_cols_existing) < len(impact_weights.columns):
        logging.warning(
            "Only %d out of %d impact columns present. Missing: %s",
            len(impact_cols_existing),
            len(impact_weights.columns),
            ", ".join(missing_cols),
        )

    impact_weight_values = impact_weights.iloc[0][impact_cols_existing]
    df["final_impact_mean"] = (df[impact_cols_existing] * impact_weight_values).mean(
        axis=1
    )

    df["final_impact"] = nature_risk_calculation(
        df,
        ["final_impact_mean"],
        zscore_scale=False,
        custom_mean=0.15718658124136442,
        custom_std=0.144003262385216,
    )
    log_nan_info(df, "final_impact")
    df["final_impact_zscore"] = nature_risk_calculation(
        df,
        ["final_impact_mean"],
        zscore_scale=True,
        custom_mean=0.15625151998029058,
        custom_std=0.07147852894889803,
    )
    df.drop(columns=["final_impact_mean"], inplace=True)
    return df


def rbics_materiality_score_calculation(
    rbics_df: pd.DataFrame,
    sector_mapping_df: pd.DataFrame,
    materiality_df: pd.DataFrame,
    df_isins: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combines several DataFrames to compute materiality scores by multiplying revenue splits
    with impact and dependency measures.

    Parameters:
        rbics_df : pd.DataFrame
            DataFrame containing companies' revenue splits.
        sector_mapping_df : pd.DataFrame
            DataFrame containing sector mappings.
        materiality_df : pd.DataFrame
            DataFrame containing nature impact and dependency variables.
        df_isins : pd.DataFrame
            DataFrame with ISIN-related identifiers.

    Returns:
        pd.DataFrame: Aggregated DataFrame with computed nature materiality metrics.
    """
    # Validate required columns
    required_rbics_columns = {"l6_revenue_pct", "l6_id", "entity_isin"}
    required_sector_columns = {"gics_sub_industry_code", "l6_id"}
    required_materiality_columns = {"gics_sub_industry_code"}

    if not required_rbics_columns.issubset(rbics_df.columns):
        raise ValueError(f"rbics_df must contain columns: {required_rbics_columns}")
    if not required_sector_columns.issubset(sector_mapping_df.columns):
        raise ValueError(
            f"sector_mapping_df must contain columns: {required_sector_columns}"
        )
    if not required_materiality_columns.issubset(materiality_df.columns):
        raise ValueError(
            f"materiality_df must contain columns: {required_materiality_columns}"
        )

    materiality_df_prep = materiality_df.copy()
    rbics_rescaled = rbics_df.copy()

    merged_df = pd.merge(
        rbics_rescaled,
        sector_mapping_df[["gics_sub_industry_code", "l6_id"]].drop_duplicates(
            subset="l6_id"
        ),
        on="l6_id",
        how="left",
    )

    grouped_df = (
        merged_df.groupby(["entity_isin", "gics_sub_industry_code"])["l6_revenue_pct"]
        .sum()
        .reset_index()
        .sort_values("entity_isin")
    )

    materiality_subset = materiality_df_prep.filter(
        regex="^(impact_|dependency_)"
    ).join(materiality_df_prep["gics_sub_industry_code"])
    final_merged_df = pd.merge(
        grouped_df,
        materiality_subset,
        on="gics_sub_industry_code",
        how="left",
    )

    columns_to_modify = (
        final_merged_df.filter(like="impact_").columns.tolist()
        + final_merged_df.filter(like="dependency_").columns.tolist()
    )

    multiplied_df = final_merged_df.copy()
    for col in columns_to_modify:
        multiplied_df[col] = multiplied_df["l6_revenue_pct"] * multiplied_df[col]

    aggregation_functions = {
        col: "sum"
        for col in multiplied_df.columns
        if col not in ["entity_isin", "entity_proper_name"]
    }
    final_result_df = (
        multiplied_df.groupby("entity_isin").agg(aggregation_functions).reset_index()
    )

    final_result_df = pd.merge(
        df_isins[["na_entity_id", "entity_isin", "entity_name"]],
        final_result_df,
        on="entity_isin",
        how="inner",
    ).drop(columns=["gics_sub_industry_code"])

    final_result_df = final_result_df[final_result_df["l6_revenue_pct"] != 0]
    logging.info(
        "Rows after merging ISIN data & excluding 0 revenue: %s", final_result_df.shape
    )

    final_result_df.iloc[:, 4:] = final_result_df.iloc[:, 4:].div(100)

    # Final aggregation for impact and dependency
    nrevx_dependencies_df = nrevx_aggregation_dependency(final_result_df)
    nrevx_impacts_df = nrevx_aggregation_impact(nrevx_dependencies_df)

    return nrevx_impacts_df
