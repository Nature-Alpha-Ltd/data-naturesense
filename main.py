import logging
import os
import re
from configparser import ConfigParser
from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.gcp_tools import (
    get_git_branch,
    last_day_of_month,
    run_query,
    save_results,
    write_df_to_bq,
)

# Configure logging
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 1. Read your config.ini
config = ConfigParser()
config.read("config.ini")

# Grab the section name from the ENVIRONMENT env‑var
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")

# Fetch that section (this yields a SectionProxy)
cfg = config[ENVIRONMENT]

PROJECT_ID = cfg["PROJECT_ID"]
BQ_DATASET = cfg["BQ_DATASET"]
ALD = cfg["ALD"]
ASSET_COUNTS_GUESTIMATES = cfg["ASSET_COUNTS_GUESTIMATES"]
NATURESENSE_COUNTRY = cfg["NATURESENSE_COUNTRY"]
FILE_NAME = cfg["FILE_NAME"]

# Define NatureSense metrics
naturesense_metrics = [
    "sensitive_locations",
    "biodiversity_importance",
    "high_ecosystem_integrity",
    "decline_in_ecosystem_integrity",
    "physical_water_risk",
    "ecosystem_services_provision_importance",
    "proximity_to_protected_areas",
    "proximity_to_kbas",
    "species_rarity_weighted_richness",
    "species_threat_abatement",
    "species_threat_abatement_marine",
    "proximity_to_mangroves",
    "ecosystem_intactness_index",
    "biodiversity_intactness_index",
    "ocean_health_index",
    "trend_in_ecosystem_intactness_index",
    "deforestation_hotspots",
    "water_availability",
    "water_pollution",
    "drought",
    "riverine_flood",
    "coastal_flood",
    "cumulative_impact_on_oceans",
    "critical_areas_for_biodiversity_and_ncp",
    "areas_of_importance_for_biodiversity_and_climate",
]


def load_data() -> tuple:
    """Load all required data from BigQuery using consistently formed SQL queries."""
    # ALD
    query_ald = f"""
    SELECT 
        na_entity_id, 
        entity_isin, 
        entity_name,
        priority_asset,
        asset_type_id,
        sensitive_locations, 
        biodiversity_importance, 
        high_ecosystem_integrity, 
        decline_in_ecosystem_integrity,
        physical_water_risk, 
        ecosystem_services_provision_importance, 
        proximity_to_protected_areas, 
        proximity_to_kbas,
        species_rarity_weighted_richness, 
        species_threat_abatement, 
        species_threat_abatement_marine, 
        proximity_to_mangroves,
        ecosystem_intactness_index, 
        biodiversity_intactness_index, 
        ocean_health_index, 
        trend_in_ecosystem_intactness_index,
        deforestation_hotspots, 
        water_availability, 
        water_pollution, 
        drought, 
        riverine_flood, 
        coastal_flood, 
        cumulative_impact_on_oceans, 
        critical_areas_for_biodiversity_and_ncp, 
        areas_of_importance_for_biodiversity_and_climate,
        in_water_scarcity
    FROM {ALD};
    """
    logging.info("Loading data from %s", ALD)
    ald = run_query(query_ald)

    # Asset counts guestimates
    query_assets_guestimates = f"""
    SELECT
        *
    FROM {ASSET_COUNTS_GUESTIMATES};
    """
    logging.info("Loading data from %s", ASSET_COUNTS_GUESTIMATES)
    assets_guestimates = run_query(query_assets_guestimates)

    # NatureSense country level
    query_ns_country = f"""
    SELECT
        *
    FROM {NATURESENSE_COUNTRY};
    """
    logging.info("Loading data from %s", NATURESENSE_COUNTRY)
    naturesense_country = run_query(query_ns_country)

    return ald, assets_guestimates, naturesense_country


def get_company_countries(company_row: pd.Series) -> List[str]:
    """
    Extract valid ISO country codes from company row where asset count > 0.

    Parameters
    ----------
    company_row : pd.Series
        Row containing company's country distribution

    Returns
    -------
    List[str]
        List of valid ISO country codes with positive asset counts
    """
    iso_pattern = re.compile(r"^[A-Z0-9]{3}$")
    return [c for c in company_row.index if iso_pattern.match(c) and company_row[c] > 0]


def calculate_country_prior(
    company_row: pd.Series,
    country_priors: pd.DataFrame,
    evidence_columns: Union[str, List[str]],
    country_codes: List[str],
    entity_id: str,
) -> List[float]:
    """
    Calculate the country-weighted prior for a company for one or multiple geospatial columns.
    Uses vectorized operations to compute weighted averages for all evidence columns at once.
    Returns None for each column if there are no valid weights.
    """
    # Ensure evidence_columns is a list
    if isinstance(evidence_columns, str):
        evidence_columns = [evidence_columns]

    # Obtain guestimated countries with company presence
    company_countries = get_company_countries(company_row)

    # Check which countries have country avg prior available
    available_countries = [c for c in company_countries if c in country_codes]
    missing_countries = set(company_countries) - set(country_codes)

    if missing_countries:
        logger.info(
            f"NA_entity_id {entity_id} has assets in countries missing from priors: {missing_countries}. "
            "These will be excluded from the weighted average calculation."
        )

    if company_row.empty:
        logger.warning(f"No country distribution found for NA_entity_id {entity_id}")
        return [None] * len(evidence_columns)

    # Get the priors matrix for available countries and evidence columns
    priors_matrix = country_priors[
        country_priors["country_code"].isin(available_countries)
    ]
    priors_matrix = priors_matrix.set_index("country_code")[evidence_columns]

    # Create weights array for available countries
    weights = pd.Series(
        {country: company_row[country] for country in available_countries}
    )

    # If no weights or all weights are zero, return None for each column
    if not available_countries or weights.sum() == 0:
        return [None] * len(evidence_columns)

    # Normalize weights and ensure index alignment
    weights = weights / weights.sum()
    weights = weights.reindex(priors_matrix.index)

    # Compute weighted average for all columns at once using matrix multiplication
    weighted_priors = weights.dot(priors_matrix)

    # Convert to list and handle any NaN values
    return [float(val) if pd.notnull(val) else None for val in weighted_priors]


def calculate_effective_k(
    k: float, estimated_material_assets_count: int, material_assets_count: int
) -> float:
    """
    Calculate the effective k value based on various scenarios.

    Parameters
    ----------
    k : float
        Original k value. If < 1, interpreted as proportion of estimated_material_assets_count
    estimated_material_assets_count : int
        Estimated total number of company assets in the world
    material_assets_count : int
        Actual number of discovered company assets

    Returns
    -------
    float
        Adjusted k value
    """
    # If k is proportional, convert to absolute number
    # Ensure effective_k is at least 1 if estimated_material_assets_count is greater than 0
    if k < 1:
        effective_k = (
            max(1, np.ceil(k * estimated_material_assets_count))
            if estimated_material_assets_count > 0
            else 0
        )
    else:
        effective_k = k

    # If we found more assets than estimated locations, update our estimate
    if (
        estimated_material_assets_count != 0
        and material_assets_count > estimated_material_assets_count
    ):
        estimated_material_assets_count = material_assets_count

    # If estimated total locations is less than k, adjust k down
    if estimated_material_assets_count < effective_k:
        effective_k = estimated_material_assets_count

    return float(effective_k)


def no_guestimates_adjust_priors_and_k(
    weighted_priors: Union[List[float], None],
    effective_k: float,
    k: float,
    default_priors: Union[List[float], None],
) -> Tuple[List[float], float]:
    """
    Adjust priors and effective_k for companies with no prior and few assets.

    Parameters
    ----------
    weighted_priors : Union[List[float], None]
        List of weighted priors, may contain None values, or be None itself
    effective_k : float
        Current effective k value
    k : float
        Original k value
    default_priors : Union[List[float], None]
        List of default prior values from global medians

    Returns
    -------
    Tuple[List[float], float]
        Adjusted weighted priors and effective k value
    """
    if weighted_priors is None or any(p is None for p in weighted_priors):
        if effective_k < k:
            if weighted_priors is None:
                weighted_priors = default_priors.copy()
            else:
                weighted_priors = [
                    default_priors[i] if p is None else p
                    for i, p in enumerate(weighted_priors)
                ]
            effective_k = k
    return weighted_priors, effective_k


def compute_posterior(
    evidences: pd.DataFrame,
    priors: pd.DataFrame,
    sample_size: int,
    k: float,
) -> pd.DataFrame:
    """
    Compute posterior scores by combining entity-specific evidences with priors
    based on sample size.

    Parameters
    ----------
    evidences : pd.DataFrame
        DataFrame containing evidence values for each metric
    priors : pd.DataFrame
        DataFrame containing prior values for each metric
    sample_size : int
        Number of assets/samples for the company
    k : float
        Threshold value for sample size adjustment

    Returns
    -------
    pd.Series
        Series containing posterior values for each metric, indexed by metric names
    """
    try:
        # Check if both inputs are DataFrames
        if not isinstance(evidences, pd.DataFrame) or evidences.empty:
            logger.error("evidences must be a non-empty DataFrame")
            return None

        if not isinstance(priors, pd.DataFrame) or priors.empty:
            logger.error("priors must be a non-empty DataFrame")
            return None

        if not isinstance(k, (int, float)) or k < 0:
            logger.error("k must be a positive number")
            return None

        # Handle edge cases
        if k == 0:
            if sample_size == 0:
                return pd.DataFrame(0, index=evidences.index, columns=evidences.columns)
            return evidences.iloc[0]
        elif sample_size == 0:
            return priors.iloc[0]

        # Compute weights safely, avoiding NaN by ensuring effective_k is never zero
        adapted_k = min(sample_size / k, 1)
        w_i = 1 if sample_size == 0 else adapted_k

        # Compute posterior using vectorized operations
        theta_i = w_i * evidences + (1 - w_i) * priors

        return theta_i.iloc[0]

    except Exception as e:
        logger.error(f"Error during posterior computation: {str(e)}")
        raise


def process_company_evidence(
    company_data: pd.DataFrame,
    country_dist: pd.DataFrame,
    country_priors: pd.DataFrame,
    evidence_columns: List[str],
    global_priors: Union[List[float], None],
    k: Union[int, float] = 10,
) -> pd.DataFrame:
    """
    Process company evidence using country distribution and priors.

    Parameters
    ----------
    company_data : pd.DataFrame
        Company-level data containing:
        - na_entity_id: Company identifier
        - material_assets_count: Number of total assets
        - evidence_columns: Columns containing evidence to process
    country_dist : pd.DataFrame
        Country distribution data containing:
        - na_entity_id: Company identifier
        - total_company_locations: Optional total locations count
        - One column per country code with number of assets
    country_priors : pd.DataFrame
        Country-level priors containing:
        - country_code: ISO alpha-3 country code
        - columns matching evidence_columns with prior values
    evidence_columns : List[str]
        List of column names in company_data to process
    global_priors : Union[List[float], None]
        List of default prior values from global medians
    k : int or float, optional
        If k > 1: interpreted as absolute number of asset locations
        If k ≤ 1: interpreted as proportion of company's total locations
        Default is 10.

    Returns
    -------
    pd.DataFrame
        DataFrame with original company data plus new posterior columns
    """
    logger.info("Starting company evidence processing")

    try:
        # Validate input company_data
        company_data_required_cols = [
            "na_entity_id",
            "material_assets_count",
            *evidence_columns,
        ]

        if not all(col in company_data.columns for col in company_data_required_cols):
            missing_cols = set(company_data_required_cols) - set(company_data.columns)
            raise ValueError(
                f"Missing required columns in company_data: {missing_cols}"
            )

        # Validate input country_dist
        if "na_entity_id" not in country_dist.columns:
            raise ValueError("country_dist must contain 'na_entity_id' column")

        ## Check if country_dist has any country codes not in country_priors
        country_codes = country_priors["country_code"].tolist()
        country_dist_basic_cols = [
            "na_entity_id",
            "entity_isin",
            "entity_name",
            "factset_entity_name",
            "factset_coverage_name",
            "total_company_locations",
            "number_material_assets",
            "primary_sector",
            "partition_date",
        ]
        invalid_country_codes = [
            col
            for col in country_dist.columns
            if col not in country_codes and col not in country_dist_basic_cols
        ]

        if invalid_country_codes:
            raise ValueError(
                f"country_dist contains country codes not found in country_priors: {invalid_country_codes}"
            )

        ## Sum of the assets listed under each country code
        country_dist_copy = country_dist.copy()
        country_cols = [col for col in country_dist.columns if col in country_codes]
        country_assets_sum = country_dist_copy[country_cols].sum(axis=1)
        country_dist_copy["total_company_locations"] = country_assets_sum

        # Validate input country_priors
        available_evidence = [
            col for col in evidence_columns if col in country_priors.columns
        ]
        missing_evidence = set(evidence_columns) - set(available_evidence)

        if missing_evidence:
            raise ValueError(
                f"Following evidence columns not found in country_priors: {missing_evidence}"
            )

        # Create output DataFrame
        result_df = company_data.copy()

        # Add posterior columns to result_df, initialized as copies of original columns
        for col in evidence_columns:
            result_df[f"{col}_posterior"] = result_df[col]

        # Add estimated_material_assets_count column initialized with 0
        result_df["estimated_material_assets_count"] = 0

        # Keep track of missing entity ids
        missing_entity_ids = []

        # Process each company
        for entity_id in tqdm(
            result_df["na_entity_id"].unique(), desc="Implementing NatureSense Priors"
        ):
            # Initialize
            material_assets_count = int(
                company_data.loc[
                    company_data["na_entity_id"] == entity_id, "material_assets_count"
                ].iloc[0]
            )

            estimated_material_assets_count = 0

            # If material_assets_count >= k don't adjust company evidence
            if material_assets_count >= k:
                continue

            # Get company evidence values for all columns
            company_evidence = result_df.loc[
                result_df["na_entity_id"] == entity_id, evidence_columns
            ].iloc[0]

            # Get company country distribution and estimated_material_assets_count
            if entity_id not in country_dist_copy["na_entity_id"].values:
                missing_entity_ids.append(entity_id)
                # Initialize with correct length
                weighted_priors = [None] * len(evidence_columns)
            else:
                company_row = country_dist_copy[
                    country_dist_copy["na_entity_id"] == entity_id
                ].iloc[0]

                estimated_material_assets_count = company_row["total_company_locations"]

                # Update estimated_material_assets_count in result_df
                result_df.loc[
                    result_df["na_entity_id"] == entity_id,
                    "estimated_material_assets_count",
                ] = estimated_material_assets_count

                # If both material_assets_count and estimated_material_assets_count are 0, set all evidence columns to None
                if material_assets_count == 0 and estimated_material_assets_count == 0:
                    for col in evidence_columns:
                        result_df.loc[
                            result_df["na_entity_id"] == entity_id, f"{col}_posterior"
                        ] = np.nan
                    continue

                # Get weighted priors
                weighted_priors = calculate_country_prior(
                    company_row=company_row,
                    country_priors=country_priors,
                    evidence_columns=evidence_columns,
                    country_codes=country_codes,
                    entity_id=entity_id,
                )

            # Calculate effective k
            effective_k = calculate_effective_k(
                k=k,
                estimated_material_assets_count=estimated_material_assets_count,
                material_assets_count=material_assets_count,
            )

            # Adjust priors and k if necessary
            weighted_priors, effective_k = no_guestimates_adjust_priors_and_k(
                weighted_priors, effective_k, k, global_priors
            )

            # Prepare DataFrames to compute posteriors
            company_evidence_df = pd.DataFrame(
                company_evidence[evidence_columns].astype(float).values.reshape(1, -1),
                columns=evidence_columns,
                index=[0],
            )

            weighted_priors_df = pd.DataFrame(
                {
                    col: [weighted_priors[idx]]
                    for idx, col in enumerate(evidence_columns)
                },
                index=[0],
            )

            # Compute posteriors
            posteriors = compute_posterior(
                evidences=company_evidence_df,
                priors=weighted_priors_df,
                sample_size=material_assets_count,
                k=effective_k,
            )

            # Populate result_df with posterior values
            posterior_cols = [f"{col}_posterior" for col in evidence_columns]
            result_df.loc[
                result_df["na_entity_id"] == entity_id, posterior_cols
            ] = posteriors.to_numpy(dtype="float64")

        # Round posterior values to 3 decimals
        result_df[posterior_cols] = result_df[posterior_cols].round(3)

        if missing_entity_ids:
            logger.warning(
                f"{len(missing_entity_ids)} companies were not found in company country distribution data"
            )

        logger.info("Completed company evidence processing")
        return result_df

    except Exception as e:
        logger.error(f"Error in process_company_evidence: {str(e)}")
        raise


def main(request):
    """
    Cloud Function entrypoint.
    Processes input (if provided), calculates metrics, and returns a status message.
    """
    try:
        # Load data from BigQuery
        ald, assets_guestimates, naturesense_country = load_data()

        # Generate companies evidences, i.e., aggregate ALD to company
        ald["material_asset"] = ~ald["asset_type_id"].isin([11, 12]).astype(bool)

        ald["in_water_scarcity"] = (
            (ald["water_availability"] > 0.6) & (ald["material_asset"] == True)
        ).astype(bool)

        ald_counts = (
            ald.groupby("na_entity_id")
            .agg(
                assets_count=("na_entity_id", "count"),
                priority_assets_count=("priority_asset", "sum"),
                material_assets_count=("material_asset", "sum"),
                in_water_scarcity_count=("in_water_scarcity", "sum"),
            )
            .reset_index()
        )

        ald_counts["priority_assets_percentage"] = round(
            (ald_counts["priority_assets_count"] / ald_counts["assets_count"]) * 100, 3
        )

        ald_counts["in_water_scarcity_percentage"] = round(
            (ald_counts["in_water_scarcity_count"] / ald_counts["assets_count"]) * 100,
            3,
        )

        ald_material = ald[ald["material_asset"] == True]

        ald_averages = (
            ald_material.groupby("na_entity_id")
            .agg(
                **{
                    f"{col}": (col, lambda x: round(x.mean(skipna=True), 3))
                    for col in naturesense_metrics
                }
            )
            .reset_index()
        )

        companies_evidences = ald_counts.merge(
            ald_averages, on="na_entity_id", how="left"
        )

        # Calculate global median for each metric in naturesense_metrics
        ald_global_median = {
            metric: round(ald_material[metric].median(skipna=True), 3)
            for metric in naturesense_metrics
        }

        ald_global_median = [float(val) for val in ald_global_median.values()]

        # Main process
        result = process_company_evidence(
            company_data=companies_evidences,
            country_dist=assets_guestimates,
            country_priors=naturesense_country,
            evidence_columns=naturesense_metrics,
            global_priors=ald_global_median,
            k=10,
        )

        # Organise columns
        result = result[
            [
                "na_entity_id",
                "assets_count",
                "material_assets_count",
                "estimated_material_assets_count",
                "priority_assets_count",
                "priority_assets_percentage",
                "in_water_scarcity_count",
                "in_water_scarcity_percentage",
                *naturesense_metrics,
                *[f"{col}_posterior" for col in naturesense_metrics],
            ]
        ]

        # Write results to BigQuery
        save_results(result, FILE_NAME, BQ_DATASET, ENVIRONMENT, PROJECT_ID)

        return f"{FILE_NAME} metrics calculated and saved successfully.", 200

    except Exception as e:
        logging.exception("Error during metrics calculation")
        raise


if __name__ == "__main__":
    # For local testing; in Cloud Functions, `main` is used as the entrypoint.
    main(request=None)
