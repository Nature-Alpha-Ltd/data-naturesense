import logging
import os
from configparser import ConfigParser
from datetime import datetime

import pandas as pd

from utils.gcp_tools import (
    get_git_branch,
    last_day_of_month,
    run_query,
    save_results,
    write_df_to_bq,
)

# 1. Read your config.ini
config = ConfigParser()
config.read("config.ini")

# 2. Grab the section name from the ENVIRONMENT env‑var
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")

# 3. Fetch that section (this yields a SectionProxy)
cfg = config[ENVIRONMENT]

PROJECT_ID = cfg["PROJECT_ID"]
BQ_DATASET = cfg["BQ_DATASET"]
ALD = cfg["ALD"]
ASSET_COUNTS_GUESTIMATES = cfg["ASSET_COUNTS_GUESTIMATES"]
NATURESENSE_COUNTRY = cfg["NATURESENSE_COUNTRY"]

# Configure logging
logging.basicConfig(level=logging.INFO)


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


def main(request):
    """
    Cloud Function entrypoint.
    Processes input (if provided), calculates metrics, and returns a status message.
    """
    try:
        # Load data
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

        ald_subset = ald[ald["material_asset"] == True]

        columns_to_average = [
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

        ald_averages = (
            ald_subset.groupby("na_entity_id")
            .agg(
                **{
                    f"{col}": (col, lambda x: round(x.mean(skipna=True), 4))
                    for col in columns_to_average
                }
            )
            .reset_index()
        )

        companies_evidences = ald_counts.merge(
            ald_averages, on="na_entity_id", how="left"
        )

        return "Processing completed successfully"

    except Exception as e:
        logging.exception("Error during metrics calculation")
        raise


if __name__ == "__main__":
    # For local testing; in Cloud Functions, `main` is used as the entrypoint.
    main(request=None)
