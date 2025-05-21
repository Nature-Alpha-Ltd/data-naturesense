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

# from utils.nrevx_engine import rbics_materiality_score_calculation
# from utils.post_process import postprocess_dataframe

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
COUNTRY_PRIORS = cfg["COUNTRY_PRIORS"]
ASSET_COUNTS_ESTIMATES = cfg["ASSET_COUNTS_ESTIMATES"]

# Configure logging
logging.basicConfig(level=logging.INFO)


def load_data() -> tuple:
    """Load all required data from BigQuery using consistently formed SQL queries."""
    query_entities = f"""
    SELECT na_entity_id, entity_isin, entity_name
    FROM {ENTITIES_TABLE_ID};
    """
    logging.info("Loading entities from %s", ENTITIES_TABLE_ID)
    entities_df = run_query(query_entities)

    query_mm = f"""
    SELECT
        gics_sub_industry_code,
        gics_sub_industry_name,
        impact_biodiversity_disturbances,
        impact_freshwater_ecosystem_use,
        impact_ghg_emissions,
        impact_marine_ecosystem_use,
        impact_non_ghg_emissions,
        impact_other_resource_use,
        impact_social_impact_on_indigenous_communities,
        impact_pollutants,
        impact_solid_waste,
        impact_land_use_change,
        dependency_climate_regulation,
        dependency_disease_control,
        dependency_fibres_and_other_materials,
        dependency_flood_and_storm_protection,
        dependency_ground_water,
        dependency_mass_stabilisation_and_erosion_control,
        dependency_nutrient_cycling,
        dependency_pest_control,
        dependency_pollination,
        dependency_recreation_and_tourism,
        dependency_soil_formation_and_fertility,
        dependency_soil_quality,
        dependency_surface_water,
        dependency_traditional_knowledge_and_livelihoods,
        dependency_ventilation,
        dependency_water_quality,
        dependency_water_regulation,
        dependency_maintain_nursery_habitats
    FROM {MM_TABLE}
    WHERE CONCAT(
        impact_biodiversity_disturbances,
        impact_freshwater_ecosystem_use,
        impact_marine_ecosystem_use,
        impact_ghg_emissions,
        impact_non_ghg_emissions,
        impact_other_resource_use,
        impact_pollutants,
        impact_solid_waste,
        impact_social_impact_on_indigenous_communities,
        impact_land_use_change,
        dependency_climate_regulation,
        dependency_disease_control,
        dependency_fibres_and_other_materials,
        dependency_flood_and_storm_protection,
        dependency_ground_water,
        dependency_mass_stabilisation_and_erosion_control,
        dependency_nutrient_cycling,
        dependency_pest_control,
        dependency_pollination,
        dependency_recreation_and_tourism,
        dependency_soil_formation_and_fertility,
        dependency_soil_quality,
        dependency_surface_water,
        dependency_traditional_knowledge_and_livelihoods,
        dependency_ventilation,
        dependency_water_quality,
        dependency_water_regulation,
        dependency_maintain_nursery_habitats
    ) IS NOT NULL;
    """
    logging.info("Loading MM data from %s", MM_TABLE)
    extended_mm = run_query(query_mm)

    query_sector = f"""
    SELECT DISTINCT
        gics_sub_industry_code,
        gics_sector_code,
        l6_id
    FROM {GICS_MAPPING}
    WHERE gics_sub_industry_code IS NOT NULL;
    """
    logging.info("Loading sector mapping from %s", GICS_MAPPING)
    sector_mapping_df = run_query(query_sector)

    query_rbics = f"SELECT * FROM {RBICS_L6};"
    logging.info("Loading RBICS revenue splits from %s", RBICS_L6)
    rbics_df = run_query(query_rbics)

    query_isin = f"""
    SELECT * EXCEPT(partition_date)
    FROM {ISIN_MAPPING};
    """
    logging.info("Loading ISIN mapping from %s", ISIN_MAPPING)
    isin_mapping_df = run_query(query_isin)

    return entities_df, extended_mm, sector_mapping_df, rbics_df, isin_mapping_df


def process_and_calculate(
    entities_df, extended_mm, sector_mapping_df, rbics_df, isin_mapping_df
) -> pd.DataFrame:
    """Merge the data and calculate file metrics."""
    rbics_isins_df = pd.merge(rbics_df, isin_mapping_df, on=["entity_isin"], how="left")
    logging.info(f"Calculating {FILE_NAME} metrics...")
    nature_revx_df = rbics_materiality_score_calculation(
        rbics_isins_df.rename(columns={"entity_name": "entity_proper_name"}),
        sector_mapping_df,
        extended_mm,
        df_isins=entities_df,
    )
    return nature_revx_df


def main(request):
    """
    Cloud Function entrypoint.
    Processes input (if provided), calculates metrics, and returns a status message.
    """
    try:
        validate_mappings()
        (
            entities_df,
            extended_mm,
            sector_mapping_df,
            rbics_df,
            isin_mapping_df,
        ) = load_data()
        nature_revx_df = process_and_calculate(
            entities_df, extended_mm, sector_mapping_df, rbics_df, isin_mapping_df
        )
        save_results(
            nature_revx_df, TABLE_NAME_OTHER, BQ_DATASET_OTHER, ENVIRONMENT, PROJECT_ID
        )
        ready_to_publish_df = postprocess_dataframe(
            nature_revx_df, DATE_TABLE, NREVX_DD
        )
        save_results(
            ready_to_publish_df, FILE_NAME, BQ_DATASET, ENVIRONMENT, PROJECT_ID
        )

        return f"{FILE_NAME} metrics calculated and saved successfully.", 200

    except Exception as e:
        logging.exception("Error during metrics calculation")
        raise


if __name__ == "__main__":
    # For local testing; in Cloud Functions, `main` is used as the entrypoint.
    main(request=None)
