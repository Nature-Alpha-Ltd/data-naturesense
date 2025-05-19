import pandas as pd

from utils.nrevx_engine import nature_materiality_scores


def test_nature_materiality_scores():
    impact_weights, dependency_weights = nature_materiality_scores()

    # Check that impact_weights is a DataFrame with the expected columns
    assert isinstance(impact_weights, pd.DataFrame)
    for col in [
        "impact_biodiversity_disturbances_metric",
        "impact_land_use_change_metric",
    ]:
        assert col in impact_weights.columns

    # Check dependency_weights is a DataFrame and has one row per dependency category
    assert isinstance(dependency_weights, pd.DataFrame)
    expected_categories = {"provisioning", "regulating", "supporting", "cultural"}
    assert set(dependency_weights.index) == expected_categories
