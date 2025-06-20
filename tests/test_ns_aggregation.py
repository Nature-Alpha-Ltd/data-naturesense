"""
python -m unittest discover tests 

Test NatureSense company aggregation, including the following conditions:

1. if material_assets_count >= threshold (e.g., 10 assets) 
  then company's sensitive_locations is calc based on ALD scores only
 
2. if material_assets_count == 0 & estimated_material_assets_count == 0
  then company's sensitive_locations is NULL
 
3. if ALD scores IS NULL 
  then company's sensitive_locations is calc based on CountryPriors only
 
4. if estimated_material_assets_count >= threshold
  then company's sensitive_locations is calc based on both ALD scores and CountryPriors, using ratio material_assets_count/threshold
  (e.g., material_assets_count 5 and threshold 10, then 50% ALD scores, 50% CountryPriors)
 
5. if material_assets_count is between 1-9 & estimated_material_assets_count == 0
  then company's sensitive_locations is calc based on both ALD scores and constant all companies avg, using ratio material_assets_count/threshold 
  (e.g., material_assets_count 4 and threshold 10, then 40% ALD scores, 60% constant)
 
6. if material_assets_count >= estimated_material_assets_count
  then company's sensitive_locations is calc based on ALD scores only
 
7. if estimated_material_assets_count < threshold
  then company's sensitive_locations is calc based on both ALD scores and CountryPriors, using ratio material_assets_count/estimated_material_assets_count 
  (e.g., material_assets_count 4 and estimated_material_assets_count 5, then 80% ALD scores, 20% CountryPriors)
"""

import logging
import unittest

import numpy as np
import pandas as pd

from main import (
    calculate_country_prior,
    calculate_effective_k,
    compute_posterior,
    no_guestimates_adjust_priors_and_k,
    process_company_evidence,
)


class TestPosteriorComputation(unittest.TestCase):
    def setUp(self):
        """Set up test data that can be used by multiple tests"""
        self.company_evidences = pd.DataFrame(
            {
                "sensitive_locations": [0.895],
                "biodiversity_importance": [0.261],
                "high_ecosystem_integrity": [0.888],
                "decline_in_ecosystem_integrity": [0.516],
                "physical_water_risk": [0.932],
                "ecosystem_services_provision_importance": [0.449],
            },
            index=[0],
        )
        self.weighted_priors = pd.DataFrame(
            {
                "sensitive_locations": [0.567],
                "biodiversity_importance": [0.409],
                "high_ecosystem_integrity": [0.750],
                "decline_in_ecosystem_integrity": [0.101],
                "physical_water_risk": [0.543],
                "ecosystem_services_provision_importance": [0.567],
            },
            index=[0],
        )

    def test_basic_posterior(self):
        """Test basic posterior computation with a single column"""
        material_assets_count = 5
        effective_k = 10
        result = compute_posterior(
            evidences=self.company_evidences,
            priors=self.weighted_priors,
            sample_size=material_assets_count,
            k=effective_k,
        )
        expected = self.company_evidences.iloc[0, 0] * (
            material_assets_count / effective_k
        ) + self.weighted_priors.iloc[0, 0] * (1 - material_assets_count / effective_k)
        self.assertAlmostEqual(result.iloc[0], expected)

    def test_k_adjustment_for_low_sample_size(self):
        """Test sample size < k, k is adjusted to sample_size"""
        result = compute_posterior(
            evidences=self.company_evidences,
            priors=self.weighted_priors,
            sample_size=4,
            k=5,
        )
        expected = self.company_evidences.iloc[0, 0] * (
            4 / 5
        ) + self.weighted_priors.iloc[0, 0] * (1 - 4 / 5)
        self.assertAlmostEqual(result.iloc[0], expected)

    ### Condition 1
    def test_sample_size_greater_than_k(self):
        """Test when sample size is greater than k"""
        result = compute_posterior(
            evidences=self.company_evidences,
            priors=self.weighted_priors,
            sample_size=15,
            k=10,
        )
        # When sample_size > k, result should equal evidence
        for i in range(len(self.company_evidences.columns)):
            self.assertAlmostEqual(result.iloc[i], self.company_evidences.iloc[0, i])

    ### Condition 1
    def test_k_equals_sample_size(self):
        """Test when k equals sample size"""
        result = compute_posterior(
            evidences=self.company_evidences,
            priors=self.weighted_priors,
            sample_size=10,
            k=10,
        )
        # When sample_size = k, result should equal evidence
        for i in range(len(self.company_evidences.columns)):
            self.assertAlmostEqual(result.iloc[i], self.company_evidences.iloc[0, i])

    ### Condition 3
    def test_zero_sample_size(self):
        """Test with zero sample size"""
        result = compute_posterior(
            evidences=self.company_evidences,
            priors=self.weighted_priors,
            sample_size=0,
            k=10,
        )
        # When sample_size is 0, result should equal prior
        for i in range(len(self.company_evidences.columns)):
            self.assertAlmostEqual(result.iloc[i], self.weighted_priors.iloc[0, i])

    def test_zero_k(self):
        """Test with zero k"""
        result = compute_posterior(
            evidences=self.company_evidences,
            priors=self.weighted_priors,
            sample_size=5,
            k=0,
        )
        # When sample_size > k, result should equal evidence
        for i in range(len(self.company_evidences.columns)):
            self.assertAlmostEqual(result.iloc[i], self.company_evidences.iloc[0, i])

    def test_large_sample_size(self):
        """Test behavior with large sample size"""
        result = compute_posterior(
            evidences=self.company_evidences,
            priors=self.weighted_priors,
            sample_size=50,
            k=10,
        )
        # When sample_size > k, result should equal evidence
        for i in range(len(self.company_evidences.columns)):
            self.assertAlmostEqual(result.iloc[i], self.company_evidences.iloc[0, i])

    def test_zero_prior(self):
        """Test edge case where prior is zero"""
        company_evidences = pd.DataFrame(
            {"sensitive_locations": [0.895]},
            index=[0],
        )
        weighted_priors = pd.DataFrame(
            {"sensitive_locations": [0.0]},
            index=[0],
        )
        material_assets_count = 5
        effective_k = 10
        result = compute_posterior(
            evidences=company_evidences,
            priors=weighted_priors,
            sample_size=material_assets_count,
            k=effective_k,
        )
        expected = 0.895 * (material_assets_count / effective_k) + 0.0 * (
            1 - material_assets_count / effective_k
        )
        self.assertAlmostEqual(result.iloc[0], expected)

    def test_zero_evidence(self):
        """Test edge case where evidence is zero"""
        company_evidences = pd.DataFrame(
            {"sensitive_locations": [0.0]},
            index=[0],
        )
        weighted_priors = pd.DataFrame(
            {"sensitive_locations": [0.567]},
            index=[0],
        )
        material_assets_count = 5
        effective_k = 10
        result = compute_posterior(
            evidences=company_evidences,
            priors=weighted_priors,
            sample_size=material_assets_count,
            k=effective_k,
        )
        expected = 0.0 * (material_assets_count / effective_k) + 0.567 * (
            1 - material_assets_count / effective_k
        )
        self.assertAlmostEqual(result.iloc[0], expected)


class TestCountryPriorCalculation(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.country_priors = pd.DataFrame(
            {"country_code": ["USA", "GBR", "JPN"], "emissions": [0.4, 0.3, 0.5]}
        )
        self.country_codes = ["USA", "GBR", "JPN"]

    def test_zero_locations(self):
        """Test when company has zero material asset locations"""
        company_row = pd.Series(
            {
                "na_entity_id": "TEST1",
                "USA": 0,
                "GBR": 0,
                "JPN": 0,
                "estimated_material_assets_count": 0,
            }
        )
        prior = calculate_country_prior(
            company_row=company_row,
            country_priors=self.country_priors,
            evidence_columns=["emissions"],
            country_codes=self.country_codes,
            entity_id="TEST1",
        )
        self.assertEqual(prior, [None])  # Now expecting [None] instead of None

    def test_missing_countries(self):
        """Test when company has assets in countries missing from priors"""
        company_row = pd.Series(
            {
                "na_entity_id": "TEST1",
                "USA": 10,
                "XXX": 5,  # Missing country
                "estimated_material_assets_count": 15,
            }
        )
        prior = calculate_country_prior(
            company_row=company_row,
            country_priors=self.country_priors,
            evidence_columns=["emissions"],
            country_codes=self.country_codes,
            entity_id="TEST1",
        )
        self.assertEqual(len(prior), 1)  # Should be a list with one element
        self.assertAlmostEqual(prior[0], 0.4)  # Only USA should be considered

    def test_multiple_evidence_columns(self):
        """Test calculation with multiple evidence columns"""
        # Update test data to include multiple columns
        self.country_priors = pd.DataFrame(
            {
                "country_code": ["USA", "GBR", "JPN"],
                "emissions": [0.4, 0.3, 0.5],
                "water_risk": [0.6, 0.2, 0.3],
            }
        )

        company_row = pd.Series(
            {
                "na_entity_id": "TEST1",
                "USA": 10,
                "GBR": 5,
                "JPN": 0,
                "estimated_material_assets_count": 15,
            }
        )

        priors = calculate_country_prior(
            company_row=company_row,
            country_priors=self.country_priors,
            evidence_columns=["emissions", "water_risk"],
            country_codes=self.country_codes,
            entity_id="TEST1",
        )

        expected_emissions = (0.4 * 10 + 0.3 * 5) / 15  # weighted avg for emissions
        expected_water = (0.6 * 10 + 0.2 * 5) / 15  # weighted avg for water_risk

        self.assertEqual(len(priors), 2)
        self.assertAlmostEqual(priors[0], expected_emissions)
        self.assertAlmostEqual(priors[1], expected_water)

    def test_vectorized_country_prior(self):
        """Test that vectorized country prior calculation works correctly for both single and multiple columns"""
        # Set up test data
        country_priors = pd.DataFrame(
            {
                "country_code": ["USA", "GBR", "JPN", "DEU"],
                "emissions": [0.4, 0.3, 0.5, 0.6],
                "water_risk": [0.6, 0.2, 0.3, 0.4],
                "biodiversity": [0.3, 0.5, 0.4, 0.2],
            }
        )

        company_row = pd.Series(
            {
                "na_entity_id": "TEST1",
                "USA": 20,  # 50% of assets
                "GBR": 10,  # 25% of assets
                "DEU": 10,  # 25% of assets
                "estimated_material_assets_count": 40,
            }
        )

        country_codes = ["USA", "GBR", "JPN", "DEU"]

        # Test single column
        single_col_result = calculate_country_prior(
            company_row=company_row,
            country_priors=country_priors,
            evidence_columns="emissions",
            country_codes=country_codes,
            entity_id="TEST1",
        )

        # Expected: (0.4 * 0.5) + (0.3 * 0.25) + (0.6 * 0.25)
        expected_emissions = 0.4 * (20 / 40) + 0.3 * (10 / 40) + 0.6 * (10 / 40)
        self.assertEqual(len(single_col_result), 1)
        self.assertAlmostEqual(single_col_result[0], expected_emissions)

        # Test multiple columns
        multi_col_result = calculate_country_prior(
            company_row=company_row,
            country_priors=country_priors,
            evidence_columns=["emissions", "water_risk", "biodiversity"],
            country_codes=country_codes,
            entity_id="TEST1",
        )

        # Calculate expected values for each column
        expected_water = 0.6 * (20 / 40) + 0.2 * (10 / 40) + 0.4 * (10 / 40)
        expected_biodiversity = 0.3 * (20 / 40) + 0.5 * (10 / 40) + 0.2 * (10 / 40)

        self.assertEqual(len(multi_col_result), 3)
        self.assertAlmostEqual(multi_col_result[0], expected_emissions)
        self.assertAlmostEqual(multi_col_result[1], expected_water)
        self.assertAlmostEqual(multi_col_result[2], expected_biodiversity)

        # Test zero weights case
        zero_weights_row = pd.Series(
            {
                "na_entity_id": "TEST2",
                "USA": 0,
                "GBR": 0,
                "DEU": 0,
                "estimated_material_assets_count": 0,
            }
        )

        zero_weights_result = calculate_country_prior(
            company_row=zero_weights_row,
            country_priors=country_priors,
            evidence_columns=["emissions", "water_risk"],
            country_codes=country_codes,
            entity_id="TEST2",
        )

        # Should return None for each column when weights sum to zero
        self.assertEqual(len(zero_weights_result), 2)
        self.assertIsNone(zero_weights_result[0])
        self.assertIsNone(zero_weights_result[1])


class TestProcessCompanyEvidence(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.company_data = pd.DataFrame(
            {
                "na_entity_id": ["TEST1", "TEST2"],
                "material_assets_count": [15, 5],
                "emissions": [0.8, 0.6],
            }
        )
        self.country_dist = pd.DataFrame(
            {
                "na_entity_id": ["TEST1", "TEST2"],
                "USA": [10, 0],
                "GBR": [5, 5],
                "JPN": [0, 0],
                "estimated_material_assets_count": [15, 5],
                "date_added": ["2025-03-17", "2025-03-17"],
            }
        )
        self.country_priors = pd.DataFrame(
            {
                "country_code": ["USA", "GBR", "JPN"],
                "emissions": [0.4, 0.3, 0.5],
            }
        )
        self.evidence_columns = ["emissions"]
        self.global_priors = [0.123, 0.456, 0.789]

    def test_full_processing(self):
        """Test complete processing pipeline"""
        result = process_company_evidence(
            company_data=self.company_data,
            country_dist=self.country_dist,
            country_priors=self.country_priors,
            evidence_columns=self.evidence_columns,
            global_priors=self.global_priors,
        )
        self.assertIn("emissions_posterior", result.columns)
        self.assertEqual(len(result), len(self.company_data))

    def test_missing_evidence_columns(self):
        """Test handling of missing evidence columns"""
        with self.assertRaises(ValueError) as context:
            process_company_evidence(
                company_data=self.company_data.copy(),
                country_dist=self.country_dist,
                country_priors=self.country_priors,
                evidence_columns=self.evidence_columns
                + ["water_usage"],  # water_usage is new
                global_priors=self.global_priors,
            )
        self.assertIn(
            "Missing required columns in company_data: {'water_usage'}",
            str(context.exception),
        )

    def test_proportional_k(self):
        """Test with proportional k"""
        result = process_company_evidence(
            company_data=self.company_data,
            country_dist=self.country_dist,
            country_priors=self.country_priors,
            evidence_columns=self.evidence_columns,
            global_priors=self.global_priors,
            k=0.5,
        )
        self.assertIn("emissions_posterior", result.columns)

    def test_process_company_evidence_with_effective_k(self):
        """Test process_company_evidence with various scenarios for effective k calculation and prior adjustment"""
        company_data = pd.DataFrame(
            {
                "na_entity_id": ["TEST1", "TEST2", "TEST3"],
                "material_assets_count": [60, 0, 0],  # Different asset scenarios
                "emissions": [0.8, 0.7, 0.6],
            }
        )

        country_dist = pd.DataFrame(
            {
                "na_entity_id": ["TEST1", "TEST2", "TEST3"],
                "USA": [40, 0, 30],
                "GBR": [20, 0, 20],
                "estimated_material_assets_count": [60, 0, 50],
                "date_added": ["2025-03-17", "2025-03-17", "2025-03-17"],
            }
        )

        result = process_company_evidence(
            company_data=company_data,
            country_dist=country_dist,
            country_priors=self.country_priors,
            evidence_columns=self.evidence_columns,
            global_priors=self.global_priors,
            k=50,
        )

        # Test case: assets_count is larger than k
        expected_weight = 50 / 50
        expected_posterior = 0.8 * expected_weight + (0.4 * 40 / 60 + 0.3 * 20 / 60) * (
            1 - expected_weight
        )
        self.assertAlmostEqual(
            result.loc[result["na_entity_id"] == "TEST1", "emissions_posterior"].iloc[
                0
            ],
            expected_posterior,
        )

        # Test case: sum of locations and assets_count are 0
        self.assertTrue(
            pd.isna(
                result.loc[
                    result["na_entity_id"] == "TEST2", "emissions_posterior"
                ].iloc[0]
            )
        )

        # Test case: assets_count is 0, rely on country_dist
        expected_posterior = 0.4 * 30 / 50 + 0.3 * 20 / 50
        self.assertAlmostEqual(
            result.loc[result["na_entity_id"] == "TEST3", "emissions_posterior"].iloc[
                0
            ],
            expected_posterior,
        )

    ### Condition 2
    def test_null_in_company_data(self):
        """Test material_assets_count = 0 and estimated_material_assets_count = 0"""
        company_data = pd.DataFrame(
            {
                "na_entity_id": ["TEST1"],
                "material_assets_count": [0],
                "emissions": [None],
            }
        )
        country_dist = pd.DataFrame(
            {
                "na_entity_id": ["TEST1"],
                "USA": [0],
                "GBR": [0],
                "JPN": [0],
                "estimated_material_assets_count": [0],
                "date_added": ["2025-03-17"],
            }
        )

        result = process_company_evidence(
            company_data=company_data,
            country_dist=country_dist,
            country_priors=self.country_priors,
            evidence_columns=self.evidence_columns,
            global_priors=self.global_priors,
        )

        # Should return NaN and this should be handle afterwards
        self.assertTrue(
            pd.isna(
                result.loc[
                    result["na_entity_id"] == "TEST1", "emissions_posterior"
                ].iloc[0]
            )
        )

    ### Condition 4
    def test_company_prior_case_4(self):
        """Test case 4: material_assets_count < k and estimated_material_assets_count > k"""
        company_data = pd.DataFrame(
            {
                "na_entity_id": ["TEST4"],
                "material_assets_count": [5],  # Below k
                "emissions": [0.9],
            }
        )

        country_dist = pd.DataFrame(
            {
                "na_entity_id": ["TEST4"],
                "USA": [10],
                "GBR": [5],
                "estimated_material_assets_count": [15],  # Above k
                "date_added": ["2025-03-17"],
            }
        )

        result = process_company_evidence(
            company_data=company_data,
            country_dist=country_dist,
            country_priors=self.country_priors,
            evidence_columns=self.evidence_columns,
            global_priors=self.global_priors,
            k=10,
        )

        expected_weight = 5 / 10  # material_assets_count / effektive_k
        expected_posterior = round(
            (0.9 * expected_weight)
            + ((0.4 * 10 / 15 + 0.3 * 5 / 15) * (1 - expected_weight)),
            3,
        )
        self.assertAlmostEqual(
            result.loc[result["na_entity_id"] == "TEST4", "emissions_posterior"].iloc[
                0
            ],
            expected_posterior,
        )

    ### Condition 5
    def test_company_prior_case_5(self):
        """Test case 5: assets_count < k and estimated_material_assets_count = 0"""
        company_data = pd.DataFrame(
            {
                "na_entity_id": ["TEST5"],
                "material_assets_count": [5],  # Below k
                "emissions": [0.9],
            }
        )

        country_dist = pd.DataFrame(
            {
                "na_entity_id": ["TEST5"],
                "USA": [0],
                "GBR": [0],
                "estimated_material_assets_count": [0],  # No country distribution
                "date_added": ["2025-03-17"],
            }
        )

        result = process_company_evidence(
            company_data=company_data,
            country_dist=country_dist,
            country_priors=self.country_priors,
            evidence_columns=self.evidence_columns,
            global_priors=self.global_priors,
            k=10,
        )

        # Should use default prior since there is no country distribution available
        expected_weight = 5 / 10  # material_assets_count / k
        expected_posterior = round(
            (0.9 * expected_weight) + (self.global_priors[0] * (1 - expected_weight)), 3
        )
        self.assertAlmostEqual(
            result.loc[result["na_entity_id"] == "TEST5", "emissions_posterior"].iloc[
                0
            ],
            expected_posterior,
        )

    def test_zero_estimated_material_assets_but_enough_material_assets(self):
        """
        Test case when a company has estimated_material_assets_count = 0 but material_assets_count > k
        """
        company_data = pd.DataFrame(
            {
                "na_entity_id": ["TEST1"],
                "material_assets_count": [15],  # More assets than minimum k
                "emissions": [0.8],
            }
        )

        country_dist = pd.DataFrame(
            {
                "na_entity_id": ["TEST1"],
                "USA": [0],
                "GBR": [0],
                "estimated_material_assets_count": [0],
                "date_added": ["2025-03-17"],
            }
        )

        result = process_company_evidence(
            company_data=company_data,
            country_dist=country_dist,
            country_priors=self.country_priors,
            evidence_columns=self.evidence_columns,
            global_priors=self.global_priors,
            k=10,
        )

        # Since material_assets_count > k, posterior should equal evidence
        self.assertAlmostEqual(
            result.loc[result["na_entity_id"] == "TEST1", "emissions_posterior"].iloc[
                0
            ],
            company_data["emissions"][0],
        )

    ### Condition 6
    def test_company_prior_case_6(self):
        """Test case 6: material_assets_count >= estimated_material_assets_count"""
        company_data = pd.DataFrame(
            {
                "na_entity_id": ["TEST6.1", "TEST6.2"],
                "material_assets_count": [9, 6],
                "emissions": [0.9, 0.8],
            }
        )

        country_dist = pd.DataFrame(
            {
                "na_entity_id": ["TEST6.1", "TEST6.2"],
                "USA": [4, 4],
                "GBR": [2, 2],
                "estimated_material_assets_count": [6, 6],
                "date_added": ["2025-03-17", "2025-03-17"],
            }
        )

        result = process_company_evidence(
            company_data=company_data,
            country_dist=country_dist,
            country_priors=self.country_priors,
            evidence_columns=self.evidence_columns,
            global_priors=self.global_priors,
            k=10,
        )

        # Since material_assets_count >= estimated_material_assets_count, posterior should equal evidence
        self.assertAlmostEqual(
            result.loc[result["na_entity_id"] == "TEST6.1", "emissions_posterior"].iloc[
                0
            ],
            company_data["emissions"][0],
        )
        self.assertAlmostEqual(
            result.loc[result["na_entity_id"] == "TEST6.2", "emissions_posterior"].iloc[
                0
            ],
            company_data["emissions"][1],
        )

    ### Condition 7
    def test_company_prior_case_7(self):
        """Test case 7: estimated_material_assets_count < k"""
        company_data = pd.DataFrame(
            {
                "na_entity_id": ["TEST7"],
                "material_assets_count": [5],  # Below estimated_material_assets_count
                "emissions": [0.9],
            }
        )

        country_dist = pd.DataFrame(
            {
                "na_entity_id": ["TEST7"],
                "USA": [6],
                "GBR": [2],
                "estimated_material_assets_count": [8],  # Below k=10
                "date_added": ["2025-03-17"],
            }
        )

        result = process_company_evidence(
            company_data=company_data,
            country_dist=country_dist,
            country_priors=self.country_priors,
            evidence_columns=self.evidence_columns,
            global_priors=self.global_priors,
            k=10,
        )

        # k should be adjusted to estimated_material_assets_count=8
        expected_weight = 5 / 8  # material_assets_count/estimated_material_assets_count
        expected_prior = 0.4 * 6 / 8 + 0.3 * 2 / 8  # Weighted country priors
        expected_posterior = round(
            0.9 * expected_weight + expected_prior * (1 - expected_weight), 3
        )
        self.assertAlmostEqual(
            result.loc[result["na_entity_id"] == "TEST7", "emissions_posterior"].iloc[
                0
            ],
            expected_posterior,
        )

    def test_missing_isin_in_country_dist(self):
        """
        Test when na_entity_id is completely missing from country_dist.
        Should default to using global_priors when material_assets_count < k and k is adjusted to 10.
        """
        # Setup test data with two companies
        company_data = pd.DataFrame(
            {
                "na_entity_id": ["TEST1", "TEST2"],
                "material_assets_count": [5, 15],  # One below 10, one above
                "emissions": [0.8, 0.7],
            }
        )

        # Country distribution missing both companies
        country_dist = pd.DataFrame(
            {
                "na_entity_id": ["TEST3"],  # Different id
                "USA": [40],
                "GBR": [20],
                "estimated_material_assets_count": [60],
                "date_added": ["2025-03-17"],
            }
        )

        result = process_company_evidence(
            company_data=company_data,
            country_dist=country_dist,
            country_priors=self.country_priors,
            evidence_columns=self.evidence_columns,
            global_priors=self.global_priors,
            k=10,
        )

        # Since material_assets_count < k, posterior should take global_priors into consideration
        expected_weight = 5 / 10
        expected_posterior = round(
            0.8 * expected_weight + self.global_priors[0] * (1 - expected_weight), 3
        )
        self.assertAlmostEqual(
            result.loc[result["na_entity_id"] == "TEST1", "emissions_posterior"].iloc[
                0
            ],
            expected_posterior,
        )

        # Since material_assets_count > k, posterior should equal evidence
        self.assertAlmostEqual(
            result.loc[result["na_entity_id"] == "TEST2", "emissions_posterior"].iloc[
                0
            ],
            company_data["emissions"][1],
        )

    def test_no_company_evidence_and_country_dist_are_available(self):
        """Test when no company_evidence and country_dist are available"""
        company_data = pd.DataFrame(
            {
                "na_entity_id": ["TEST1", "TEST2"],
                "material_assets_count": [5, None],  # No data for TEST2
                "emissions": [0.8, None],  # No data for TEST2
            }
        )

        country_dist = pd.DataFrame(
            {
                "na_entity_id": ["TEST1"],  # TEST2 is missing
                "USA": [5],
                "GBR": [5],
                "estimated_material_assets_count": [10],
                "date_added": ["2025-03-17"],
            }
        )

        # Capture the warning message
        with self.assertLogs(level="WARNING") as log:
            result = process_company_evidence(
                company_data=company_data,
                country_dist=country_dist,
                country_priors=self.country_priors,
                evidence_columns=self.evidence_columns,
                global_priors=self.global_priors,
                k=10,
            )

            # Verify warning was logged
            self.assertTrue(
                any(
                    "1 companies were not found in company country distribution data"
                    in msg
                    for msg in log.output
                )
            )

        # It should set all posterior columns to NaN
        missing_entity_row = result[result["na_entity_id"] == "TEST2"]
        self.assertTrue(missing_entity_row["emissions_posterior"].isna().all())

    def test_no_country_dist_is_available(self):
        """Test when no country_dist is available"""
        company_data = pd.DataFrame(
            {
                "na_entity_id": ["TEST1", "TEST2", "TEST3"],
                "material_assets_count": [10, 5, 15],
                "emissions": [0.7, 0.8, 0.9],
            }
        )

        country_dist = pd.DataFrame(
            {
                "na_entity_id": ["TEST1"],  # TEST2 and TEST3 are missing
                "USA": [5],
                "GBR": [5],
                "estimated_material_assets_count": [10],
                "date_added": ["2025-03-17"],
            }
        )

        # Capture the warning message
        with self.assertLogs(level="WARNING") as log:
            result = process_company_evidence(
                company_data=company_data,
                country_dist=country_dist,
                country_priors=self.country_priors,
                evidence_columns=self.evidence_columns,
                global_priors=self.global_priors,
                k=10,
            )

            # Verify warning was logged
            self.assertTrue(
                any(
                    "2 companies were not found in company country distribution data"
                    in msg
                    for msg in log.output
                )
            )

        # TEST2 should use global_priors since material_assets_count < k and no country_dist
        test2_row = result[result["na_entity_id"] == "TEST2"]
        expected_weight = 5 / 10
        expected_posterior = round(
            0.8 * expected_weight + self.global_priors[0] * (1 - expected_weight), 3
        )
        self.assertEqual(test2_row["emissions_posterior"].iloc[0], expected_posterior)

        # TEST3 should keep original values since material_assets_count > k
        test3_row = result[result["na_entity_id"] == "TEST3"]
        self.assertEqual(test3_row["emissions_posterior"].iloc[0], 0.9)


class TestAdjustPriorsAndK(unittest.TestCase):
    def test_calculate_effective_k(self):
        """Test the calculate_effective_k function"""
        # Test proportional k
        result = calculate_effective_k(
            k=0.5, estimated_material_assets_count=100, material_assets_count=40
        )
        self.assertEqual(result, 50)  # 0.5 * 100 = 50

        # Test when estimated_material_assets_count is less than k
        result = calculate_effective_k(
            k=50, estimated_material_assets_count=30, material_assets_count=25
        )
        self.assertEqual(
            result, 30
        )  # k should be reduced to match estimated_material_assets_count

        # Test when material_assets_count exceeds estimated_material_assets_count but is less than k
        result = calculate_effective_k(
            k=100, estimated_material_assets_count=40, material_assets_count=60
        )
        self.assertEqual(result, 60)  # k should match the higher material_assets_count

        # Test when material_assets_count exceeds both estimated_material_assets_count and k
        result = calculate_effective_k(
            k=50, estimated_material_assets_count=40, material_assets_count=60
        )
        self.assertEqual(
            result, 50
        )  # k should remain unchanged as it's less than material_assets_count

        # Test with proportional k and material_assets_count exceeding estimated_material_assets_count
        result = calculate_effective_k(
            k=0.5, estimated_material_assets_count=100, material_assets_count=150
        )
        self.assertEqual(
            result, 50
        )  # 0.5 * 150 = 75 (using updated total from material_assets_count)

    def test_adjust_priors_and_k(self):
        """Test the adjustment of priors and k"""
        # Test case 1: effective_k < k and prior is None
        # Should update both k and priors
        priors = [None, 0.3, None]
        effective_k = 5
        k = 10
        default_priors = [0.123, 0.456, 0.789]
        adjusted_priors, adjusted_k = no_guestimates_adjust_priors_and_k(
            priors, effective_k, k, default_priors
        )
        self.assertEqual(adjusted_k, 10)  # Should be set to k
        self.assertEqual(adjusted_priors, [0.123, 0.3, 0.789])  # None values replaced

        # Test case 2: effective_k > k and prior has None values
        # Should keep effective_k and priors unchanged
        priors = [None, 0.3, None]
        effective_k = 15
        k = 10
        adjusted_priors, adjusted_k = no_guestimates_adjust_priors_and_k(
            priors, effective_k, k, default_priors
        )
        self.assertEqual(adjusted_k, 15)  # Should remain unchanged
        self.assertEqual(adjusted_priors, [None, 0.3, None])  # Should remain unchanged

        # Test case 3: effective_k < k but no None values
        # Should keep priors but update k
        priors = [0.4, 0.3, 0.5]
        effective_k = 5
        k = 10
        adjusted_priors, adjusted_k = no_guestimates_adjust_priors_and_k(
            priors, effective_k, k, default_priors
        )
        self.assertEqual(adjusted_k, 5)  # Should remain unchanged
        self.assertEqual(adjusted_priors, [0.4, 0.3, 0.5])  # Should remain unchanged

        # Test case 4: weighted_priors is None (not a list)
        # Should create new list with default value and update k
        priors = [None, None, None]
        effective_k = 5
        k = 10
        adjusted_priors, adjusted_k = no_guestimates_adjust_priors_and_k(
            priors, effective_k, k, default_priors
        )
        self.assertEqual(adjusted_k, 10)  # Should be set to k
        self.assertEqual(
            adjusted_priors, default_priors
        )  # Should be list with default value


if __name__ == "__main__":
    unittest.main()
