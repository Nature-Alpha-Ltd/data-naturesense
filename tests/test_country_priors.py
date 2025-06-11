"""
python -m unittest discover tests 
"""

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
        # Create a smaller test dataset with just a few columns
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

    def test_sample_size_greater_than_k(self):  # Condition 1
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

    def test_k_equals_sample_size(self):  # Condition 1
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

    def test_zero_sample_size(self):  # Condition 3
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
                "total_company_locations": 0,
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
                "total_company_locations": 15,
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
                "total_company_locations": 15,
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
                "total_company_locations": 40,
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
                "total_company_locations": 0,
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


if __name__ == "__main__":
    unittest.main()
