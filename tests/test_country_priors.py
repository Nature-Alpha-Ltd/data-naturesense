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

    # def test_large_sample_size(self):
    #     """Test with sample size larger than k"""
    #     result = compute_posterior(
    #         evidences=self.company_evidences,
    #         priors=self.weighted_priors,
    #         sample_size=15,
    #         k=10
    #     )
    #     # When sample_size > k, result should equal evidence
    #     for i in range(len(self.company_evidences.columns)):
    #         self.assertAlmostEqual(result.iloc[i], self.company_evidences.iloc[0, i])

    # def test_different_evidence_prior_combinations(self):
    #     """Test with different combinations of evidence and prior values"""
    #     test_evidences = pd.DataFrame({"test": [0.0, 0.5, 1.0]}, index=[0, 1, 2])
    #     test_priors = pd.DataFrame({"test": [0.0, 0.5, 1.0]}, index=[0, 1, 2])

    #     result = compute_posterior(
    #         evidences=test_evidences,
    #         priors=test_priors,
    #         sample_size=5,
    #         k=10
    #     )

    #     # Test different combinations
    #     self.assertAlmostEqual(result.iloc[0], 0.0)  # 0.0 evidence, 0.0 prior
    #     self.assertAlmostEqual(result.iloc[1], 0.5)  # 0.5 evidence, 0.5 prior
    #     self.assertAlmostEqual(result.iloc[2], 1.0)  # 1.0 evidence, 1.0 prior


if __name__ == "__main__":
    unittest.main()
