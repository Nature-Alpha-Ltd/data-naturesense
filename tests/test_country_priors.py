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
    def test_basic_posterior(self):
        """Test basic posterior computation with numeric inputs"""
        company_evidences = pd.Series(
            {
                "sensitive_locations": 0.895,
                "biodiversity_importance": 0.261,
                "high_ecosystem_integrity": 0.888,
            }
        )
        weighted_priors_df = pd.DataFrame(
            {
                "sensitive_locations": [0.567],
                "biodiversity_importance": [0.409],
                "high_ecosystem_integrity": [0.750],
            },
            index=[0],
        )
        result = compute_posterior(
            evidences=company_evidences, priors=weighted_priors_df, sample_size=5, k=10
        )
        self.assertAlmostEqual(result.iloc[0], 0.895 * (5 / 10) + 0.567 * (1 - 5 / 10))
        self.assertAlmostEqual(result.iloc[1], 0.261 * (5 / 10) + 0.409 * (1 - 5 / 10))
        self.assertAlmostEqual(result.iloc[2], 0.888 * (5 / 10) + 0.750 * (1 - 5 / 10))


if __name__ == "__main__":
    unittest.main()
