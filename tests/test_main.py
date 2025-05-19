import numpy as np
import pandas as pd
import pytest

from utils.nrevx_engine import nature_risk_calculation


def test_nature_risk_calculation():
    # Create a sample DataFrame with numeric data
    data = {"col1": [1, 2, 3, 4, 5], "col2": [2, 3, 4, 5, 6]}
    df = pd.DataFrame(data)

    # Call the function and get the results (assume probability scores are returned)
    # Here we are not using custom_mean or custom_std
    results = nature_risk_calculation(df, ["col1", "col2"], zscore_scale=False)

    # Check the output is a numpy array of the right shape
    assert isinstance(results, np.ndarray)
    assert results.shape[0] == df.shape[0]

    # Check that probability scores are between 0 and 1
    assert results.min() >= 0 and results.max() <= 1
