from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from utils.gcp_tools import last_day_of_month, write_df_to_bq


def test_last_day_of_month():
    # Call the function.
    last_day_str = last_day_of_month()

    # Check the returned value is a string of format YYYYMMDD.
    assert isinstance(last_day_str, str)
    assert len(last_day_str) == 8


# A fixture for a sample DataFrame.
@pytest.fixture
def sample_df():
    return pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})


def test_write_df_to_bq_success(mocker, sample_df):
    # Mock pandas_gbq.to_gbq
    mock_to_gbq = mocker.patch("utils.gcp_tools.pandas_gbq.to_gbq")

    # Patch get_bq_client() directly to avoid credentials logic
    mock_client = MagicMock()
    mock_client.project = "my_project"
    mocker.patch("utils.gcp_tools.get_bq_client", return_value=mock_client)

    dest_table_id = "my_project.my_dataset.my_table"

    # Call the function
    write_df_to_bq(sample_df, dest_table_id)

    # Assert correct call
    mock_to_gbq.assert_called_once_with(
        sample_df,
        dest_table_id,
        project_id="my_project",
        if_exists="replace",
    )
