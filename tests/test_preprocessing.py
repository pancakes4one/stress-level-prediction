import pandas as pd
import numpy as np
import pytest
from src.preprocessing import (
    validate_dataframe,
    simulate_missing_values,
    handle_missing_values,
    encode_categoricals,
)

# small fake dataset to test with
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40],
        "gender": ["male", "female", "female", "male"],
        "sleep_duration_hours": [7.0, 6.5, 8.0, 5.5],
    })


def test_fills_missing_values_with_mean(sample_df):
    df_with_nan = sample_df.copy()
    df_with_nan.loc[0, "sleep_duration_hours"] = np.nan

    result = handle_missing_values(df_with_nan, strategy="mean")

    assert result["sleep_duration_hours"].isna().sum() == 0


def test_drop_strategy_removes_rows_with_nan(sample_df):
    df_with_nan = sample_df.copy()
    df_with_nan.loc[0, "sleep_duration_hours"] = np.nan

    result = handle_missing_values(df_with_nan, strategy="drop")

    assert len(result) == 3


def test_encode_categoricals_turns_gender_into_numbers(sample_df):
    result = encode_categoricals(sample_df, columns=["gender"])

    assert "gender" not in result.columns

    new_cols = [c for c in result.columns if c.startswith("gender_")]
    assert len(new_cols) > 0


def test_encoded_column_is_actually_numeric(sample_df):
    result = encode_categoricals(sample_df, columns=["gender"])
    new_col = [c for c in result.columns if c.startswith("gender_")][0]

    assert pd.api.types.is_numeric_dtype(result[new_col])


def test_handle_missing_values_does_not_change_original_df(sample_df):
    df_with_nan = sample_df.copy()
    df_with_nan.loc[0, "sleep_duration_hours"] = np.nan

    handle_missing_values(df_with_nan, strategy="mean")

    assert df_with_nan["sleep_duration_hours"].isna().sum() == 1


def test_simulate_missing_values_does_not_change_original_df(sample_df):
    original = sample_df.copy()

    simulate_missing_values(sample_df, columns=["age"], rate=0.5)

    pd.testing.assert_frame_equal(sample_df, original)


def test_validate_dataframe_errors_if_not_a_dataframe():
    with pytest.raises(TypeError):
        validate_dataframe([1, 2, 3], required_columns=["age"])


def test_validate_dataframe_errors_if_column_missing(sample_df):
    with pytest.raises(ValueError):
        validate_dataframe(sample_df, required_columns=["age", "nonexistent_column"])