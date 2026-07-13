import pandas as pd
import pytest
from src.preprocessing import load_config, resolve_data_path



@pytest.fixture
def config():
    return load_config("configs/params.yaml")


@pytest.fixture
def df(config):
    data_path = resolve_data_path(config["data"]["path"])
    return pd.read_csv(data_path)


def test_expected_columns_present(df, config):
    required = config["data"]["required_columns"]
    for col in required:
        assert col in df.columns, f"Missing expected column: {col}"


def test_target_variable_within_expected_range(df, config):
    target_col = config["data"]["target_column"]

    # note: this test runs on the raw CSV, before bucketing happens in train.py,
    # so it should still check the raw 1-10 range here
    assert df[target_col].min() >= 1, f"{target_col} has values below 1"
    assert df[target_col].max() <= 10, f"{target_col} has values above 10"


def test_numeric_features_within_expected_ranges(df):
    range_checks = {
        "age": (0, 120),
        "daily_screen_time_hours": (0, 24),
        "phone_usage_before_sleep_minutes": (0, 1440),
        "sleep_duration_hours": (0, 24),
        "sleep_quality_score": (0, 100),
        "caffeine_intake_cups": (0, 20),
        "physical_activity_minutes": (0, 1440),
        "notifications_received_per_day": (0, 1000),
        "mental_fatigue_score": (0, 100),
    }

    for col, (min_val, max_val) in range_checks.items():
        col_data = df[col].dropna()
        assert col_data.min() >= min_val, f"{col} has values below {min_val}"
        assert col_data.max() <= max_val, f"{col} has values above {max_val}"