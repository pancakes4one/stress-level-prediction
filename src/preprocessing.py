import pandas as pd
import numpy as np
import yaml


def load_config(config_path):
    # opens the yaml file and turns it into a python dictionary
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def validate_dataframe(df, required_columns):
    # check it's actually a dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    # check that all the columns we need are actually there
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    return True


def simulate_missing_values(df, columns, rate, seed=42):
    # copy so we don't mess up the original dataframe
    df_copy = df.copy()

    np.random.seed(seed)

    for col in columns:
        # pick random rows to turn into NaN
        n_rows = len(df_copy)
        random_mask = np.random.rand(n_rows) < rate
        df_copy.loc[random_mask, col] = np.nan

    return df_copy


def handle_missing_values(df, strategy):
    df_copy = df.copy()

    if strategy == "drop":
        df_copy = df_copy.dropna()
        return df_copy

    for col in df_copy.columns:
        # skip columns that aren't numbers, can't take a mean/median of text
        if not pd.api.types.is_numeric_dtype(df_copy[col]):
            continue

        if strategy == "mean":
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        elif strategy == "median":
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    return df_copy


def encode_categoricals(df, columns):
    df_copy = df.copy()

    # one-hot encoding: turns each category into its own 0/1 column
    df_copy = pd.get_dummies(df_copy, columns=columns, drop_first=True)

    return df_copy