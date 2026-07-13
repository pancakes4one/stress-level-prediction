from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "sleep_stress.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "production"


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load and clean the raw stress dataset."""
    df = pd.read_csv(path)

    if "user_id" in df.columns:
        df = df.drop(columns=["user_id"])

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    categorical_columns = df.select_dtypes(exclude=[np.number]).columns
    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    return df


def split_dataset(df: pd.DataFrame, reference_ratio: float = 0.6):
    """Split the data into a reference set and three production batches."""
    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_index = int(len(shuffled) * reference_ratio)

    reference = shuffled.iloc[:split_index].copy()
    remaining = shuffled.iloc[split_index:].copy()

    batch_size = len(remaining) // 3
    month1 = remaining.iloc[:batch_size].copy()
    month2 = remaining.iloc[batch_size : batch_size * 2].copy()
    month3 = remaining.iloc[batch_size * 2 :].copy()

    return reference, month1, month2, month3


def introduce_drift(month2: pd.DataFrame, month3: pd.DataFrame):
    """Simulate moderate drift in month 2 and stronger drift in month 3."""
    month2_copy = month2.copy()
    month3_copy = month3.copy()

    month2_copy["daily_screen_time_hours"] = np.clip(
        month2_copy["daily_screen_time_hours"] + np.random.normal(1.0, 0.3, len(month2_copy)),
        0,
        24,
    )
    month2_copy["phone_usage_before_sleep_minutes"] = np.clip(
        month2_copy["phone_usage_before_sleep_minutes"] + np.random.normal(20, 8, len(month2_copy)),
        0,
        180,
    )
    month2_copy["sleep_duration_hours"] = np.clip(
        month2_copy["sleep_duration_hours"] - np.random.normal(0.3, 0.1, len(month2_copy)),
        3,
        10,
    )
    month2_copy["stress_level"] = np.clip(
        month2_copy["stress_level"] + np.random.normal(0.8, 0.2, len(month2_copy)),
        1,
        10,
    )

    month3_copy["daily_screen_time_hours"] = np.clip(
        month3_copy["daily_screen_time_hours"] + np.random.normal(2.2, 0.4, len(month3_copy)),
        0,
        24,
    )
    month3_copy["phone_usage_before_sleep_minutes"] = np.clip(
        month3_copy["phone_usage_before_sleep_minutes"] + np.random.normal(40, 10, len(month3_copy)),
        0,
        180,
    )
    month3_copy["sleep_duration_hours"] = np.clip(
        month3_copy["sleep_duration_hours"] - np.random.normal(0.7, 0.15, len(month3_copy)),
        3,
        10,
    )
    month3_copy["sleep_quality_score"] = np.clip(
        month3_copy["sleep_quality_score"] - np.random.normal(0.6, 0.2, len(month3_copy)),
        1,
        10,
    )
    month3_copy["caffeine_intake_cups"] = np.clip(
        month3_copy["caffeine_intake_cups"] + np.random.normal(1.0, 0.4, len(month3_copy)),
        0,
        10,
    )
    month3_copy["notifications_received_per_day"] = np.clip(
        month3_copy["notifications_received_per_day"] + np.random.normal(45, 12, len(month3_copy)),
        0,
        500,
    )
    month3_copy["mental_fatigue_score"] = np.clip(
        month3_copy["mental_fatigue_score"] + np.random.normal(1.0, 0.2, len(month3_copy)),
        0,
        10,
    )

    if "occupation" in month3_copy.columns:
        shift_mask = np.random.rand(len(month3_copy)) < 0.2
        month3_copy.loc[shift_mask, "occupation"] = "student"

    return month2_copy, month3_copy


def save_splits(reference: pd.DataFrame, month1: pd.DataFrame, month2: pd.DataFrame, month3: pd.DataFrame, output_dir: Path = OUTPUT_DIR):
    """Save the reference and production batches to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    reference.to_csv(output_dir / "reference.csv", index=False)
    month1.to_csv(output_dir / "month1.csv", index=False)
    month2.to_csv(output_dir / "month2.csv", index=False)
    month3.to_csv(output_dir / "month3.csv", index=False)


def main():
    df = load_dataset()
    reference, month1, month2, month3 = split_dataset(df)
    month2, month3 = introduce_drift(month2, month3)
    save_splits(reference, month1, month2, month3)

    print(f"Loaded {len(df)} rows from {RAW_DATA_PATH}")
    print(f"Reference rows: {len(reference)}")
    print(f"Month 1 rows: {len(month1)}")
    print(f"Month 2 rows: {len(month2)}")
    print(f"Month 3 rows: {len(month3)}")
    print(f"Files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
