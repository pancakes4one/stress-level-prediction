from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "production"
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_split(filename: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename)


reference = load_split("reference.csv")
month1 = load_split("month1.csv")
month2 = load_split("month2.csv")
month3 = load_split("month3.csv")

batches = [
    ("Month 1 (expected: no drift)", month1, "drift_month1.html"),
    ("Month 2 (expected: moderate drift)", month2, "drift_month2.html"),
    ("Month 3 (expected: significant drift)", month3, "drift_month3.html"),
]

for label, current_data, output_name in batches:
    print("=" * 60)
    print(f"DRIFT REPORT: {label}")
    print("=" * 60)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current_data)
    output_path = REPORTS_DIR / output_name
    report.save_html(str(output_path))
    print(f"Report saved to {output_path}")

print("\nOpen the HTML files in your browser to explore the reports.")