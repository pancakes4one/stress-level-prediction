# Stress Level Prediction MLOps Project

This project builds an end-to-end machine learning pipeline for predicting stress level from behavioral and lifestyle features. The model is intentionally simple, while the focus is on production-ready MLOps practices such as data versioning, experiment tracking, automated testing, CI/CD, and drift monitoring.

## Project Goal

The goal is to classify a person’s stress level into one of three categories:

- Low
- Moderate
- High

The target values in the dataset are originally scored from 1 to 10. For modeling, these scores are grouped into the three classes above:

- 1–3: low stress
- 4–7: moderate stress
- 8–10: high stress

## What This Project Includes

- Data preprocessing and feature engineering
- Model training with scikit-learn
- MLflow experiment tracking
- DVC-based data versioning
- Automated pytest test suite
- GitHub Actions workflow for CI/CD
- Drift monitoring with Evidently

## Repository Structure

- [src](src): training, preprocessing, evaluation, and experiment comparison code
- [configs](configs): YAML configuration for preprocessing and training
- [tests](tests): unit, data validation, and model validation tests
- [data](data): raw and processed datasets tracked with DVC
- [reports](reports): drift monitoring scripts and generated reports
- [mlruns](mlruns): local MLflow tracking artifacts
- [.github/workflows](.github/workflows): CI/CD pipeline definition

## Dataset

The project uses the dataset stored in [data/raw/sleep_stress.csv](data/raw/sleep_stress.csv). It contains a mix of numeric and categorical features such as age, occupation, sleep duration, screen time, caffeine intake, and stress level.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Pull the DVC-tracked data:

   ```bash
   dvc pull
   ```

## Configuration

Training and preprocessing settings are stored in [configs/params.yaml](configs/params.yaml). This includes:

- data paths
- target column
- missing value settings
- categorical feature columns
- model hyperparameters
- minimum accuracy threshold

## Training

Run the training pipeline with:

```bash
python src/train.py
```

This script will:

- load the dataset
- simulate and handle missing values
- encode categorical variables
- train a RandomForestClassifier
- log metrics and artifacts to MLflow
- fail if the model accuracy drops below the configured threshold

## Experiment Tracking

Training runs are tracked with MLflow. To view experiments locally, run:

```bash
mlflow ui
```

You can also compare runs programmatically with:

```bash
python src/compare_experiments.py
```

## Testing

Run the full test suite with:

```bash
pytest tests/ -v
```

The tests cover:

- preprocessing behavior
- dataset validation
- model output and minimum performance expectations

## Drift Monitoring

Drift monitoring is implemented in [reports/monitor_drift.py](reports/monitor_drift.py). It compares a reference dataset with a simulated production batch and produces an HTML report in [reports](reports).

To run it:

```bash
python reports/monitor_drift.py
```

A written analysis of the monitoring results is available in [MONITORING.md](MONITORING.md).

## CI/CD

The GitHub Actions workflow in [.github/workflows/ml-pipeline.yml](.github/workflows/ml-pipeline.yml) automatically runs the test suite and training job on pushes and pull requests to the main branch.

## Notes

This repository demonstrates a practical MLOps workflow rather than focusing only on model accuracy. The emphasis is on reproducibility, automation, and reliable deployment preparation.

