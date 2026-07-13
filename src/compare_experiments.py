from src.preprocessing import load_config
from src.train import train

base_config = load_config("configs/params.yaml")

experiments = [
    {"model": {"n_estimators": 50, "max_depth": 5}},
    {"model": {"n_estimators": 100, "max_depth": 10}},
    {"model": {"n_estimators": 200, "max_depth": 10}},
    {"model": {"n_estimators": 300, "max_depth": 15}},
    {"model": {"n_estimators": 500, "max_depth": 20}},
]

print(f"Running {len(experiments)} experiments...\n")

for i, overrides in enumerate(experiments):
    print(f"\n{'=' * 60}")
    print(f"Experiment {i + 1}/{len(experiments)}")
    print(f"{'=' * 60}")

    # start from the base config and layer the override on top
    current_config = base_config.copy()
    current_config["model"] = overrides["model"]

    try:
        run_id = train(current_config)
        print(f"Completed. Run ID: {run_id}")
    except Exception as e:
        print(f"Failed: {e}")

print("\nAll experiments complete. Run 'mlflow ui' to compare results.")