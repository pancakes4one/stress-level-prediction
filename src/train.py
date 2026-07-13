import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import (
    load_config,
    validate_dataframe,
    simulate_missing_values,
    handle_missing_values,
    bucket_stress_level,
    encode_categoricals,
)
from src.evaluate import evaluate_model


def train():
    
    config = load_config("configs/params.yaml")

    df = pd.read_csv(config["data"]["path"])

    # dataset is clean, simulate missing values
    df = simulate_missing_values(
        df,
        columns=config["preprocessing"]["missing_columns"],
        rate=config["preprocessing"]["missing_rate"],
    )

    validate_dataframe(df, config["data"]["required_columns"])
    df = handle_missing_values(df, strategy=config["preprocessing"]["strategy"])

    # turn the raw 1-10 stress score into low/moderate/high classes
    df = bucket_stress_level(df, config["data"]["target_column"])

    df = encode_categoricals(df, columns=config["preprocessing"]["categorical_columns"])

    target_col = config["data"]["target_column"]
    X = df.drop(columns=[target_col, "user_id"])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
    )

    mlflow.set_experiment("stress-level-prediction")

    with mlflow.start_run():

        # log all hyperparameters from config
        mlflow.log_param("n_estimators", config["model"]["n_estimators"])
        mlflow.log_param("max_depth", config["model"]["max_depth"])
        mlflow.log_param("missing_strategy", config["preprocessing"]["strategy"])
        mlflow.log_param("test_size", config["training"]["test_size"])
        mlflow.log_param("random_state", config["training"]["random_state"])

        # log the data version (DVC hash or label from config)
        mlflow.log_param("data_version", config["data"]["version"])

        # train a simple model, hyperparameters come from config
        model = RandomForestClassifier(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            random_state=config["training"]["random_state"],
        )
        model.fit(X_train, y_train)

        # evaluate and log at least 3 metrics
        metrics = evaluate_model(model, X_test, y_test)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # log trained model as an MLflow artifact
        mlflow.sklearn.log_model(model, "model")

        print(
            f"accuracy: {metrics['accuracy']:.4f}, "
            f"f1: {metrics['f1_score']:.4f}, "
            f"precision: {metrics['precision']:.4f}"
        )

        # exit with code 1 if the model doesn't meet the minimum threshold
        min_accuracy = config["training"]["min_accuracy_threshold"]
        if metrics["accuracy"] < min_accuracy:
            print(f"FAILED: accuracy {metrics['accuracy']:.4f} is below threshold {min_accuracy}")
            sys.exit(1)


if __name__ == "__main__":
    train()