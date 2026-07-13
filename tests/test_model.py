import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import load_config, handle_missing_values, encode_categoricals
from src.evaluate import evaluate_model


def get_small_sample():
    config = load_config("configs/params.yaml")
    df = pd.read_csv(config["data"]["path"])

    df = df.sample(n=200, random_state=42)

    df = handle_missing_values(df, strategy=config["preprocessing"]["strategy"])
    df = encode_categoricals(df, columns=config["preprocessing"]["categorical_columns"])

    target_col = config["data"]["target_column"]
    X = df.drop(columns=[target_col, "user_id"])
    y = df[target_col]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_model_predictions_correct_type_and_shape():
    X_train, X_test, y_train, y_test = get_small_sample()

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]


def test_model_meets_minimum_performance_threshold():
    X_train, X_test, y_train, y_test = get_small_sample()

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    min_threshold = 0.3
    assert metrics["accuracy"] >= min_threshold, (
        f"Accuracy {metrics['accuracy']:.4f} below minimum {min_threshold}"
    )