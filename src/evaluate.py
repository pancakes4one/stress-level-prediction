from sklearn.metrics import accuracy_score, f1_score, precision_score


def evaluate_model(model, X_test, y_test):
    # run the trained model on the test set
    preds = model.predict(X_test)

    # calculate the metrics we care about
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    precision = precision_score(y_test, preds, average="macro")

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
    }

    return metrics