from src.preprocessing import (
    load_config,
    validate_dataframe,
    simulate_missing_values,
    handle_missing_values,
    encode_categoricals,
)

def train():
    config = load_config("configs/params.yaml")

    df = pd.read_csv(config["data"]["path"])

    # dataset is clean, so simulate missing values
    df = simulate_missing_values(
        df,
        columns=config["preprocessing"]["missing_columns"],
        rate=config["preprocessing"]["missing_rate"],
    )

    validate_dataframe(df, config["data"]["required_columns"])
    df = handle_missing_values(df, strategy=config["preprocessing"]["strategy"])
    df = encode_categoricals(df, columns=config["preprocessing"]["categorical_columns"])