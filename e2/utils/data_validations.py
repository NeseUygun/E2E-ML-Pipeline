import pandas as pd


def validate_data(data: pd.DataFrame, expected_cols):
    data_columns = data.columns.tolist()

    for column in data_columns:
        if column not in expected_cols:
            raise ValueError(
                f"Unexpected column: {column}",
                f"Expected columns: {expected_cols}",
            )
