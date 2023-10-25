import os

import pandas as pd

ALLOWED_EXTENSIONS = [".csv"]


class DatasetIO:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.__validate_parameters()

    def read_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.data_path)
        self.validate_data(data)
        return data

    def save_data(self, data: pd.DataFrame, save_path: str) -> None:
        data.to_csv(save_path, index=False)

    def validate_data(self, data: pd.DataFrame):
        expected_columns = [
            "sepal.length",
            "sepal.width",
            "petal.length",
            "petal.width",
            "variety",
        ]
        data_columns = data.columns.tolist()

        for column in data_columns:
            if column not in expected_columns:
                raise ValueError(
                    f"Unexpected column: {column}",
                    f"Expected columns: {expected_columns}",
                )

    def __validate_parameters(self):
        if not isinstance(self.data_path, str):
            raise TypeError(f"Expected data_path as str, found {type(self.data_path)}")

        data_ext = os.path.splitext(self.data_path)[-1]
        if data_ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Expected csv file but found {data_ext}")
