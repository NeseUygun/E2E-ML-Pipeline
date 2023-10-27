import os

import pandas as pd

from e2.utils.data_validations import validate_data

ALLOWED_EXTENSIONS = [".csv", ".txt", ".parquet", ".xlsx", ".json"]
EXPECTED_COLS = [
    "sepal.length",
    "sepal.width",
    "petal.length",
    "petal.width",
    "variety",
]

DATA_READ_MAPPING = {
    ".txt": pd.read_csv,
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".xlsx": pd.read_excel,
    ".json": pd.read_json,
}

SAVE_DATA_MAPPING = {
    ".txt": "to_csv",
    ".csv": "to_csv",
    ".parquet": "to_parquet",
    ".xlsx": "to_excel",
}


class DatasetIO:
    """Class for Dataset I/O operations."""

    def __init__(self, data_path: str):
        """Constructor the for the class."""
        self.data_path = data_path
        self.__validate_parameters()

    def read_data(self) -> pd.DataFrame:
        """Reads the data from the path.

        Returns:
              The file, read as pandas Dataframe.

        Raises:
            - ValueError: If the extension is not supported.
        """

        extension = os.path.splitext(self.data_path)[-1]

        read_func = DATA_READ_MAPPING.get(extension, None)
        if read_func is None:
            raise ValueError(f"Received invalid extension, found {extension}.")

        data = read_func(self.data_path)

        validate_data(data, EXPECTED_COLS)

        return data

    def save_data(self, data: pd.DataFrame, save_path: str) -> None:
        save_path_ext = os.path.splitext(save_path)[-1]

        if not save_path_ext:
            # If save_path param does not contain the save extension,
            # we assume user wants to save as the same extension while given in the
            # reading process.
            save_path = save_path + os.path.splitext(self.data_path)[-1]

        save_path_ext = os.path.splitext(save_path)[-1]
        save_func = SAVE_DATA_MAPPING.get(save_path_ext)
        save_func = getattr(data, save_func)

        save_func(save_path, index=False)

    def __validate_parameters(self):
        if not isinstance(self.data_path, str):
            raise TypeError(f"Expected data_path as str, found {type(self.data_path)}")

        data_ext = os.path.splitext(self.data_path)[-1]
        if data_ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Expected csv file, found {data_ext}")
