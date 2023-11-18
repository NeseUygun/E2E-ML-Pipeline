import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_data(data: pd.DataFrame, expected_cols) -> None:
    """Validated the data against the expected columns.

    Args:
        data: Data to be validated.
        expected_cols: Expected columns.

    Returns:
        None

    Raises:
        ValueError: If the data contains unexpected columns.
    """
    data_columns = data.columns.tolist()

    for column in data_columns:
        if column not in expected_cols:
            error_msg = (
                f"Unexpected column: {column}",
                f"Expected columns: {expected_cols}",
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
