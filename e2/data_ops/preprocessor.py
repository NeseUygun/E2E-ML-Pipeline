from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from e2.utils.logging_utils import get_logger

NUMERIC_FEATURES = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
]
CATEGORICAL_FEATURES = [
    "Month",
    "Occupation",
    "Type_of_Loan",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]

LABEL = ["Credit_Score"]

TO_BE_DROPPED = ["ID", "SSN", "Customer_ID", "Name"]

MATCH_GIBBERISH = r"[!@#$]"


class Preprocessor:
    def __init__(self, data):
        self.data = data
        self.logger = get_logger(__name__, "logs/log_details.log")

    def create_processing_pipeline(self) -> ColumnTransformer:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        category_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=5),
                ),
            ]
        )

        preprocessing_pipeline = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, NUMERIC_FEATURES),
                ("category", category_transformer, CATEGORICAL_FEATURES),
            ]
        )

        return preprocessing_pipeline

    def clean_and_split_data(self, test_size: float) -> Union[pd.DataFrame, Tuple]:
        self.logger.info("Preprocessing data.")

        if TO_BE_DROPPED:
            self.data.drop(TO_BE_DROPPED, axis=1, inplace=True)

        self._process_numeric_features()
        self._process_categoric_features()
        self.logger.info("Data preprocessing completed.")

        if test_size:
            self.logger.info("Splitting the data into train and test sets.")
            X_train, X_test, y_train, y_test = self._split_data(test_size)
            return X_train, X_test, y_train, y_test

        return self.data

    def _process_numeric_features(self):
        all_object_cols = set(self.data.select_dtypes(include="object").columns)
        numeric_cols = set(NUMERIC_FEATURES)

        should_be_float_but_object = numeric_cols.intersection(all_object_cols)

        # "Credit_History_Age" is a special column to be processed later.
        should_be_float_but_object = list(
            should_be_float_but_object - {"Credit_History_Age"}
        )

        if should_be_float_but_object:
            # Replace "_" with "" in the values.
            # TODO: Only replace "_" in the columns that has "_".
            self.data[should_be_float_but_object] = self.data[
                should_be_float_but_object
            ].replace("_", "", regex=True)

            # If the values are not numeric, convert them to numeric or NaN.
            self.data[should_be_float_but_object] = self.data[
                should_be_float_but_object
            ].apply(pd.to_numeric, errors="coerce")

        # Use regular expressions to extract years and months into separate Series
        years_series = (
            self.data["Credit_History_Age"]
            .str.extract(r"(\d+)\s*Years")
            .astype(float)
            .fillna(0)
        )
        months_series = (
            self.data["Credit_History_Age"]
            .str.extract(r"(\d+)\s*Months")
            .astype(float)
            .fillna(0)
        )

        # calculate the total number of months
        self.data["Credit_History_Age"] = years_series * 12 + months_series
        self.data.loc[self.data["Age"] < 0, "Age"] = np.nan

    def _process_categoric_features(self):
        special_chars_in_pay_behaviour = self.data["Payment_Behaviour"].str.contains(
            MATCH_GIBBERISH, regex=True
        )
        self.data.loc[special_chars_in_pay_behaviour, "Payment_Behaviour"] = "Unknown"

        for col in self.data.select_dtypes(include="object").columns:
            val_counts = self.data[col].value_counts(dropna=False, normalize=True) * 100

            to_replace = val_counts[val_counts < 2].index
            if len(to_replace) > 0:
                print(f"{col} has {len(to_replace)} values less than 2%")
            self.data[col] = self.data[col].replace(to_replace, "Other")

    def _split_data(self, test_size):
        X_train, X_test, y_train, y_test = train_test_split(
            self.data.drop(LABEL, axis=1),
            self.data[LABEL],
            test_size=test_size,
            random_state=42,
        )

        return X_train, X_test, y_train, y_test
