import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from ..utils.config_validations import ModelConfig as ConfigValidator


class Training:
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        validator = ConfigValidator(**config["model"])

        # valide edilen config
        self.config = validator.model_dump()
        self.model = RandomForestClassifier(self.config["hyperparameters"])

    def train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int
    ) -> None:
        """Methods to train the model and print the cross validation scores.
        Args:
            X_train: Training data
            y_train: Training labels
            cv_folds: Number of cross validation folds

        Returns:
            None
        """
        mean_scores = np.mean(cross_val_score(self.model, X_train, y_train, cv=cv_folds))
        print(f"{cv_folds} CV mean scores: {mean_scores}")
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Method to evaluate the model on testdata.
        Args:
            X_test: Test data
            y_test: Test labels

        Returns:
            None
        """
        pass

    def save_results(self):
        pass
