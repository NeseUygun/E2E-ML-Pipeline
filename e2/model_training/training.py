import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ..utils.config_validations import ModelsConfig as ConfigValidator
from ..utils.logging_utils import get_logger

UNWANTED_METRICS = ["accuracy", "macro avg", "weighted avg"]

model_mapping = {
    "RandomForestClassifier": RandomForestClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
}


class TrainModel:
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        validator = ConfigValidator(**config["model"])

        # validated config
        self.config = validator.model_dump()

        self.logger = get_logger(__name__, "logs/log_details.log")

    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        preprocess_pipeline: ColumnTransformer,
        model_save_path: str,
    ) -> None:
        """Methods to train the model in the YAML config, print the cross
        validation scores and save the model.

        Args:
            X_train: Training data
            y_train: Training labels
            preprocess_pipeline: Preprocessing pipeline of the training data
            model_save_path: Path to save the model

        Returns:
            None
        """
        for model in self.config["model"]:
            print(model["name"], "--", model["hyperparameters"])
            print()

    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        preprocess_pipeline: ColumnTransformer,
        # model_pipeline: Pipeline,
        # parameters_training: Dict[str, Union[int, float, bool]],
        # hyperparameters: Dict[str, Union[
        #    int, List[int], float, List[float], str, List[str], bool, List[bool]]],
        model_save_path: str,
    ) -> None:
        """Methods to train the model and print the cross validation scores.
        Args:
            X_train: Training data
            y_train: Training labels
            preprocess_pipeline: Preprocessing pipeline
                model_pipeline: Model pipeline
                parameters_training: Training parameters such as cv_folds,
                hyperparameters: Model hyperparametersi can be a list of values or a
                single value. If a list of values is provided, optuna will be used to
                tune the hyperparameters.
            model_save_path: Path to save the model

        Returns:
            None
        """
        # optimization_required = any(
        #    isinstance(value, list) for value in hyperparameters.values()
        # )

        self.logger.info("Training the model.")
        cv_folds = self.config["parameters_training"].get("cv_fold", None)

        model_pipeline = Pipeline(
            steps=[("preprocessor", preprocess_pipeline), ("model", self.model)]
        )

        if cv_folds != 0:
            if cv_folds < 2:
                error_msg = "\tPlease provide a value greater than 1 for cv_folds."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            overfitting_threshold = self.config["parameters_training"].get(
                "overfitting_threshold", 0.1
            )

            cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=cv_folds)
            print(f"{cv_folds} CV mean scores: {np.mean(cv_scores)}")

            std_scores = np.std(cv_scores)
            if std_scores > overfitting_threshold:
                error_msg = (
                    f"Model is overfitting with std: {std_scores} > "
                    f"threshold: {overfitting_threshold} in {cv_folds} CV folds."
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        pipeline = model_pipeline.fit(X_train, y_train)

        # save the pipeline
        with open(model_save_path, "wb") as file:
            pickle.dump(pipeline, file)
        self.logger.info("Model training completed.")

    def evaluate_model(
        self, X_test: pd.DataFrame, y_test: pd.Series, save_path: str
    ) -> None:
        """Method to evaluate the model on the test data.
        Args:
            X_test: Test data
            y_test: Test labels
            save_path: Path to save the evaluation metrics

        Returns:
            None
        """

        self.logger.info("Evaluating the model.")

        try:
            predictions = self.model.predict(X_test)
        except NotFittedError:
            error_msg = "\tModel not fitted. Please train the model first."
            self.logger.error(error_msg)
            raise NotFittedError(error_msg)

        self.logger.info("Retrieved the predictions.")
        metrics = classification_report(y_test, predictions, output_dict=True, digits=4)
        metrics = pd.DataFrame(metrics).transpose()

        # remove unwanted metrics using index.
        metrics = metrics[~metrics.index.isin(UNWANTED_METRICS)]
        self.logger.info("Evaluation metrics computed.")

        if save_path is not None:
            if not save_path.endswith(".csv"):
                error_msg = "Please provide a valid csv file path."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            self.save_results(metrics, save_path)

    def save_results(self, metrics: pd.DataFrame, save_path: str) -> None:
        """Method to save the model evaluation metrics
        Args:
            metrics: Evaluation metrics
            save_path: Path to save the evaluation metrics
        Returns:
            None
        """

        self.logger.info("Saving the evaluation metrics.")
        metrics.to_csv(save_path, index=True)
