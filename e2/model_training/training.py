import pickle
from typing import Dict, List, Union

import mlflow.sklearn
import optuna
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from e2.model_training.hyperparam_opt import OptunaConfig

# from e2.utils.config_validations import ModelsConfig as ConfigValidator
from e2.utils.logging_utils import get_logger

UNWANTED_METRICS = ["accuracy", "macro avg", "weighted avg"]

model_mapping = {
    "RandomForestClassifier": RandomForestClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
}

mlflow.set_experiment("end-to-end-ml")


class TrainModel:
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        # validator = ConfigValidator(**config["model"])

        # validated config
        self.config = config["model"]  # validator.model_dump()

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
        for model in self.config:
            hyperparameters = self.config.get(model).get("hyperparameters", None)
            current_model = model_mapping.get(model, None)()
            self.logger.info(f"Training model: {model}")

            if current_model is None or hyperparameters is None:
                error_msg = (
                    f"\tModel {model} not in the model mapping or"
                    f"hyperparameters not found in the config."
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            model_pipeline = Pipeline(
                steps=[("preprocessor", preprocess_pipeline), ("model", current_model)]
            )
            self._train_model(
                X_train=X_train,
                y_train=y_train,
                model_pipeline=model_pipeline,
                parameters_training=self.config[model]["parameters_training"],
                hyperparameters=hyperparameters,
                model_save_path=model_save_path,
            )
            self.logger.info(f"\tModel {model} training completed.")
        self.logger.info("All models training completed.")

    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_pipeline: Pipeline,
        parameters_training: Dict[str, Union[int, float, bool]],
        hyperparameters: Dict[
            str,
            Union[int, List[int], float, List[float], str, List[str], bool, List[bool]],
        ],
        model_save_path: str,
    ) -> None:
        """Methods to train the model and print the cross validation scores.
        Args:
            X_train: Training data
            y_train: Training labels
            model_pipeline: Model pipeline
            parameters_training: Training parameters such as cv_folds, overfitting
                threshold
            hyperparameters: Model hyperparametersi can be a list of values or a
                single value. If a list of values is provided, optuna will be used to
                tune the hyperparameters.
            model_save_path: Path to save the model

        Returns:
            None
        """
        self.logger.info("Training the model.")

        optimization_required = any(
            isinstance(value, list) for value in hyperparameters.values()
        )

        cv_folds = parameters_training.get("cv_folds", None)

        if cv_folds != 0:
            if cv_folds < 2:
                error_msg = "\tPlease provide a value greater than 1 for cv_folds."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        # overfitting_threshold = parameters_training.get("overfitting_threshold", 0.1)

        if optimization_required:
            optuna_config = OptunaConfig(
                hyperparameters=hyperparameters,
                cv_folds=cv_folds,
                scoring="f1_macro",
            )

            objective = optuna_config.get_objective(
                X_train=X_train,
                y_train=y_train,
                model_pipeline=model_pipeline,
            )
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=10)

            best_parameters = study.best_params
            model_pipeline.set_params(**best_parameters)
        else:
            model_pipeline.steps[-1][-1].set_params(**hyperparameters)

        pipeline = model_pipeline.fit(X_train, y_train)

        # save the pipeline
        self.model_save(model_save_path, pipeline)

        self.logger.info("Model training completed.")

    def model_save(self, model_save_path, pipeline):
        with open(model_save_path, "wb") as file:
            pickle.dump(pipeline, file)

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
            metrics.to_csv(save_path, index=True)
