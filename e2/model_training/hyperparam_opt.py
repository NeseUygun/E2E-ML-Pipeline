from dataclasses import dataclass

import mlflow
import numpy as np
from sklearn.model_selection import cross_val_score

# from sklearn.pipeline import Pipeline

mlflow.set_experiment("end-to-end-ml")


@dataclass
class OptunaConfig:
    hyperparameters: dict
    cv_folds: int
    scoring: str

    def get_objective(self, X_train, y_train, model_pipeline):
        def objective(trial):
            trial_hyperparameters = self.get_hyperparameters(trial)
            return self.compute_score(
                trial_hyperparameters, X_train, y_train, model_pipeline
            )

        mlflow.sklearn.log_model(model_pipeline[-1][-1], "model")
        return objective

    def get_hyperparameters(self, trial) -> dict:
        trial_hyperparameters = {}
        for param_name, values in self.hyperparameters.items():
            if isinstance(values, list):
                if all(isinstance(value, int) for value in values):
                    trial_hyperparameters[param_name] = trial.suggest_int(
                        param_name, values[0], values[1]
                    )
                elif all(isinstance(value, float) for value in values):
                    trial_hyperparameters[param_name] = trial.suggest_float(
                        param_name, values[0], values[1]
                    )
                else:
                    trial_hyperparameters[param_name] = trial.suggest_categorical(
                        param_name, values
                    )
        for param_name, values in trial_hyperparameters.items():
            mlflow.log_param(param_name, values)

        return trial_hyperparameters

    def compute_score(
        self, trial_hyperparameters, X_train, y_train, model_pipeline
    ) -> float:
        model_pipeline.steps[-1][-1].set_params(**trial_hyperparameters)
        mean_scores = np.mean(
            cross_val_score(
                model_pipeline,
                X_train,
                y_train,
                cv=self.cv_folds,
                scoring=self.scoring,
            )
        )
        mlflow.log_metric(self.scoring, mean_scores)
        return mean_scores
