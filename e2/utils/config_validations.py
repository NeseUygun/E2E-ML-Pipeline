from typing import Union

from pydantic import BaseModel


class ModelParameters(BaseModel):
    max_depth: int
    random_state: int
    n_estimators: Union[int, None] = None


class ParametersTraining(BaseModel):
    cv_folds: Union[int, None]
    overfitting_threshold: float
    optimization: bool


class ModelConfig(BaseModel):
    hyperparameters: ModelParameters
    parameters_training: ParametersTraining


class ModelsConfig(BaseModel):
    RandomForestClassifier: ModelConfig
    DecisionTreeClassifier: ModelConfig
