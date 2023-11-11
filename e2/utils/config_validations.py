import typing
from typing import Dict, Union

from pydantic import BaseModel


class ModelParameters(BaseModel):
    __annotations__ = Dict[str, typing.Union[int, float]]


class ParametersTraining(BaseModel):
    cv_folds: Union[int, None]
    overfitting_threshold: float
    optimization: bool


class ModelConfig(BaseModel):
    name: str
    hyperparameters: ModelParameters
    parameters_training: ParametersTraining


class ModelsConfig(BaseModel):
    RandomForestClassifier: ModelConfig
    DecisionTreeClassifier: ModelConfig
