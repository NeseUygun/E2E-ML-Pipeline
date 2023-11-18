from typing import Union

from pydantic import BaseModel, Extra


class ModelParameters(BaseModel):
    class Config:
        extra = Extra.allow


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
